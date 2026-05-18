# Multi-Expert Pipeline FP8 Direct Scaled Dot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fp8 e4m3 `direct_scaled_dot` support to `kernels/multi_expert_pipeline.py`, keeping flat `(bt, d)` shapes (no `t_packing`), with a combined gate+up scale-group loop, plus a MiMo V2 Pro benchmark YAML config.

**Architecture:** Extend the existing bf16 multi-expert pipeline kernel with an fp8 compute path that performs per-scale-group dot products (`x[bt, qbk] @ w[qbk, bf]` in fp8→f32) followed by f32 scale multiplication. Scales are loaded via DMA alongside weights using shared semaphores. Gate and up are computed together in a single `lax.fori_loop` over `n_sg` scale groups. FFN2 applies activation per scale group before the down-projection dot. The kernel API gains `w1_scale`, `w2_scale`, `w3_scale`, and `quant_block_k` parameters; when scales are `None`, the existing bf16 path runs unchanged.

**Tech Stack:** JAX, Pallas TPU, `jnp.float8_e4m3fn`, `lax.fori_loop`

**Spec:** `docs/superpowers/specs/2026-05-18-fused-moe-v3-design.md`
**Base code:** `kernels/multi_expert_pipeline.py` (current bf16-only, 408 lines)
**Reference:** `kernels/_fused_moe_v2_impl.py` (v2 fp8 direct_scaled_dot at lines 1149-1338)

---

## Key Design Decisions

1. **Flat shapes, no `t_packing`**: Tokens stay `(bt, d)`, weights stay `(d, bf)` / `(bf, d)`. Scale groups iterate over `n_sg = d // qbk` (not `h_per_t // qbk`). This avoids reshape complexity while the kernel targets single-device use.

2. **Combined gate+up path**: A single `lax.fori_loop` computes both gate and up accumulators per scale group (no bt-size branching). This maximizes MXU utilization since both W1 and W3 are already loaded.

3. **FFN2 per-scale-group activation**: When using fp8 W2 with scales, activation is applied per `(bt, qbk)` slice before the W2 dot, matching the v2 `direct_scaled_dot` pattern.

4. **Scale DMA shares weight semaphores**: Each scale is fetched on the same semaphore as its corresponding weight (e.g., `weight_sems[slot, 0]` for both W1 and W1_scale), matching v2's proven pattern. A single `wait` drains both the weight and its scale.

5. **Scales can fit in SMEM** (per spec §4.3): For MiMo (d=6144, qbk=128, bf=512), each scale tile is `(d//qbk) × 1 × bf × 4B = 48 × 512 × 4 = 96 KB`. With 3 weights × 2 slots = ~576 KB total — well within VMEM/SMEM budget. We use VMEM scratch (matching v2's approach).

6. **Gate/up accumulators as scratch**: New `b_gate_acc_vmem` and `b_up_acc_vmem` scratch buffers `(bt, bf)` f32 replace the inline gate/up locals from the bf16 path, needed because fp8 accumulates across scale groups.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `kernels/multi_expert_pipeline.py` | Modify | Add fp8 compute path, scale DMA, scratch buffers, update `kernel_fn` and `config` |
| `tests/test_multi_expert_pipeline.py` | Modify | Add structural tests for fp8 API surface (scales params, config entries) |
| `sweep_multi_expert_mimo_v2_fp8.yaml` | Create | Benchmark config for real fp8 MiMo V2 Pro (no bf16 approximation) |
| `scripts/sweep_multi_expert_device.py` | Modify | Add `mimo-v2-fp8` profile with real MiMo dimensions |

---

## Chunk 1: FP8 Kernel Implementation

### Task 1: Add structural tests for fp8 API surface

**Files:**
- Modify: `tests/test_multi_expert_pipeline.py`

- [ ] **Step 1: Add tests for fp8 config entries and API params**

Add to the existing test file:

```python
class TestFp8Config:
    @pytest.fixture(autouse=True)
    def _extract_config(self):
        source = KERNEL_FILE.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "config":
                        module = ast.Module(body=[node], type_ignores=[])
                        ast.fix_missing_locations(module)
                        code = compile(module, str(KERNEL_FILE), "exec")
                        ns = {}
                        exec(code, ns)
                        self.config = ns["config"]
                        return
        pytest.fail("config not found")

    def test_config_has_quant_block_k(self):
        assert "quant_block_k" in self.config
        assert self.config["quant_block_k"] == 128


class TestFp8MultiExpertFfnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_multi_expert_ffn_accepts_scale_params(self):
        sig = inspect.signature(self.module.multi_expert_ffn)
        params = set(sig.parameters.keys())
        for p in ("w1_scale", "w2_scale", "w3_scale", "quant_block_k"):
            assert p in params, f"multi_expert_ffn missing param: {p}"

    def test_kernel_fn_accepts_fp8_params(self):
        sig = inspect.signature(self.module.kernel_fn)
        params = set(sig.parameters.keys())
        for p in ("quant_block_k",):
            assert p in params, f"kernel_fn missing param: {p}"


class TestFp8RefSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_ref_accepts_scale_params(self):
        sig = inspect.signature(self.module._ref_multi_expert_ffn)
        params = set(sig.parameters.keys())
        for p in ("w1_scale", "w2_scale", "w3_scale", "quant_block_k"):
            assert p in params, f"_ref_multi_expert_ffn missing param: {p}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multi_expert_pipeline.py -v -k "Fp8"`
Expected: FAIL — `multi_expert_ffn` and `kernel_fn` don't have scale params yet.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_multi_expert_pipeline.py
git commit -m "test: add structural tests for fp8 API surface in multi_expert_pipeline"
```

---

### Task 2: Add fp8 parameters to kernel API and config

**Files:**
- Modify: `kernels/multi_expert_pipeline.py`

- [ ] **Step 1: Update the module `config` dict**

Add `quant_block_k` to the config:

```python
config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
        "num_experts": 8,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 256,
    "quant_block_k": 128,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Multi-expert double-buffer FFN — EP decode "
        "(N experts sequential, expert-to-expert DMA overlap)"
    ),
}
```

- [ ] **Step 2: Add scale params to `multi_expert_ffn` signature**

Update the function signature to accept optional scale arrays and `quant_block_k`:

```python
@functools.partial(jax.jit, static_argnames=["act_fn", "bf", "num_experts", "quant_block_k"])
def multi_expert_ffn(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    quant_block_k: int | None = None,
) -> jax.Array:
```

Add validation for fp8 params inside the function body (after existing validation):

```python
    use_fp8 = w1_scale is not None
    if use_fp8:
        if w2_scale is None or w3_scale is None:
            raise ValueError("All three scales must be provided for fp8")
        if quant_block_k is None:
            raise ValueError("quant_block_k required when scales are provided")
        if d % quant_block_k != 0:
            raise ValueError(
                f"hidden_size={d} must be divisible by quant_block_k={quant_block_k}"
            )
        if f_full % quant_block_k != 0:
            raise ValueError(
                f"intermediate_size={f_full} must be divisible by quant_block_k={quant_block_k}"
            )
```

- [ ] **Step 3: Add scale params to `_ref_multi_expert_ffn`**

Update the reference function to handle fp8 with scales:

```python
def _ref_multi_expert_ffn(
    tokens, w1, w2, w3, *,
    act_fn="silu",
    w1_scale=None, w2_scale=None, w3_scale=None,
    quant_block_k=None,
):
    """Pure-JAX reference: loop over experts, each does (silu(x@W1) * (x@W3)) @ W2."""
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        if w1_scale is not None:
            # Per-scale-group matmul for fp8
            d = x.shape[-1]
            f = w1.shape[-1]
            n_sg = d // quant_block_k
            gate = jnp.zeros((x.shape[0], f), dtype=jnp.float32)
            up = jnp.zeros_like(gate)
            for sg in range(n_sg):
                sg_off = sg * quant_block_k
                x_slice = x[:, sg_off:sg_off + quant_block_k]
                w1_tile = w1[e, sg_off:sg_off + quant_block_k, :].astype(jnp.float32)
                w3_tile = w3[e, sg_off:sg_off + quant_block_k, :].astype(jnp.float32)
                s1 = w1_scale[e, sg:sg+1, 0, :].reshape(1, f)
                s3 = w3_scale[e, sg:sg+1, 0, :].reshape(1, f)
                gate += (x_slice @ w1_tile) * s1
                up += (x_slice @ w3_tile) * s3
            act = activation_fn(gate, up, act_fn)
            # FFN2: per-scale-group
            n_sg2 = f // quant_block_k
            out = jnp.zeros((x.shape[0], d), dtype=jnp.float32)
            for sg in range(n_sg2):
                sg_off = sg * quant_block_k
                act_slice = act[:, sg_off:sg_off + quant_block_k]
                w2_tile = w2[e, sg_off:sg_off + quant_block_k, :].astype(jnp.float32)
                s2 = w2_scale[e, sg:sg+1, 0, :].reshape(1, d)
                out += (act_slice @ w2_tile) * s2
            outputs.append(out.astype(tokens.dtype))
        else:
            gate = x @ w1[e].astype(jnp.float32)
            up = x @ w3[e].astype(jnp.float32)
            act = activation_fn(gate, up, act_fn)
            out = (act @ w2[e].astype(jnp.float32)).astype(tokens.dtype)
            outputs.append(out)
    return jnp.stack(outputs)
```

- [ ] **Step 4: Add scale params to `kernel_fn`**

Update `kernel_fn` to optionally create fp8 weights and scales:

```python
def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int = 8,
    quant_block_k: int = 128,
    **_kwargs,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts, num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w2 = jax.random.normal(k3, (num_experts, intermediate_size, hidden_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w3 = jax.random.normal(k4, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)

    w1_scale = w2_scale = w3_scale = None
    qbk = None
    if jnp.dtype(weight_dtype) == jnp.float8_e4m3fn:
        n_sg_h = hidden_size // quant_block_k
        n_sg_i = intermediate_size // quant_block_k
        w1_scale = jnp.ones((num_experts, n_sg_h, 1, intermediate_size), dtype=jnp.float32) * 0.01
        w2_scale = jnp.ones((num_experts, n_sg_i, 1, hidden_size), dtype=jnp.float32) * 0.01
        w3_scale = jnp.ones((num_experts, n_sg_h, 1, intermediate_size), dtype=jnp.float32) * 0.01
        qbk = quant_block_k

    def run():
        return multi_expert_ffn(
            tokens, w1, w2, w3,
            act_fn=act_fn, bf=bf, num_experts=num_experts,
            w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
            quant_block_k=qbk,
        )

    return run
```

- [ ] **Step 5: Run structural tests to verify they pass**

Run: `pytest tests/test_multi_expert_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add kernels/multi_expert_pipeline.py tests/test_multi_expert_pipeline.py
git commit -m "feat: add fp8 scale params to multi_expert_pipeline API (no kernel logic yet)"
```

---

### Task 3: Implement fp8 kernel body — scale DMA and scratch buffers

**Files:**
- Modify: `kernels/multi_expert_pipeline.py`

This is the core implementation task. Changes are in three areas: (A) scratch buffer allocation in `multi_expert_ffn`, (B) DMA helpers in `_multi_expert_kernel`, (C) compute logic in `compute_tile`.

- [ ] **Step 1: Add scale HBM inputs and scratch buffers to `multi_expert_ffn`**

In the `multi_expert_ffn` function, update the pallas_call setup to pass scale HBM arrays and allocate scale VMEM scratch + gate/up accumulator scratch.

Update the kernel signature to accept scale HBM refs:

```python
def _multi_expert_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    w1_scale_hbm,   # NEW: None or (num_experts, d//qbk, 1, f) f32
    w2_scale_hbm,   # NEW: None or (num_experts, f//qbk, 1, d) f32
    w3_scale_hbm,   # NEW: None or (num_experts, d//qbk, 1, f) f32
    output_hbm,
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_w1_scale_x2_vmem,  # NEW: None or scratch (2, d//qbk, 1, bf) f32
    b_w3_scale_x2_vmem,  # NEW
    b_w2_scale_x2_vmem,  # NEW: None or scratch (2, f//qbk, 1, d) f32 — but note we only need bf-sized slices
    b_x_x2_vmem,
    b_y_acc_vmem,
    b_y_out_vmem,
    b_gate_acc_vmem,  # NEW: None or scratch (bt, bf) f32
    b_up_acc_vmem,    # NEW: None or scratch (bt, bf) f32
    weight_sems,
    x_sem,
    y_out_sem,
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
    num_experts: int,
    quant_block_k: int | None,
):
```

Update scratch_shapes in `multi_expert_ffn`:

```python
    n_sg = d // quant_block_k if use_fp8 else 0
    n_sg2 = f_full // quant_block_k if use_fp8 else 0

    scratch_shapes = (
        pltpu.VMEM((2, d, bf), weight_dtype),            # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),             # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),             # b_w2_x2_vmem
        # Scale scratch — (2, n_sg, 1, tile_out_dim) f32, or None
        pltpu.VMEM((2, n_sg, 1, bf), jnp.float32) if use_fp8 else None,    # b_w1_scale
        pltpu.VMEM((2, n_sg, 1, bf), jnp.float32) if use_fp8 else None,    # b_w3_scale
        pltpu.VMEM((2, n_sg2, 1, d), jnp.float32) if use_fp8 else None,    # b_w2_scale
        pltpu.VMEM((2, bt, d), dtype),                    # b_x_x2_vmem
        pltpu.VMEM((bt, d), jnp.float32),                 # b_y_acc_vmem
        pltpu.VMEM((2, bt, d), dtype),                    # b_y_out_vmem
        # Gate/up accumulators for fp8 (bf-tiled, accumulated across scale groups)
        pltpu.VMEM((bt, bf), jnp.float32) if use_fp8 else None,  # b_gate_acc
        pltpu.VMEM((bt, bf), jnp.float32) if use_fp8 else None,  # b_up_acc
        pltpu.SemaphoreType.DMA((2, 3)),                  # weight_sems
        pltpu.SemaphoreType.DMA((2,)),                    # x_sem
        pltpu.SemaphoreType.DMA((2,)),                    # y_out_sem
    )
```

Update in_specs to include scale HBM arrays:

```python
    in_specs = [hbm, hbm, hbm, hbm,
                hbm if use_fp8 else None,   # w1_scale
                hbm if use_fp8 else None,   # w2_scale
                hbm if use_fp8 else None]   # w3_scale
```

And update the pallas_call inputs to pass scale arrays:

```python
    return jax.named_scope(scope_name)(kernel)(
        pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
        pltpu.with_memory_space_constraint(w1, pltpu.HBM),
        pltpu.with_memory_space_constraint(w2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w3, pltpu.HBM),
        *([pltpu.with_memory_space_constraint(s, pltpu.HBM) for s in (w1_scale, w2_scale, w3_scale)]
          if use_fp8 else []),
    )
```

- [ ] **Step 2: Add scale DMA helpers to `_multi_expert_kernel`**

Inside the kernel body, update `start_fetch_w1/w3/w2` and `wait_fetch_w1/w3/w2` to also transfer scales when present:

```python
    def start_fetch_w1(slot, expert_idx, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start(priority=priority)
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=w1_scale_hbm.at[expert_idx, :, pl.ds(0, 1), pl.ds(tile_idx * bf, bf)],
                dst_ref=b_w1_scale_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 0],
            ).start(priority=priority)

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w1_scale_x2_vmem.at[slot],
                dst_ref=b_w1_scale_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 0],
            ).wait()
```

Same pattern for w3 (semaphore channel 1) and w2 (semaphore channel 2). For w2, the scale HBM layout is `(num_experts, f//qbk, 1, d)` and we slice as:

```python
        if w2_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=w2_scale_hbm.at[expert_idx, pl.ds(tile_idx * bf // quant_block_k, bf // quant_block_k), pl.ds(0, 1), :],
                dst_ref=b_w2_scale_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 2],
            ).start(priority=priority)
```

- [ ] **Step 3: Implement fp8 `compute_tile` with direct scaled dot**

Replace the existing `compute_tile` with a version that branches on fp8:

```python
    n_sg = intermediate_size // quant_block_k if quant_block_k else 0
    n_sg2 = bf // quant_block_k if quant_block_k else 0

    def compute_tile(x_slot, w_slot, is_first_tile):
        x = b_x_x2_vmem[x_slot]

        if w1_scale_hbm is not None:
            # FP8 direct scaled dot — combined gate+up over scale groups
            gate = jnp.zeros((x.shape[0], bf), dtype=jnp.float32)
            up = jnp.zeros_like(gate)

            def _ffn1_sg_body(sg_id, carry):
                gate_acc, up_acc = carry
                sg_off = sg_id * quant_block_k
                x_slice = x[:, pl.ds(sg_off, quant_block_k)]

                w1_tile = b_w1_x2_vmem[w_slot, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                d1 = jnp.dot(x_slice, w1_tile, preferred_element_type=jnp.float32)
                s1 = b_w1_scale_x2_vmem[w_slot, pl.ds(sg_id, 1), 0, pl.ds(0, bf)].reshape(1, bf)
                gate_acc = gate_acc + d1 * jnp.broadcast_to(s1, d1.shape)

                w3_tile = b_w3_x2_vmem[w_slot, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                d3 = jnp.dot(x_slice, w3_tile, preferred_element_type=jnp.float32)
                s3 = b_w3_scale_x2_vmem[w_slot, pl.ds(sg_id, 1), 0, pl.ds(0, bf)].reshape(1, bf)
                up_acc = up_acc + d3 * jnp.broadcast_to(s3, d3.shape)

                return gate_acc, up_acc

            gate, up = lax.fori_loop(0, n_sg, _ffn1_sg_body, (gate, up), unroll=n_sg)
            b_gate_acc_vmem[...] = gate
            b_up_acc_vmem[...] = up

            wait_fetch_w2(w_slot)

            # FFN2: per-scale-group activation + down-projection
            n_sg2_local = bf // quant_block_k

            def _ffn2_sg_body(sg_id, partial_acc):
                sg_off = sg_id * quant_block_k
                gate_slice = b_gate_acc_vmem[:, pl.ds(sg_off, quant_block_k)]
                up_slice = b_up_acc_vmem[:, pl.ds(sg_off, quant_block_k)]
                act_slice = activation_fn(gate_slice, up_slice, act_fn)

                w2_tile = b_w2_x2_vmem[w_slot, pl.ds(sg_off, quant_block_k), :]
                d = jnp.dot(act_slice, w2_tile, preferred_element_type=jnp.float32)
                s = b_w2_scale_x2_vmem[w_slot, pl.ds(sg_id, 1), 0, :].reshape(1, d.shape[-1])
                return partial_acc + d * jnp.broadcast_to(s, d.shape)

            partial = lax.fori_loop(
                0, n_sg2_local, _ffn2_sg_body,
                jnp.zeros((x.shape[0], x.shape[1]), dtype=jnp.float32),
                unroll=n_sg2_local,
            )
        else:
            # Original bf16 path
            w1 = b_w1_x2_vmem[w_slot]
            w3 = b_w3_x2_vmem[w_slot]
            gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)
            up = jnp.dot(x, w3, preferred_element_type=jnp.float32)
            act_up = activation_fn(gate, up, act_fn)

            wait_fetch_w2(w_slot)

            w2 = b_w2_x2_vmem[w_slot]
            partial = jnp.dot(act_up, w2, preferred_element_type=jnp.float32)

        if is_first_tile:
            b_y_acc_vmem[...] = partial
        else:
            b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial
```

**Key detail for FFN2 output shape**: `b_w2_x2_vmem` has shape `(2, bf, d)`, so `w2_tile` is `(qbk, d)` and the dot output is `(bt, d)`, which matches `b_y_acc_vmem` shape `(bt, d)`. The scale shape for W2 is `(2, n_sg2, 1, d)` where `n_sg2 = f_full // qbk` in HBM, but in VMEM we load only the bf-tile's portion: `(2, bf//qbk, 1, d)`.

- [ ] **Step 4: Update `__main__` block for fp8 testing**

Update the `if __name__ == "__main__"` block to optionally test fp8:

```python
if __name__ == "__main__":
    num_experts_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bt = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    use_fp8 = "--fp8" in sys.argv
    bf_arg = 512 if use_fp8 else 256

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d, f = 6144 if use_fp8 else 8192, 2048
    w_dtype = jnp.float8_e4m3fn if use_fp8 else jnp.bfloat16
    qbk = 128 if use_fp8 else None

    tokens = jax.random.normal(k1, (num_experts_arg, bt, d), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (num_experts_arg, d, f), dtype=jnp.bfloat16).astype(w_dtype)
    w2 = jax.random.normal(k3, (num_experts_arg, f, d), dtype=jnp.bfloat16).astype(w_dtype)
    w3 = jax.random.normal(k4, (num_experts_arg, d, f), dtype=jnp.bfloat16).astype(w_dtype)

    w1_scale = w2_scale = w3_scale = None
    if use_fp8:
        n_sg_h = d // qbk
        n_sg_i = f // qbk
        w1_scale = jnp.ones((num_experts_arg, n_sg_h, 1, f), dtype=jnp.float32) * 0.01
        w2_scale = jnp.ones((num_experts_arg, n_sg_i, 1, d), dtype=jnp.float32) * 0.01
        w3_scale = jnp.ones((num_experts_arg, n_sg_h, 1, f), dtype=jnp.float32) * 0.01

    result = multi_expert_ffn(
        tokens, w1, w2, w3, bf=bf_arg, num_experts=num_experts_arg,
        w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
        quant_block_k=qbk,
    )
    ref = _ref_multi_expert_ffn(
        tokens, w1, w2, w3,
        w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
        quant_block_k=qbk,
    )

    max_errs = []
    rel_errs = []
    for e in range(num_experts_arg):
        r = result[e].astype(jnp.float32)
        f_ref = ref[e].astype(jnp.float32)
        me = jnp.max(jnp.abs(r - f_ref))
        re = me / (jnp.max(jnp.abs(f_ref)) + 1e-6)
        max_errs.append(float(me))
        rel_errs.append(float(re))
        print(f"  expert {e}: max_abs_err={me:.4f}, rel_err={re:.4f}")

    worst_rel = max(rel_errs)
    mode = "fp8" if use_fp8 else "bf16"
    print(f"[{mode}] num_experts={num_experts_arg}, bt={bt}, bf={bf_arg}, worst_rel_err={worst_rel:.4f}")
    assert worst_rel < 0.05, f"worst rel_err too high: {worst_rel}"
    print("PASS")
```

- [ ] **Step 5: Run structural tests to verify all pass**

Run: `pytest tests/test_multi_expert_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add kernels/multi_expert_pipeline.py
git commit -m "feat: implement fp8 direct_scaled_dot in multi_expert_pipeline kernel"
```

---

## Chunk 2: Benchmark Config and Sweep Profile

### Task 4: Create MiMo V2 Pro fp8 benchmark YAML and sweep profile

**Files:**
- Create: `sweep_multi_expert_mimo_v2_fp8.yaml`
- Modify: `scripts/sweep_multi_expert_device.py`

- [ ] **Step 1: Create the sweep YAML config**

```yaml
# Multi-expert pipeline sweep — MiMo V2 Pro decode profile (real FP8)
# =====================================================================
# Stage:  Decode (autoregressive)
# Model:  MiMo V2 Pro (384 routed experts, top_k=8, SwiGLU)
#         d=6144, f=2048, FP8 e4m3 weights, quant_block_k=128
# Kernel: multi_expert_pipeline (sequential N-expert double-buffer FFN)
#
# Real FP8: uses float8_e4m3fn weights with per-block scales,
# direct_scaled_dot compute path. No bf16 approximation.
#   - bf=512 (real intermediate tile size for fp8)
#   - Per bf tile weights: 3 × (6144 × 512 × 1B) = 9.0 MB
#   - Per bf tile scales: 3 × (48 × 512 × 4B) ≈ 288 KB
#   - n_w = 2048 / 512 = 4 tiles per expert
#
# EP layout: 2x2x1 = 4 chips = 8 devices.
# EP=8 → 384/8 = 48 experts per device.
# Sweep: [1, 4, 8, 12, 16, 24, 48] experts.
#
# Shape encoding (metadata): num_tokens,hidden_size,intermediate_size
#
# Run:
#   bash scripts/run_benchmark.sh kernels.multi_expert_pipeline \
#     --shape "256,6144,2048" --no-ir-dump \
#     --runner-script scripts/sweep_multi_expert_device.py \
#     --runner-args "--profile mimo-v2-fp8"

kernel: kernels.multi_expert_pipeline
shape: "256,6144,2048"
tpu_type: v7x
tpu_topology: 2x2x1
no_ir_dump: true
runner_script: scripts/sweep_multi_expert_device.py
runner_args: "--profile mimo-v2-fp8"
```

- [ ] **Step 2: Add `mimo-v2-fp8` profile to sweep runner**

In `scripts/sweep_multi_expert_device.py`, add the new profile to `PROFILES`:

```python
PROFILES = {
    "ling2.6": dict(
        hidden_size=8192, intermediate_size=2048, bf=256,
        experts=[1, 4, 8, 16, 32, 64, 256],
    ),
    "mimo-v2": dict(
        hidden_size=6144, intermediate_size=1024, bf=256,
        experts=[1, 4, 8, 12, 16, 24, 48],
    ),
    "mimo-v2-fp8": dict(
        hidden_size=6144, intermediate_size=2048, bf=512,
        weight_dtype="float8_e4m3fn", quant_block_k=128,
        experts=[1, 4, 8, 12, 16, 24, 48],
    ),
}
```

Update the `main()` function to pass fp8 params through to `kernel_fn`:

```python
    # After extracting defaults:
    weight_dtype_str = defaults.get("weight_dtype", "bfloat16")
    weight_dtype = jnp.float8_e4m3fn if weight_dtype_str == "float8_e4m3fn" else jnp.bfloat16
    qbk = defaults.get("quant_block_k", 128)

    # In the expert loop:
    for ne in expert_counts:
        run = kernel_fn(
            num_experts=ne, num_tokens=bt,
            hidden_size=d, intermediate_size=f, bf=bf,
            weight_dtype=weight_dtype, quant_block_k=qbk,
        )
```

- [ ] **Step 3: Commit**

```bash
git add sweep_multi_expert_mimo_v2_fp8.yaml scripts/sweep_multi_expert_device.py
git commit -m "feat: add MiMo V2 Pro fp8 benchmark config and sweep profile"
```

---

## VMEM Budget Verification (MiMo fp8, d=6144, f=2048, bf=512, bt=256, qbk=128)

| Buffer | Shape | Dtype | Size |
|--------|-------|-------|------|
| b_w1_x2_vmem | (2, 6144, 512) | fp8 | 6.0 MB |
| b_w3_x2_vmem | (2, 6144, 512) | fp8 | 6.0 MB |
| b_w2_x2_vmem | (2, 512, 6144) | fp8 | 6.0 MB |
| b_w1_scale_x2 | (2, 48, 1, 512) | f32 | 192 KB |
| b_w3_scale_x2 | (2, 48, 1, 512) | f32 | 192 KB |
| b_w2_scale_x2 | (2, 4, 1, 6144) | f32 | 192 KB |
| b_x_x2_vmem | (2, 256, 6144) | bf16 | 6.0 MB |
| b_y_acc_vmem | (256, 6144) | f32 | 6.0 MB |
| b_y_out_vmem | (2, 256, 6144) | bf16 | 6.0 MB |
| b_gate_acc | (256, 512) | f32 | 512 KB |
| b_up_acc | (256, 512) | f32 | 512 KB |
| Semaphores | (2,3) + (2,) + (2,) | — | ~100 B |
| **Total** | | | **~37.6 MB** |

VMEM capacity: 64 MB → **58.7% utilization** → OK.
