# Multi-Expert Pipeline FP8 Block Quantization — Design Spec

**Date:** 2026-05-19
**Base kernel:** `kernels/multi_expert_pipeline.py` (bf16 multi-expert pipeline)
**Reference implementations:**
- `kernels/_fused_moe_v2_impl.py` (FP8 block quant, direct_scaled_dot + VMEM dequant)
- `~/Code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py` (sub-channel quant)

**Target model:** MiMo V2 Pro — E=384, H=6144, I=2048, FP8 e4m3fn, EP=32

---

## 1. Goal

Add FP8 e4m3fn block quantization support to the multi-expert double-buffer FFN pipeline. This enables benchmarking the actual FP8 data path on TPU (replacing the current "FP8 approximation" approach of halving intermediate_size in bf16).

## 2. Scope

**In scope:**
- New file `kernels/multi_expert_pipeline_fp8.py` with `multi_expert_ffn_fp8()` entry point
- VMEM dequant path: load fp8 from HBM → dequant to bf16 in VMEM → bf16 matmul
- v2-style per-block scale: `(E, contracting_dim // quant_block_k, 1, non_contracting_dim)` f32
- FP8 e4m3fn weights, bf16 tokens (weight-only quantization)
- DMA double-buffered pipeline with scale co-fetching
- Pure-JAX reference implementation for correctness validation
- `config` dict and `kernel_fn()` factory for sweep integration

**Out of scope:**
- `direct_scaled_dot` path (future work)
- Activation quantization (tokens stay bf16)
- Modification to existing `multi_expert_pipeline.py`
- MoE routing / all-to-all scatter-gather
- Shared expert path

## 3. API

```python
@functools.partial(jax.jit, static_argnames=[
    "act_fn", "bf", "num_experts", "quant_block_k",
])
def multi_expert_ffn_fp8(
    tokens: jax.Array,          # (num_experts, bt, d) bf16
    w1: jax.Array,              # (num_experts, d, f) float8_e4m3fn
    w2: jax.Array,              # (num_experts, f, d) float8_e4m3fn
    w3: jax.Array,              # (num_experts, d, f) float8_e4m3fn
    *,
    w1_scale: jax.Array,        # (num_experts, d // quant_block_k, 1, f) f32
    w2_scale: jax.Array,        # (num_experts, f // quant_block_k, 1, d) f32
    w3_scale: jax.Array,        # (num_experts, d // quant_block_k, 1, f) f32
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
    quant_block_k: int = 256,
) -> jax.Array:                 # (num_experts, bt, d) bf16
```

**Constraints:**
- `quant_block_k` must be 128-aligned
- `d % quant_block_k == 0`
- `bf % quant_block_k == 0`
- `f % bf == 0` and `f // bf >= 2` (need ≥ 2 weight tiles for double-buffering)
- All scale tensors are f32

## 4. VMEM Scratch Layout

| Buffer | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| b_w1_x2_vmem | (2, d, bf) | fp8 e4m3 | Weight double-buffer: gate projection |
| b_w3_x2_vmem | (2, d, bf) | fp8 e4m3 | Weight double-buffer: up projection |
| b_w2_x2_vmem | (2, bf, d) | fp8 e4m3 | Weight double-buffer: down projection |
| b_w1_scale_x2 | (2, d//qbk, 1, bf) | f32 | Scale double-buffer for w1 |
| b_w3_scale_x2 | (2, d//qbk, 1, bf) | f32 | Scale double-buffer for w3 |
| b_w2_scale_x2 | (2, bf//qbk, 1, d) | f32 | Scale double-buffer for w2 |
| b_w1_dq_vmem | (d, bf) | bf16 | Dequant scratch for w1 |
| b_w3_dq_vmem | (d, bf) | bf16 | Dequant scratch for w3 |
| b_w2_dq_vmem | (bf, d) | bf16 | Dequant scratch for w2 |
| b_x_x2_vmem | (2, bt, d) | bf16 | Token double-buffer |
| b_y_acc_vmem | (bt, d) | f32 | Output accumulator |
| b_y_out_vmem | (2, bt, d) | bf16 | Output double-buffer for writeback |

### VMEM Budget (MiMo V2: d=6144, f=2048, bt=256, bf=256, qbk=256)

| Category | Calculation | Size |
|----------|------------|------|
| fp8 weight bufs × 2 | 3 × 2 × (6144 × 256) × 1B | 9.0 MiB |
| f32 scale bufs × 2 | 2 × (2 × (24 × 1 × 256) + (1 × 1 × 6144)) × 4B | 0.14 MiB |
| bf16 dequant scratch | 3 × (6144 × 256) × 2B | 9.0 MiB |
| b_x_x2 (bf16) | 2 × (256 × 6144) × 2B | 6.0 MiB |
| b_y_acc (f32) | (256 × 6144) × 4B | 6.0 MiB |
| b_y_out (bf16) | 2 × (256 × 6144) × 2B | 6.0 MiB |
| **Total** | | **~36.1 MiB** |

36.1 MiB < 64 MiB VMEM limit. Sufficient headroom.

### Semaphores

| Semaphore | Shape | Purpose |
|-----------|-------|---------|
| weight_sems | (2, 3) | DMA: [slot, channel] — w1/w1_scale=0, w3/w3_scale=1, w2/w2_scale=2 |
| x_sem | (2,) | DMA: token load per x_slot |
| y_out_sem | (2,) | DMA: output writeback per y_slot |

Scale DMA shares semaphore with corresponding weight channel. Two async_copy calls on the same semaphore: both must complete before wait returns.

## 5. Compute Tile — VMEM Dequant Path

```
compute_tile(x_slot, w_slot, is_first_tile):
    # --- FFN1: gate + up ---
    wait_fetch_w1(w_slot)        # waits for both fp8 weight + f32 scale
    wait_fetch_w3(w_slot)

    # Dequant w1: fp8 → bf16 in VMEM scratch
    for sg_id in range(n_sg):    # n_sg = d // quant_block_k, unrolled
        sg_off = sg_id * quant_block_k
        w1_fp8 = b_w1_x2_vmem[w_slot, sg_off:sg_off+qbk, :]      # (qbk, bf) fp8
        s1     = b_w1_scale_x2[w_slot, sg_id, 0, :]               # (1, bf) f32
        b_w1_dq[sg_off:sg_off+qbk, :] = (w1_fp8.astype(f32) * s1).astype(bf16)

    # Same for w3 → b_w3_dq

    gate = dot(x_bf16, b_w1_dq_bf16)    # (bt, d) × (d, bf) → (bt, bf) f32
    up   = dot(x_bf16, b_w3_dq_bf16)

    # --- FFN2: down-projection ---
    wait_fetch_w2(w_slot)

    # Dequant w2 → b_w2_dq (n_sg2 = bf // quant_block_k sub-groups)
    for sg_id in range(n_sg2):
        sg_off = sg_id * quant_block_k
        w2_fp8 = b_w2_x2_vmem[w_slot, sg_off:sg_off+qbk, :]      # (qbk, d) fp8
        s2     = b_w2_scale_x2[w_slot, sg_id, 0, :]               # (1, d) f32
        b_w2_dq[sg_off:sg_off+qbk, :] = (w2_fp8.astype(f32) * s2).astype(bf16)

    act = silu(gate) * up
    partial = dot(act, b_w2_dq_bf16)    # (bt, bf) × (bf, d) → (bt, d) f32

    if is_first_tile:
        y_acc = partial
    else:
        y_acc += partial
```

The dequant loops use `lax.fori_loop(..., unroll=n_sg)` — fully unrolled at trace time, matching `_fused_moe_v2_impl.py` lines 1042-1082.

## 6. DMA Pipeline

### 6.1 Global Prologue

```
start_load_x(slot=0, expert=0)
start_fetch_w1(slot=0, expert=0, tile=0) + w1_scale
start_fetch_w3(slot=0, expert=0, tile=0) + w3_scale
start_fetch_w2(slot=0, expert=0, tile=0) + w2_scale
start_fetch_w1(slot=1, expert=0, tile=1) + w1_scale
start_fetch_w3(slot=1, expert=0, tile=1) + w3_scale
start_fetch_w2(slot=1, expert=0, tile=1) + w2_scale
wait_load_x(slot=0)
```

### 6.2 Expert Loop (fori_loop, unroll=False)

```
expert_body(e, x_slot):
    next_xs = 1 - x_slot

    # Tile 0
    wait_fetch_w1(0), wait_fetch_w3(0)
    compute_tile(x_slot, w_slot=0, is_first_tile=True)
    # Note: compute_tile internally waits for w2

    # Steady state: tiles 1..n_w-2
    # Note: NOT using compute_tile() — waits and prefetches are interleaved
    for tile in range(1, n_w - 1):
        slot = tile % 2
        next_slot = 1 - slot

        # FFN1: wait → dequant → gate/up dots
        wait_fetch_w1(slot), wait_fetch_w3(slot)
        dequant_w1(slot) → b_w1_dq
        dequant_w3(slot) → b_w3_dq
        gate = dot(x, b_w1_dq)
        up   = dot(x, b_w3_dq)

        # ──── Timing C: prefetch next tile AFTER wait_w2 ────
        wait_fetch_w2(slot)
        start_fetch_w1(next_slot, e, tile+1) + scale
        start_fetch_w3(next_slot, e, tile+1) + scale
        start_fetch_w2(next_slot, e, tile+1) + scale

        # FFN2: dequant w2 → act → down dot (overlaps with next DMA)
        dequant_w2(slot) → b_w2_dq
        act = silu(gate) * up
        partial = dot(act, b_w2_dq)
        y_acc += partial

    # Epilogue: last tile
    if n_w >= 2:
        last_slot = (n_w - 1) % 2

        # FFN1
        wait_fetch_w1(last_slot), wait_fetch_w3(last_slot)
        dequant_w1(last_slot), dequant_w3(last_slot)
        gate = dot(x, b_w1_dq)
        up   = dot(x, b_w3_dq)

        # Timing C: prefetch NEXT EXPERT after wait_w2
        wait_fetch_w2(last_slot)
        @pl.when(e < num_experts - 1):
            start_load_x(next_xs, e + 1)
            start_fetch_w1(0, e+1, 0) + scale
            start_fetch_w3(0, e+1, 0) + scale
            start_fetch_w2(0, e+1, 0) + scale

        # FFN2
        dequant_w2(last_slot)
        act = silu(gate) * up
        partial = dot(act, b_w2_dq)
        y_acc += partial

    # Expert boundary: double-buffered writeback
    @pl.when(e >= 2):
        wait_writeback(x_slot)
    b_y_out[x_slot] = y_acc.astype(bf16)
    start_writeback(e, x_slot)

    @pl.when(e < num_experts - 1):
        if n_w >= 2:
            start_fetch_w1(1, e+1, 1) + scale
            start_fetch_w3(1, e+1, 1) + scale
            start_fetch_w2(1, e+1, 1) + scale
        wait_load_x(next_xs)

    return next_xs
```

### 6.3 Epilogue

```
lax.fori_loop(0, num_experts, expert_body, jnp.int32(0), unroll=False)
last_y_slot = (num_experts - 1) % 2
wait_writeback(last_y_slot)
if num_experts >= 2:
    wait_writeback(1 - last_y_slot)
```

### 6.4 Scale Co-fetching

Each `start_fetch_wN(slot, expert, tile)` issues two `pltpu.make_async_copy` calls on the same semaphore:
1. fp8 weight tile
2. f32 scale tile (same slot index)

`wait_fetch_wN(slot)` waits on that semaphore (both copies must complete).

Example for w1:
```python
def start_fetch_w1(slot, expert_idx, tile_idx, priority=1):
    pltpu.make_async_copy(
        src_ref=w1_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
        dst_ref=b_w1_x2_vmem.at[slot],
        sem=weight_sems.at[slot, 0],
    ).start(priority=priority)
    pltpu.make_async_copy(
        src_ref=w1_scale_hbm.at[expert_idx, :, :, pl.ds(tile_idx * bf, bf)],
        dst_ref=b_w1_scale_x2_vmem.at[slot],
        sem=weight_sems.at[slot, 0],
    ).start(priority=priority)
```

## 7. Reference Implementation

```python
def _ref_multi_expert_ffn_fp8(tokens, w1, w2, w3, *,
                               w1_scale, w2_scale, w3_scale,
                               quant_block_k, act_fn="silu"):
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        w1_f32 = _dequant_weight(w1[e], w1_scale[e], quant_block_k)
        w3_f32 = _dequant_weight(w3[e], w3_scale[e], quant_block_k)
        w2_f32 = _dequant_weight(w2[e], w2_scale[e], quant_block_k)
        gate = x @ w1_f32
        up = x @ w3_f32
        act = activation_fn(gate, up, act_fn)
        out = (act @ w2_f32).astype(tokens.dtype)
        outputs.append(out)
    return jnp.stack(outputs)


def _dequant_weight(w_fp8, scale, quant_block_k):
    """Dequant fp8 weight using per-block scale.
    w_fp8: (K, N) fp8
    scale: (K // quant_block_k, 1, N) f32
    Returns: (K, N) f32
    """
    w_f32 = w_fp8.astype(jnp.float32)
    s = jnp.repeat(scale.squeeze(1), quant_block_k, axis=0)  # (K, N) f32
    return w_f32 * s
```

## 8. Config and Sweep Integration

```python
config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 6144,
        "intermediate_size": 2048,
        "num_experts": 8,
    },
    "dtype": "bfloat16",
    "weight_dtype": "float8_e4m3fn",
    "act_fn": "silu",
    "bf": 256,
    "quant_block_k": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Multi-expert double-buffer FFN — FP8 e4m3 block quantization "
        "(VMEM dequant path, N experts sequential)"
    ),
}
```

`kernel_fn()` generates random fp8 weights (via `jax.random.normal → astype(fp8)`), computes per-block scales, and returns a zero-arg closure.

## 9. File Structure

```
kernels/
  multi_expert_pipeline.py       # unchanged — bf16 only
  multi_expert_pipeline_fp8.py   # NEW — this spec
    ├── config                   # sweep integration
    ├── _dequant_weight()        # helper for reference
    ├── _multi_expert_kernel_fp8()  # Pallas kernel body
    ├── multi_expert_ffn_fp8()   # public JIT entry point
    ├── _ref_multi_expert_ffn_fp8()  # correctness reference
    ├── kernel_fn()              # sweep factory
    └── __main__                 # standalone correctness test
```

## 10. Future Work

- `direct_scaled_dot` path (fp8 dot + per-block scale, no VMEM dequant scratch)
- Integration into v3 EP framework (A2A scatter/gather + metadata)
- Activation quantization (fp8 tokens)
