# Double-Buffer Expert Kernel — Design

**Status**: Design approved, pending implementation
**Date**: 2026-05-11
**Source**: `docs/fused-moe-perf-analysis.md` §5.8 (decode, B=1 pipeline)
**Target hardware**: TPU v7x (2x2x1 topology)

---

## 1. Goal

Implement a Pallas kernel, `kernels/double_buffer_expert.py`, that realizes the
§5.8 pipeline **exactly** (B=1 decode, persistent x, dual weight slots, low-
priority W2 DMA, single fp32 y_acc), sized for Ling 2.6 1T per-expert compute
(`d=8192`, `f=2048`, bf16). The kernel is a standalone benchmark target — it
runs a **single routed expert** (no shared expert, no MoE routing).

It supports exactly the two `(num_tokens, bf)` pairings recommended in §5.8:

| num_tokens (bt) | bf (f-tile)   | Tile size | N_w = 2048 / bf | VMEM peak (§5.8 + fp32 y_acc) |
|-----------------|---------------|-----------|-----------------|--------------------------------|
| 256             | 512           | 8 MB      | 4 rounds        | ~60 MB                         |
| 512             | 256           | 4 MB      | 8 rounds        | ~48 MB                         |

Other combinations are rejected with `ValueError`.

---

## 2. Public Interface

File: `kernels/double_buffer_expert.py`

```python
config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 512,                                # default pairs with num_tokens=256
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Double-buffer expert FFN — §5.8 B=1 decode "
        "(persistent x, dual W slots, low-pri W2)"
    ),
}

def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype: jnp.dtype = jnp.bfloat16,
    weight_dtype: jnp.dtype = jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 512,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure `run()` that
    invokes `double_buffer_expert`. Matches the runner contract used by
    `scripts/benchmark_runner.py` (see `expert_ffn.py` for precedent)."""
    ...

def double_buffer_expert(
    tokens: jax.Array,                         # (bt, d) bf16
    w1: jax.Array,                             # (d, f_full) bf16
    w2: jax.Array,                             # (f_full, d) bf16
    w3: jax.Array,                             # (d, f_full) bf16
    *,
    act_fn: str = "silu",
    bf: int = 512,
) -> jax.Array:                                # (bt, d) bf16
    ...

def _ref_expert_ffn(tokens, w1, w2, w3, *, act_fn="silu") -> jax.Array:
    ...
```

### Constraints (enforced in `double_buffer_expert` entry)

- `d = hidden_size = 8192`, `f_full = intermediate_size = 2048` (hard-coded)
- `(num_tokens, bf) ∈ {(256, 512), (512, 256)}` — any other pair raises
  `ValueError`
- `f_full % bf == 0` (always true for allowed configs)
- Shapes: `w1.shape == (d, f_full)`, `w3.shape == (d, f_full)`,
  `w2.shape == (f_full, d)`
- dtype: tokens + weights bf16

---

## 3. VMEM Layout

No `t_packing` dimension — shapes use native `(bt, d)` / `(d, bf)` / `(bf, d)`;
Pallas/XLA handles TPU MXU sublane packing transparently.

```python
scratch_shapes = (
    pltpu.VMEM((2, hidden_size, bf), weight_dtype),      # b_w1_x2_vmem (2, 8192, bf)
    pltpu.VMEM((2, hidden_size, bf), weight_dtype),      # b_w3_x2_vmem
    pltpu.VMEM((2, bf, hidden_size), weight_dtype),      # b_w2_x2_vmem (2, bf, 8192)
    pltpu.VMEM((num_tokens, hidden_size), dtype),        # b_x_vmem — persistent
    pltpu.VMEM((num_tokens, hidden_size), jnp.float32),  # b_y_acc_vmem — fp32 accumulator
    pltpu.VMEM((num_tokens, hidden_size), dtype),        # b_y_out_vmem — bf16 output staging

    pltpu.SemaphoreType.DMA((2, 3)),                     # weight_sems[slot, channel]
    pltpu.SemaphoreType.DMA((1,)),                       # x_sem
    pltpu.SemaphoreType.DMA((1,)),                       # y_out_sem
)
```

### VMEM budget (fp32 y_acc + bf16 output staging)

| Buffer | bt=256, bf=512 | bt=512, bf=256 |
|--------|----------------|----------------|
| `b_x_vmem` (bf16) | 4 MB | 8 MB |
| `b_w1_x2_vmem` (2 × bf16) | 16 MB | 8 MB |
| `b_w3_x2_vmem` (2 × bf16) | 16 MB | 8 MB |
| `b_w2_x2_vmem` (2 × bf16) | 16 MB | 8 MB |
| `b_y_acc_vmem` (fp32) | 8 MB | 16 MB |
| `b_y_out_vmem` (bf16) | 4 MB | 8 MB |
| **Peak** | **64 MB** | **56 MB** |

`compiler_params` uses `vmem_limit_bytes=96*1024*1024` (matches
`expert_ffn.py`), so both configs fit with headroom. Note: peak for
`(bt=256, bf=512)` equals the 64 MB VMEM ceiling cited in
`docs/fused-moe-perf-analysis.md` §5 but the repo's established
`vmem_limit_bytes` of 96 MB is the enforced bound — see `expert_ffn.py`.

---

## 4. Pipeline — §5.8 Prologue + Steady + Epilogue

### 4.1 DMA helpers

```python
def start_load_x():
    pltpu.make_async_copy(
        src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0]
    ).start()

def wait_load_x(): ...

def start_fetch_w1(slot, tile_idx):
    pltpu.make_async_copy(
        src_ref=w1_hbm.at[:, pl.ds(tile_idx * bf, bf)],   # (d, bf) sliced along f
        dst_ref=b_w1_x2_vmem.at[slot],
        sem=weight_sems.at[slot, 0],
    ).start()

# start/wait: fetch_w3 (channel 1), fetch_w2 (channel 2) — analogous
# W2 source slice is w2_hbm.at[pl.ds(tile_idx * bf, bf), :]  — (bf, d) along reduction f
```

### 4.2 Compute function (SwiGLU intermediates resident in VREG)

```python
def compute_tile(slot, is_first_tile):
    x  = b_x_vmem[...]               # (bt, d) bf16
    w1 = b_w1_x2_vmem[slot]          # (d, bf) bf16
    w3 = b_w3_x2_vmem[slot]          # (d, bf) bf16

    gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)  # (bt, bf) fp32 VREG
    up   = jnp.dot(x, w3, preferred_element_type=jnp.float32)  # (bt, bf) fp32 VREG
    act_up = activation_fn(gate, up, act_fn)                   # silu(gate) * up

    w2 = b_w2_x2_vmem[slot]          # (bf, d) bf16
    partial = jnp.dot(act_up, w2, preferred_element_type=jnp.float32)  # (bt, d) fp32

    if is_first_tile:
        b_y_acc_vmem[...] = partial
    else:
        b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial
```

### 4.3 Main pipeline

```python
N_w = f_full // bf                   # 4 (bf=512) or 8 (bf=256)

# ── Prologue (§5.8 stages L1..C2) ──
start_load_x()
start_fetch_w1(0, 0); start_fetch_w3(0, 0); start_fetch_w2(0, 0)  # W2 low-pri: issued same time
wait_load_x()
wait_fetch_w1(0); wait_fetch_w3(0)                                # x, W1_a, W3_a ready

if N_w >= 2:
    start_fetch_w1(1, 1); start_fetch_w3(1, 1); start_fetch_w2(1, 1)

wait_fetch_w2(0)                                                  # W2_a ready (delayed wait)
compute_tile(slot=0, is_first_tile=True)

# ── Steady state: Python for-loop over tile ∈ [1, N_w - 1) ──
for tile in range(1, N_w - 1):
    slot = tile % 2
    next_slot = 1 - slot
    start_fetch_w1(next_slot, tile + 1)
    start_fetch_w3(next_slot, tile + 1)
    start_fetch_w2(next_slot, tile + 1)
    wait_fetch_w1(slot); wait_fetch_w3(slot)
    wait_fetch_w2(slot)
    compute_tile(slot, is_first_tile=False)

# ── Epilogue: last tile, no more loads ──
if N_w >= 2:
    last_slot = (N_w - 1) % 2
    wait_fetch_w1(last_slot); wait_fetch_w3(last_slot)
    wait_fetch_w2(last_slot)
    compute_tile(last_slot, is_first_tile=False)

# ── Write-back: y_acc (fp32 VMEM) → b_y_out_vmem (bf16 VMEM) → output_hbm ──
b_y_out_vmem[...] = b_y_acc_vmem[...].astype(dtype)      # cast in-VMEM
pltpu.make_async_copy(
    src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0],
).start()
pltpu.make_async_copy(                                    # wait via sem
    src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0],
).wait()
```

### 4.4 Boundary behaviour

- `N_w = 4` (bf=512): prologue → steady (tile 1, 2) → epilogue (tile 3)
- `N_w = 8` (bf=256): prologue → steady (tile 1..6) → epilogue (tile 7)
- `N_w = 1` is unreachable because allowed configs force `N_w ∈ {4, 8}`.

### 4.5 W2 "low priority"

Pallas/TPU has no explicit DMA-priority API. "Low priority W2" is realised as:
1. **Issue W1/W3/W2 starts in that order** — DMA engine scheduler serialises
   requests in issue order when there is HBM contention.
2. **Delay `wait_fetch_w2` until right before the FFN2 MXU stage** — by waiting
   on W1/W3 first and computing gate/up, W2's DMA overlaps with MXU work and
   is consumed last, mirroring the §5.7/§5.8 table.

---

## 5. Correctness & Testing

### 5.1 Reference implementation

`_ref_expert_ffn(tokens, w1, w2, w3, act_fn="silu")` — pure-JAX fp32:

```python
gate = tokens.astype(jnp.float32) @ w1.astype(jnp.float32)
up   = tokens.astype(jnp.float32) @ w3.astype(jnp.float32)
act  = activation_fn(gate, up, act_fn)
return (act @ w2.astype(jnp.float32)).astype(tokens.dtype)
```

### 5.2 `__main__` self-check

Accepts `num_tokens` via `sys.argv[1]` (default 256), picks matching `bf`
(`bf = 512 if bt==256 else 256`), generates random bf16 inputs, runs kernel vs
reference, asserts `rel_err < 0.05`, prints `PASS`.

### 5.3 Structural unit tests

`tests/test_double_buffer_expert.py` — AST/import-only checks (no TPU), modelled
on `tests/test_expert_ffn_kernel.py`:

- `test_kernel_file_exists`
- `test_defines_kernel_fn`
- `test_defines_config`
- `test_defines_double_buffer_expert`
- `test_config_default_shape` (num_tokens=256, bf=512, H=8192, I=2048)
- `test_config_dtype_bfloat16`
- `test_kernel_fn_signature` — accepts `num_tokens`, `bf`, `hidden_size`,
  `intermediate_size`, `dtype`, `weight_dtype`, `act_fn`

---

## 6. Benchmark Runner Integration

### 6.1 Runner change

`scripts/benchmark_runner.py` currently passes `chunk_size`, `ep_size`, `bf`,
`bd` as optional kwargs into `kernel_fn`. Extend the same plumbing for
`num_tokens`:

1. Add `--num-tokens INT` to argparse.
2. Add `num_tokens=None` to `run_benchmark()` signature; when non-`None`, set
   `kwargs["num_tokens"] = num_tokens` before calling `kernel_fn(**kwargs)`.
3. Thread `args.num_tokens` from main into `run_benchmark(...)` call site.

### 6.2 YAML file

New `double_buffer_expert_ling2.6.yaml`:

```yaml
kernel: kernels.double_buffer_expert
shape: "256,8192,2048"
tpu_type: v7x
tpu_topology: 2x2x1
```

(`shape` is metadata; runtime uses `--num-tokens` / `--bf` on CLI.)

### 6.3 Invocation recipe

```bash
# Variant A: bt=256, bf=512 (8MB tile, §5.8 recommended)
python scripts/benchmark_runner.py \
  --kernel kernels.double_buffer_expert \
  --shape "256,8192,2048" --job-name dbe_bt256_bf512 \
  --num-tokens 256 --bf 512 \
  --output-dir /tmp/bench

# Variant B: bt=512, bf=256 (4MB tile, §5.8 recommended)
python scripts/benchmark_runner.py \
  --kernel kernels.double_buffer_expert \
  --shape "512,8192,2048" --job-name dbe_bt512_bf256 \
  --num-tokens 512 --bf 256 \
  --output-dir /tmp/bench
```

### 6.4 Runner test

Extend `tests/test_benchmark_runner.py` with a case verifying `--num-tokens`
plumbing (mirrors existing `--bf` / `--bd` / `--chunk-size` tests).

---

## 7. Out of Scope

- Shared expert (separate batched-GEMM workload; arithmetic intensity
  ~269 FLOPs/Byte vs. 1 for the routed expert — mixing distorts benchmark
  signal).
- Quantization (INT8/INT4).
- Multi-batch (`B ≥ 2`) / x-slot rolling (§5.7 pipeline).
- Weight tiling along `d` (per §5.8 the pipeline tiles only along `f`; `d` is
  carried whole in each tile).
- Expert-parallel sharding (single-chip target for this benchmark).

---

## 8. Acceptance Criteria

1. `kernels/double_buffer_expert.py` implements the §5.8 pipeline exactly as
   §3–§4 above.
2. `__main__` passes `rel_err < 0.05` for both allowed `(bt, bf)` pairs.
3. `tests/test_double_buffer_expert.py` passes (AST-only, no TPU needed).
4. `scripts/benchmark_runner.py` plumbs `--num-tokens`; runner tests still pass.
5. `double_buffer_expert_ling2.6.yaml` present at repo root, matches
   `fusedMoE_ling2.6.yaml` style.
6. Both benchmark variants in §6.3 run to completion on TPU v7x and emit
   JSON results to `--output-dir`.
