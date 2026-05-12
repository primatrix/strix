# Multi-Expert Double-Buffer Pipeline — Design Spec

**Date:** 2026-05-12
**Base kernel:** `kernels/double_buffer_expert.py` (§5.8 B=1 decode)

## 1. Goal

Extend the single-expert double-buffer kernel to process N experts sequentially on one device, with DMA overlap between adjacent experts. This targets EP (expert parallel) decode where each device handles `num_local_experts = total_experts / num_devices` experts (e.g., 256 / 8 = 32).

## 2. Scope

**In scope:**
- Sequential multi-expert FFN with expert-level double-buffered token loading
- Expert-to-expert DMA overlap (prefetch next expert's tokens during current expert's last tile)
- Compile-time fixed `num_experts` (Python for-loop fully unrolled)
- Stacked input tensors: tokens `(num_experts, bt, d)`, weights `(num_experts, d, f)` / `(num_experts, f, d)`
- Correctness validation against pure-JAX reference

**Out of scope:**
- MoE routing / all-to-all scatter-gather (caller provides pre-routed tokens)
- Shared expert path
- Weight quantization
- Multi-batch (B > 1)
- Expert-parallel sharding (shard_map wrapping)

## 3. API

```python
@functools.partial(jax.jit, static_argnames=["act_fn", "bf", "num_experts"])
def multi_expert_ffn(
    tokens: jax.Array,     # (num_experts, bt, d) bf16
    w1: jax.Array,         # (num_experts, d, f) bf16
    w2: jax.Array,         # (num_experts, f, d) bf16
    w3: jax.Array,         # (num_experts, d, f) bf16
    *,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
) -> jax.Array:            # (num_experts, bt, d) bf16
```

**Constraints (unchanged from single-expert):**
- d = 8192, f = 2048 (hard-coded)
- (bt, bf) ∈ {(256, 256), (512, 256)}
- All inputs bf16
- `num_experts` is a compile-time constant (static arg)

**Caller responsibility:**
- Route tokens to experts before calling (all-to-all scatter complete)
- Stack per-expert tensors along axis 0
- Unstack / scatter outputs after return

## 4. VMEM Scratch Layout

Single change from `double_buffer_expert.py`: token buffer upgraded from single to double-buffer.

| Buffer | Shape | Dtype | Size (bt=256) | Change |
|--------|-------|-------|---------------|--------|
| b_w1_x2_vmem | (2, d, bf) | bf16 | 8 MiB | — |
| b_w3_x2_vmem | (2, d, bf) | bf16 | 8 MiB | — |
| b_w2_x2_vmem | (2, bf, d) | bf16 | 8 MiB | — |
| **b_x_x2_vmem** | **(2, bt, d)** | **bf16** | **8 MiB** | **+4 MiB** |
| b_y_acc_vmem | (bt, d) | fp32 | 8 MiB | — |
| b_y_out_vmem | (bt, d) | bf16 | 4 MiB | — |
| **Total** | | | **44 MiB** | **+4 MiB** |

44 MiB < 64 MiB VMEM limit.

**Semaphores:**

| Semaphore | Shape | Change |
|-----------|-------|--------|
| weight_sems | (2, 3) | — |
| x_sem | **(2,)** | was (1,) — one per x slot |
| y_out_sem | (1,) | — |

**Weight buffer reuse:** With n_w = 8, the last tile uses w_slot = 7 % 2 = 1. At expert boundary, w_slot 0 is free for the next expert's tile 0 weights.

## 5. Pipeline Design

### 5.1 Overview

Two-level nested pipeline:
- **Outer loop:** Python for-loop over experts (compile-time unrolled)
- **Inner loop:** Python for-loop over weight tiles (same as single-expert)

Expert-level overlap: during the epilogue of expert `e`, prefetch tokens for expert `e+1` into the alternate x slot.

### 5.2 DMA Helpers

Extended from single-expert to accept `expert_idx` parameter for HBM indexing:

```python
def start_load_x(x_slot, expert_idx, priority=1):
    # tokens_hbm.at[expert_idx] → b_x_x2_vmem.at[x_slot]

def start_fetch_w1(w_slot, expert_idx, tile_idx, priority=1):
    # w1_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)] → b_w1_x2_vmem.at[w_slot]

def start_fetch_w3(w_slot, expert_idx, tile_idx, priority=1):
    # w3_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)] → b_w3_x2_vmem.at[w_slot]

def start_fetch_w2(w_slot, expert_idx, tile_idx, priority=0):
    # w2_hbm.at[expert_idx, pl.ds(tile_idx * bf, bf), :] → b_w2_x2_vmem.at[w_slot]

def start_writeback(expert_idx):
    # b_y_out_vmem → output_hbm.at[expert_idx]
```

`compute_tile(x_slot, w_slot, is_first_tile)` reads from `b_x_x2_vmem[x_slot]` instead of `b_x_vmem[...]`.

### 5.3 Full Pipeline Pseudocode

```
# ── Global Prologue ──
start_load_x(x_slot=0, expert=0)
start_fetch_w1(w_slot=0, expert=0, tile=0)
start_fetch_w3(w_slot=0, expert=0, tile=0)
start_fetch_w2(w_slot=0, expert=0, tile=0)
if n_w ≥ 2:
    start_fetch_w1(w_slot=1, expert=0, tile=1)
    start_fetch_w3(w_slot=1, expert=0, tile=1)
    start_fetch_w2(w_slot=1, expert=0, tile=1)
wait_load_x(x_slot=0)
x_slot ← 0

# ── Expert Loop ──
for e in range(num_experts):

    # ── First tile ──
    wait_fetch_w1(0)
    wait_fetch_w3(0)
    compute_tile(x_slot, w_slot=0, is_first=True)

    # ── Steady state: tiles [1, n_w - 1) ──
    for tile in range(1, n_w - 1):
        w_slot ← tile % 2
        next_w ← 1 - w_slot
        start_fetch_w1(next_w, e, tile + 1)
        start_fetch_w3(next_w, e, tile + 1)
        start_fetch_w2(next_w, e, tile + 1)
        wait_fetch_w1(w_slot)
        wait_fetch_w3(w_slot)
        compute_tile(x_slot, w_slot, is_first=False)

    # ── Epilogue: last tile ──
    if n_w ≥ 2:
        last_w ← (n_w - 1) % 2
        if e < num_experts - 1:
            # Prefetch next expert's tokens (overlaps with last tile compute)
            start_load_x(1 - x_slot, e + 1, priority=0)
        wait_fetch_w1(last_w)
        wait_fetch_w3(last_w)
        compute_tile(x_slot, last_w, is_first=False)

    # ── Expert Boundary: writeback + next expert prefetch ──
    b_y_out_vmem ← b_y_acc_vmem.astype(bf16)       # sync cast
    start_writeback(e)                               # async DMA

    if e < num_experts - 1:
        # Prefetch next expert's first weight tiles (overlaps with writeback)
        start_fetch_w1(0, e + 1, 0)
        start_fetch_w3(0, e + 1, 0)
        start_fetch_w2(0, e + 1, 0)
        if n_w ≥ 2:
            start_fetch_w1(1, e + 1, 1)
            start_fetch_w3(1, e + 1, 1)
            start_fetch_w2(1, e + 1, 1)
        wait_writeback()          # ← MUST complete before next compute_tile writes y_acc
        wait_load_x(1 - x_slot)  # ← likely already done (started during epilogue)
        x_slot ← 1 - x_slot
    else:
        wait_writeback()
```

### 5.4 DMA Priority Strategy

| DMA | Priority | Rationale |
|-----|----------|-----------|
| x load (current expert) | 1 | Required before first tile compute |
| w1, w3 fetch | 1 | Gate/up projection — on critical path |
| w2 fetch | 0 | Deferred wait inside compute_tile — overlaps with MXU |
| x[e+1] prefetch (epilogue) | 0 | Background — must not compete with current expert's DMA |

### 5.5 Expert Boundary Overlap Analysis

At the expert boundary, concurrent DMA activity:

| Direction | Operation | Size |
|-----------|-----------|------|
| VMEM → HBM | Writeback: y_out → output_hbm[e] | 4 MiB |
| HBM → VMEM | x[e+1] load (already in flight) | 4 MiB |
| HBM → VMEM | w1[e+1] tile 0 + w3[e+1] tile 0 + w2[e+1] tile 0 | 12 MiB |
| HBM → VMEM | w1[e+1] tile 1 + w3[e+1] tile 1 + w2[e+1] tile 1 | 12 MiB |

Writeback (VMEM→HBM) and loads (HBM→VMEM) are opposite-direction and can overlap.

**Boundary bubble estimate:** ~8-10 µs for weight loads at ~3,000 GiB/s. This is the minimum inter-expert latency.

### 5.6 Correctness Constraint

**Wait-before-write invariant:** `wait_writeback()` must complete before the next expert's `compute_tile()` executes. This ensures:
1. y_out DMA has finished reading from `b_y_out_vmem` — safe for next cast to overwrite
2. All HBM writes for expert `e` are committed before expert `e+1` starts

## 6. Pallas Setup

```python
scratch_shapes = (
    pltpu.VMEM((2, d, bf), weight_dtype),       # b_w1_x2_vmem
    pltpu.VMEM((2, d, bf), weight_dtype),       # b_w3_x2_vmem
    pltpu.VMEM((2, bf, d), weight_dtype),       # b_w2_x2_vmem
    pltpu.VMEM((2, bt, d), dtype),              # b_x_x2_vmem  ← was (bt, d)
    pltpu.VMEM((bt, d), jnp.float32),           # b_y_acc_vmem
    pltpu.VMEM((bt, d), dtype),                 # b_y_out_vmem
    pltpu.SemaphoreType.DMA((2, 3)),            # weight_sems[slot, channel]
    pltpu.SemaphoreType.DMA((2,)),              # x_sem[x_slot]  ← was (1,)
    pltpu.SemaphoreType.DMA((1,)),              # y_out_sem
)
```

Input/output specs with HBM memory space:
```python
in_specs = [hbm, hbm, hbm, hbm]   # tokens, w1, w2, w3 — all (num_experts, ...)
out_specs = hbm                     # output (num_experts, bt, d)
```

Grid: `()` (single cell, same as single-expert).

## 7. Reference Implementation

```python
def _ref_multi_expert_ffn(tokens, w1, w2, w3, *, act_fn="silu"):
    """Pure-JAX reference: loop over experts, each does (silu(x@W1) * (x@W3)) @ W2."""
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        gate = x @ w1[e].astype(jnp.float32)
        up = x @ w3[e].astype(jnp.float32)
        act = activation_fn(gate, up, act_fn)
        out = (act @ w2[e].astype(jnp.float32)).astype(tokens.dtype)
        outputs.append(out)
    return jnp.stack(outputs)
```

## 8. File Layout

- `kernels/multi_expert_pipeline.py` — kernel implementation
- Follows existing `kernel_fn` + `config` contract for benchmark_runner compatibility

## 9. Acceptance Criteria

1. Correctness: `rel_err < 0.05` per expert vs reference (bf16 tolerance)
2. All existing single-expert tests unaffected
3. `kernel_fn` compatible with `scripts/benchmark_runner.py`
4. Benchmarkable on TPU v7x 2x2x1
