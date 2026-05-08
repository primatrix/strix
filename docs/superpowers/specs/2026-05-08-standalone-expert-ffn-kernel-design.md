# Standalone Expert FFN Kernel — Design Spec

**Date**: 2026-05-08
**Status**: Approved
**Goal**: Extract the expert FFN computation from `_fused_ep_moe_kernel` into a standalone Pallas kernel for comprehensive profiling (MXU utilization, tile size tuning, weight staging analysis).

## Context

The fused MoE kernel (`kernels/_fused_moe_impl.py`, ~3600 lines) fuses routing, All-to-All scatter, expert FFN compute, All-to-All gather, shared expert, and output accumulation into a single Pallas kernel. Profiling the expert FFN in isolation is impossible within this monolith because A2A communication and routing overhead are interleaved with the compute pipeline.

The expert FFN (`expert_ffn` function, lines 1911-2354) is the computational core: it executes FFN1 (gate + up projection) followed by gated activation and FFN2 (down projection) with a fully pipelined DMA staging strategy.

## Scope

### Included
- Complete expert FFN pipeline: weight double-buffering, token staging (HBM→VMEM), FFN1 (gate+up), gated activation, FFN2 (down), result staging (VMEM→HBM)
- Manual async DMA with semaphores (faithful to fused kernel behavior)
- BF16 weights and activations only
- Ling 2.6 default model shape (H=8192, I=2048)
- Static token count (compile-time known, no dynamic `fori_loop` bounds)
- Compatible with existing `benchmark_runner.py` harness

### Excluded
- All-to-All scatter/gather (inter-device communication)
- Expert routing and metadata allreduce
- Shared expert computation
- FP8 / per-group quantization paths
- Bias terms (b1/b2/b3)
- Multi-expert loop
- `shard_map` / multi-device mesh

## Architecture

### File Structure

Single new file: `kernels/expert_ffn.py`

Follows the `kernel_fn`/`config` contract defined in `kernels/__init__.py`. No separate `_impl.py` — the kernel is compact enough (~500-700 lines) for one file.

### Kernel I/O

**Note**: BF16 `t_packing = 32 // 16 = 2`. All token-dimension arrays use the packed layout `(N, t_packing, H // t_packing)`.

**Inputs (HBM refs)**:

| Ref | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `tokens_hbm` | `(N, 2, H//2)` | BF16 | Input tokens (packed: `t_packing=2` for BF16) |
| `w1_hbm` | `(H, I)` | BF16 | Gate projection weights (single expert, no expert dim) |
| `w2_hbm` | `(I, H)` | BF16 | Down projection weights |
| `w3_hbm` | `(H, I)` | BF16 | Up projection weights |

Where `N = num_tokens`, `H = hidden_size = 8192`, `I = intermediate_size = 2048`, `P = t_packing = 2`.

**Output (HBM ref)**:

| Ref | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `output_hbm` | `(N, H)` | BF16 | FFN output (reshaped from packed `(N, 2, H//2)` internally) |

### Tile Parameters

Reuses `FusedMoEBlockConfig` field names (subset):

| Param | Role | Ling 2.6 Default | Derived Loop Count |
|-------|------|-------------------|--------------------|
| `bf` | Intermediate-size block | 2048 | `num_bf = I/bf = 1` |
| `bd1` / `bd1c` | Hidden-size block for FFN1 | 1024 / 1024 | `num_bd1 = H/bd1 = 8` |
| `bd2` / `bd2c` | Hidden-size block for FFN2 | 1024 / 1024 | `num_bd2 = H/bd2 = 8` |
| `bts` | Token staging tile | = N | `num_token_tiles = N/bts` |
| `btc` | Token compute tile | = N | `num_loops = bts/btc` |
| `bfc` | Intermediate compute tile | 2048 | `cdiv(bf, bfc)` |

### Scratch Buffers

**VMEM**:

| Buffer | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| `b_w1_x2` | `(2, P, bd1//P, bf)` | BF16 | W1 weight ping-pong |
| `b_w3_x2` | `(2, P, bd1//P, bf)` | BF16 | W3 weight ping-pong |
| `b_w2_x2` | `(2, P, bf, bd2//P)` | BF16 | W2 weight ping-pong |
| `b_acc` | `(2, N, 1, bf)` | F32 | FFN1 gate (acc1) + up (acc3) accumulators |
| `b_stage_x2` | `(2, bts, P, bd1//P)` | BF16 | Token staging double-buffer |
| `b_acc_stage_x3` | `(3, bts, P, bd2//P)` | BF16 | Result staging triple-buffer |

Where `P = t_packing = 2` (BF16).

**Semaphores**:

| Semaphore | Type | Count | Purpose |
|-----------|------|-------|---------|
| `token_stage_sems` | DMA | (2,) | Token staging double-buffer |
| `acc_stage_sems` | DMA | (3,) | Result staging triple-buffer |
| `weight_sems` | DMA | (2, 3) | Weight loading: dim0 = ping-pong, dim1 = w1/w2/w3 |

## Pipeline Structure

The kernel body faithfully reproduces the `expert_ffn` pipeline from the fused kernel:

```
for bf_id in range(num_bf):
  ┌─ FFN1 Phase (gate + up) ──────────────────────────────────┐
  │ prefetch W1[bd1=0, bf_id], W3[bd1=0, bf_id]               │
  │ for bd1_id in range(num_bd1):                              │
  │   wait W1[bd1_id], W3[bd1_id]                              │
  │   prefetch W1[bd1_id+1], W3[bd1_id+1]  (if not last)      │
  │   for tile_id in range(num_token_tiles):                    │
  │     prefetch tokens[tile+1, bd1_id]     (double-buffer)    │
  │     wait tokens[tile]                                       │
  │     dynamic_ffn1(tokens, W1, W3 → acc1, acc3)              │
  │   if last bd1: prefetch W2[bf_id, bd2=0]                   │
  └────────────────────────────────────────────────────────────┘

  ┌─ FFN2 Phase (activation + down) ──────────────────────────┐
  │ for bd2_id in range(num_bd2):                              │
  │   wait W2[bd2_id]                                          │
  │   prefetch W2[bd2_id+1] or W1/W3[next bf_id]              │
  │   for tile_id in range(num_token_tiles):                    │
  │     load result[tile, bd2] from HBM     (bf>0, triple-buf)│
  │     dynamic_ffn2(acc1, acc3, W2 → result_tile)             │
  │     store result[tile, bd2] to HBM      (triple-buf)      │
  └────────────────────────────────────────────────────────────┘
```

### Compute Functions

`dynamic_ffn1` and `dynamic_ffn2` are copied from `_fused_moe_impl.py` with simplifications:

**`dynamic_ffn1`** (gate + up projection):
- Input: token tile `(bts, P, bd1//P)`, W1/W3 `(P, bd1//P, bf)`
- Output: acc1/acc3 `(bts, bf)` in F32
- Stripped: all FP8/scale/bias branches
- Compute: nested loops over `(bd1c, bfc, t_packing)`, `jnp.dot` with `preferred_element_type=jnp.float32`

**`dynamic_ffn2`** (activation + down projection):
- Input: acc1/acc3 `(bts, bf)`, W2 `(P, bf, bd2//P)`
- Output: result tile `(bts, P, bd2//P)` in BF16
- Applies `activation_fn(acc1, acc3)` then `act @ W2`
- Stripped: all FP8/scale/bias branches

**`activation_fn`**: Unchanged from original — supports `silu`/`gelu`/`swigluoai` via the `act_fn` parameter.

## Wrapper

### `config` dict

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
    "tpu_type": "v7x",
    "tpu_topology": "1x1",
    "description": "Standalone Expert FFN — Ling 2.6 per-expert profile (H=8192, I=2048)",
}
```

### `kernel_fn`

```python
def kernel_fn(
    num_tokens=256,
    hidden_size=8192,
    intermediate_size=2048,
    dtype=jnp.bfloat16,
    act_fn="silu",
    block_config=None,  # FusedMoEBlockConfig or None for defaults
):
    # 1. Generate random inputs (single device, no mesh)
    # 2. Build pallas_call with PrefetchScalarGridSpec
    # 3. CompilerParams: vmem_limit_bytes=96MB (no collective_id)
    # 4. Return run() closure
```

Key differences from `fused_moe.kernel_fn`:
- No mesh / shard_map (single device)
- No A2A scratch allocation
- No topk_weights / topk_ids
- Weights have no expert dimension

### Benchmark compatibility

Invoked as: `python -m scripts.benchmark_runner --kernel kernels.expert_ffn`

No changes needed to `benchmark_runner.py`.

## Testing

The standalone kernel can be validated against `ref_moe` (from `_fused_moe_impl.py`) configured with `num_experts=1, top_k=1` and uniform routing weights. However, structural tests (AST checks, import checks) matching `test_fused_moe_kernel.py` patterns are sufficient for the initial implementation since correctness will be verified on-device via the benchmark harness.

## Non-goals

- This kernel is not intended for production serving — it's a profiling tool
- No auto-tuning infrastructure for the standalone kernel's tile parameters
- No FP8 support (can be added later if needed)
- No multi-expert batching
