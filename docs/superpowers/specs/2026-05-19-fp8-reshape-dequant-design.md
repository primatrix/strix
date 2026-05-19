# FP8 Reshape Pre-Dequant Optimization

**Date:** 2026-05-19
**Scope:** `kernels/multi_expert_pipeline_fp8.py` — dequant functions only
**Parent spec:** `2026-05-19-multi-expert-pipeline-fp8-design.md`

## Problem

The three dequant functions (`dequant_w1`, `dequant_w3`, `dequant_w2`) in `_multi_expert_kernel_fp8` use `lax.fori_loop` with full unroll to iterate over scale groups. Each iteration performs VMEM ref indexing, fp8-to-f32 cast, scale broadcast-multiply, f32-to-bf16 cast, and VMEM write — 24 iterations for w1/w3 (d=6144, qbk=256) and 1 iteration for w2 (bf=256, qbk=256).

The fori_loop approach has:
- Per-iteration loop control overhead (24 iterations)
- 24 separate VMEM ref read/write operations with `pl.ds()` indexing
- Limited compiler visibility — Mosaic sees 24 independent sub-operations instead of one bulk operation

Dequant sits on the critical path between DMA wait and matmul start. Any latency reduction directly improves kernel performance.

## Solution

Replace the `lax.fori_loop` dequant pattern with a reshape-based bulk multiply, following the pattern from `tpu-inference/megablox/gmm_v2.py:382-390`.

### Before (fori_loop)

```python
def dequant_w1(slot):
    def _dq(sg_id, _):
        sg_off = sg_id * quant_block_k
        w_fp8 = b_w1_x2_vmem[slot, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
        s = b_w1_scale_x2_vmem[slot, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
        s = s.reshape(1, bf)
        w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, bf))).astype(jnp.bfloat16)
        b_w1_dq_vmem.at[pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
        return None
    lax.fori_loop(0, n_sg, _dq, None, unroll=n_sg)
```

### After (reshape)

```python
def dequant_w1(slot):
    w_fp8 = b_w1_x2_vmem[slot]
    s = b_w1_scale_x2_vmem[slot]
    w_f32 = w_fp8.astype(jnp.float32).reshape(n_sg, quant_block_k, bf)
    b_w1_dq_vmem[...] = (w_f32 * s).astype(jnp.bfloat16).reshape(d, bf)
```

### All three functions

| Function | Weight shape | Scale shape | Reshape dims |
|----------|-------------|-------------|-------------|
| `dequant_w1` | (d, bf) fp8 | (n_sg, 1, bf) f32 | (n_sg, qbk, bf) |
| `dequant_w3` | (d, bf) fp8 | (n_sg, 1, bf) f32 | (n_sg, qbk, bf) |
| `dequant_w2` | (bf, d) fp8 | (n_sg2, 1, d) f32 | (n_sg2, qbk, d) |

Where `n_sg = d // quant_block_k`, `n_sg2 = bf // quant_block_k`.

## Correctness

Mathematically equivalent. The fori_loop computes:

```
for sg in [0, n_sg):
    w_dq[sg*qbk:(sg+1)*qbk, :] = cast_bf16(cast_f32(w_fp8[sg*qbk:(sg+1)*qbk, :]) * scale[sg, :])
```

The reshape computes:

```
w_dq = cast_bf16(cast_f32(w_fp8).reshape(n_sg, qbk, N) * scale).reshape(K, N)
```

Same operations, same result. Reshape is a zero-cost logical reinterpretation in Pallas.

## What does NOT change

- Pipeline structure (DMA -> dequant -> matmul critical path)
- `compute_tile` function
- `expert_body` loop (Timing-C pipeline)
- DMA helpers (start/wait fetch functions)
- Writeback logic
- `multi_expert_ffn_fp8` JIT entry point
- scratch_shapes / VMEM allocations
- `_ref_multi_expert_ffn_fp8` reference implementation
- `kernel_fn` factory and `__main__` correctness test

## Expected benefits

- Eliminates fori_loop control overhead (24 iterations -> 1 bulk operation)
- Compiler sees full operation graph for better instruction scheduling
- Fewer VMEM ref indexing operations (1 bulk read/write vs 24 sub-group read/write)
- Simpler code (~4 lines per function vs ~8 lines)

## Risks

- Actual speedup depends on Mosaic compiler optimization quality for large-array reshape+multiply
- Dequant remains on the critical path — if compiler generates equivalent code, no improvement
- Should be verified with TPU profiling after implementation

## Verification

Run the existing `__main__` correctness test:
```
python -m kernels.multi_expert_pipeline_fp8 [num_experts] [bt] [d] [f]
```
Expect: `PASS` with `worst_rel_err < 0.1` (same tolerance as current).
