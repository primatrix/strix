# Fused MoE Kernel Performance Analysis

**Model**: Ling 2.6 1T (万亿参数旗舰版)  
**Stage**: Decode (autoregressive, 1 token per sequence)  
**Configuration**: `fusedMoE_ling2.6.yaml`  
**Target Hardware**: TPU v7x (2x2x1 topology)

---

## 1. Model Architecture Overview

### MoE Configuration (Ling 2.6 1T)
```
Routed experts:        256
Active experts/token:  8
Shared expert:         1 (always active)
Total experts/token:   9 (8 routed + 1 shared)
Hidden size (d):       8192
Expert intermediate:   2048
Activation:            SwiGLU (silu)
Precision:             BF16
Batch size:            256 tokens
```

### Layer Distribution
- **Total layers**: 80
- **Dense layers**: First 4 layers (0-3)
- **MoE layers**: 76 layers (4-79)
- **Attention**: MLA + Lightning Linear (replaces GQA)

---

## 2. Expert Computation Logic

### 2.1 Single Expert FFN Structure

Each expert implements a **SwiGLU-gated FFN**:

```
Input:  x ∈ ℝ^(d)           where d = 8192
Gate:   W1 ∈ ℝ^(d × f)      where f = 2048
Up:     W3 ∈ ℝ^(d × f)
Down:   W2 ∈ ℝ^(f × d)

Forward pass:
  gate_proj = x @ W1          # (d) @ (d × f) → (f)
  up_proj   = x @ W3          # (d) @ (d × f) → (f)
  act       = silu(gate_proj) * up_proj  # element-wise
  output    = act @ W2        # (f) @ (f × d) → (d)
```

### 2.2 Shared Expert

The shared expert is **always active** for all tokens:
```
Shared W1: (d × f_se)  where f_se = 2048
Shared W3: (d × f_se)
Shared W2: (f_se × d)

Shared output is added to routed expert output:
  final_output = routed_output + shared_output
```

---

## 3. Workload Analysis (Decode @ bs=256)

### 3.1 Per-Token Computation

For **1 token** through **1 expert**:

| Operation | Shape | FLOPs | Memory Read (BF16) |
|-----------|-------|-------|-------------------|
| x @ W1 | (8192) @ (8192×2048) | 2 × 8192 × 2048 = 33.6M | x: 16KB, W1: 32MB |
| x @ W3 | (8192) @ (8192×2048) | 2 × 8192 × 2048 = 33.6M | x: 16KB, W3: 32MB |
| silu + mul | (2048) | ~4K | gate: 4KB, up: 4KB |
| act @ W2 | (2048) @ (2048×8192) | 2 × 2048 × 8192 = 33.6M | act: 4KB, W2: 32MB |
| **Total** | | **~100.8M FLOPs** | **~96MB** |

### 3.2 Batch-Level Computation (256 tokens)

**Routed Experts** (8 active per token):
- Total expert invocations: 256 tokens × 8 experts = 2048 expert calls
- Total FLOPs: 2048 × 100.8M = **206.4 GFLOPs**
- Total weight reads: 2048 × 96MB = **196.6 GB** (without reuse)

**Shared Expert** (1 active for all tokens):
- Batch GEMM: (256 × 8192) @ (8192 × 2048) → (256 × 2048)
- FLOPs: 2 × 256 × 8192 × 2048 = **8.6 GFLOPs**
- Weight reads: 3 × 32MB = **96 MB**

**Total per MoE layer**:
- FLOPs: 206.4 + 8.6 = **215 GFLOPs**
- Weight traffic (no reuse): **196.7 GB**

### 3.3 Full Model (76 MoE layers)

- Total FLOPs: 76 × 215 = **16.3 TFLOPs**
- Total weight traffic: 76 × 196.7 = **14.9 TB** (theoretical, no reuse)

---

## 4. Kernel Implementation Structure

### 4.1 Key Components

The `_fused_ep_moe_kernel` implements:

1. **Routing & Metadata AllReduce**
   - Compute per-expert token counts
   - AllGather across EP mesh (dp_size × tp_size devices)
   - Compute global expert starts/sizes

2. **All-to-All Scatter** (Token Distribution)
   - Scatter tokens to owning devices based on expert assignment
   - Each device owns `local_num_experts = 256 / ep_size` experts
   - Ring buffer: `a2a_s_x2_hbm` (double-buffered)

3. **Expert FFN Computation** (`expert_ffn`)
   - **FFN1** (Gate + Up projections): `dynamic_ffn1`
   - **Activation**: SwiGLU
   - **FFN2** (Down projection): `dynamic_ffn2`
   - Double-buffered weight loading
   - Token staging in `bts`-sized tiles

4. **All-to-All Gather** (Result Collection)
   - Gather expert outputs back to source devices
   - Accumulate into `a2a_g_hbm`

5. **Shared Expert Computation** (`shared_expert_ffn`)
   - Parallel to routed expert processing
   - Uses separate weight buffers
   - Final addition to output

6. **Output Reduction**
   - Weight by top-k routing scores
   - Write to `output_hbm`

### 4.2 Block Configuration (Tuned for v7x, ep_size=32)

From the config comment:
```python
# Closest tuned block config (v7x, ep_size=32):
#   ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 32, True, False):
#     (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c, bse, bts)
#     (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 256)
```

**Tile Sizes**:
- `bt = 8`: Outer token tile (routing/comm/output)
- `bts = 256`: Token staging tile inside expert_ffn
- `btc = 8`: Compute tile for tokens
- `bf = 2048`: Intermediate dimension tile (full)
- `bfc = 8`: Compute tile for intermediate
- `bd1 = bd2 = 2048`: Hidden dimension tile
- `bd1c = bd2c = 2048`: Compute tile for hidden (full)
- `bse = 2048`: Shared expert intermediate tile (full)

**Key Observations**:
- `bf = 2048` → Full intermediate dimension loaded per tile
- `bd1 = bd2 = 2048` → 1/4 of hidden dimension per tile
- `bts = 256` → Large token staging (up to `bt × ep_size`)
- `btc = 8` → Small compute tile for decode (low token count)

---

## 5. Pipeline Structure Analysis

### 5.1 Expert FFN Pipeline Stages

The kernel implements a **multi-level pipelined execution**:

#### **Outer Loop** (over `bf` blocks):
```python
for bf_id in range(num_bf):  # num_bf = intermediate_size / bf = 2048 / 2048 = 1
    # Stage 1: FFN1 (Gate + Up)
    for bd1_id in range(num_bd1):  # num_bd1 = hidden_size / bd1 = 8192 / 2048 = 4
        # Load W1[bd1_id, bf_id], W3[bd1_id, bf_id]
        # For each token tile (bts-sized):
        #   - DMA: HBM → VMEM (tokens)
        #   - Compute: tokens @ W1, tokens @ W3
        #   - Accumulate into acc1_vmem, acc3_vmem
    
    # Stage 2: Activation
    # acc1_vmem, acc3_vmem → silu(acc1) * acc3
    
    # Stage 3: FFN2 (Down)
    for bd2_id in range(num_bd2):  # num_bd2 = hidden_size / bd2 = 8192 / 2048 = 4
        # Load W2[bf_id, bd2_id]
        # For each token tile:
        #   - Compute: act @ W2
        #   - DMA: VMEM → HBM (output accumulator)
```

#### **Double Buffering**:
- **Weight buffers**: `b_w1_x2_vmem[2]`, `b_w3_x2_vmem[2]`, `b_w2_x2_vmem[2]`
  - Ping-pong between `bw_sem_id = 0/1`
  - Prefetch `bd1_id+1` while computing `bd1_id`
  
- **Token buffers**: `b_stage_x2_vmem[2]`
  - Ping-pong between `token_buf_id = 0/1`
  - Prefetch `tile_id+1` while computing `tile_id`

- **Accumulator buffers**: `a2a_s_acc_stage_x3_vmem[3]`
  - Triple buffering for output staging
  - Load → Compute → Store overlap

### 5.2 Critical Path (Per Expert)

For **1 expert** processing **dyn_sz tokens** (dynamic, ≤ `bt × ep_size`):

```
1. Metadata AllReduce:        ~O(log(ep_size)) rounds
2. A2A Scatter (tokens):       ~O(ep_size) DMA ops
3. Expert FFN:
   ├─ FFN1 (4 bd1 slices):
   │  ├─ W1/W3 load:           4 × (2 × 32MB) = 256 MB
   │  ├─ Token load:           dyn_sz × 16 KB
   │  └─ Compute:              dyn_sz × 67.2M FLOPs
   ├─ Activation:              dyn_sz × 4K FLOPs
   └─ FFN2 (4 bd2 slices):
      ├─ W2 load:              4 × 32MB = 128 MB
      ├─ Compute:              dyn_sz × 33.6M FLOPs
      └─ Output store:         dyn_sz × 16 KB
4. A2A Gather (outputs):       ~O(ep_size) DMA ops
5. Shared Expert FFN:          (parallel to routed)
6. Output reduction:           256 × 8 × 16 KB = 32 KB
```

**Total per expert**:
- Weight traffic: **384 MB** (256 + 128)
- Token traffic: **dyn_sz × 32 KB** (read + write)
- Compute: **dyn_sz × 100.8M FLOPs**

---

## 6. Data Movement Analysis

### 6.1 Memory Hierarchy

```
HBM (High Bandwidth Memory)
  ↕ ~1.6 TB/s per device (v7x)
VMEM (Vector Memory / SRAM)
  ↕ ~10 TB/s (on-chip)
MXU (Matrix Multiply Unit)
  ↕ Peak: 197 TFLOPs BF16 (v7x)
```

### 6.2 Weight Reuse Pattern

**Routed Experts** (256 experts, ep_size=32):
- Each device owns: 256 / 32 = **8 local experts**
- Per `bt=8` tile: ~8 tokens × 8 experts / 32 devices = **2 tokens/expert** (average)
- Weight reuse factor: **~2× per bt tile** (low in decode)

**Shared Expert**:
- Processes all 256 tokens in batch
- Weight reuse: **256× per layer** (high)

### 6.3 Arithmetic Intensity

**Routed Expert** (per token):
```
FLOPs:        100.8M
Memory:       96 MB (weights) + 32 KB (tokens)
AI:           100.8M / 96M ≈ 1.05 FLOPs/Byte
```

**Shared Expert** (batch):
```
FLOPs:        8.6 GFLOPs
Memory:       96 MB (weights) + 8 MB (tokens)
AI:           8.6G / 104M ≈ 82.7 FLOPs/Byte
```

**TPU v7x Roofline**:
- Peak compute: 197 TFLOPs BF16
- Peak bandwidth: 1.6 TB/s
- Balanced AI: 197 / 1.6 ≈ **123 FLOPs/Byte**

**Bottleneck**:
- Routed experts (AI ≈ 1): **Memory-bound** (far below roofline)
- Shared expert (AI ≈ 83): **Compute-bound** (near roofline)

---

## 7. Pipeline Scheduling (Manual Analysis)

### 7.1 Ideal Pipeline (FFN1, single bd1 slice)

Assuming perfect overlap:

```
Time →
Cycle:  0    1    2    3    4    5    6    7    8
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
W1/W3:  │ L0 │ C0 │ C0 │ C0 │ L1 │ C1 │ C1 │ C1 │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
Tokens: │    │ L0 │ C0 │ C0 │    │ L1 │ C1 │ C1 │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
Compute:│    │    │ C0 │ C0 │    │    │ C1 │ C1 │
        └────┴────┴────┴────┴────┴────┴────┴────┘

L = Load (DMA), C = Compute (MXU)
```

**Observations**:
- Weight load (256 MB) takes **~160 μs** @ 1.6 TB/s
- Compute (dyn_sz × 67.2M FLOPs) takes **~0.34 μs** @ 197 TFLOPs (for dyn_sz=1)
- **Load dominates** → Pipeline stalls waiting for weights

### 7.2 Actual Pipeline (with prefetch)

The kernel implements:
```python
# Prefetch bd1_id+1 while computing bd1_id
@pl.when(has_tokens & (next_bd1_id < num_bd1))
def _():
    start_fetch_bw1(local_e_id, next_bw_sem_id, bf_id, next_bd1_id)
    start_fetch_bw3(local_e_id, next_bw_sem_id, bf_id, next_bd1_id)
```

**Timeline** (4 bd1 slices):
```
bd1=0: Load W1[0], W3[0] → Compute → Prefetch W1[1], W3[1]
bd1=1: Wait W1[1], W3[1] → Compute → Prefetch W1[2], W3[2]
bd1=2: Wait W1[2], W3[2] → Compute → Prefetch W1[3], W3[3]
bd1=3: Wait W1[3], W3[3] → Compute → Prefetch W2[0] (for FFN2)
```

**Latency hiding**:
- If `Compute(bd1_id) ≥ Load(bd1_id+1)`: **Perfect overlap**
- Else: **Stall** (memory-bound)

For decode (dyn_sz ≈ 2-8 tokens/expert):
- Compute time: **~0.7-2.7 μs** (too short)
- Load time: **~160 μs** (dominant)
- **Stall ratio**: ~98% (poor utilization)

### 7.3 Token Staging Pipeline

The kernel uses **triple buffering** for token tiles:

```python
# Token tile loop (bts=256 sized tiles)
for tile_id in range(num_token_tiles):
    # Prefetch tile_id+1 while computing tile_id
    start_stage_a2a_s_tile_from_hbm(next_start, bd1_id, next_buf_id)
    wait_stage_a2a_s_tile(token_buf_id)
    dynamic_ffn1(b_stage_x2_vmem[token_buf_id], ...)
```

**Benefit**:
- Token load (bts × 16 KB = 4 MB) takes **~2.5 μs**
- Compute (bts × 67.2M FLOPs) takes **~85 μs** @ 197 TFLOPs
- **Compute dominates** → Good overlap for token staging

---

## 8. Bottleneck Analysis

### 8.1 Decode-Specific Challenges

1. **Low Token Count per Expert**:
   - Average: 256 tokens × 8 experts / 256 experts = **8 tokens/expert**
   - With ep_size=32: 8 / 32 ≈ **0.25 tokens/expert/device** (highly imbalanced)
   - Actual distribution: Poisson-like, many experts get 0-2 tokens

2. **Poor Weight Reuse**:
   - Weights (384 MB) loaded for ~2 tokens
   - Reuse factor: **~2× per layer** (vs. 256× for shared expert)

3. **Memory-Bound Execution**:
   - AI ≈ 1 FLOPs/Byte << 123 (roofline)
   - MXU utilization: **~0.8%** (1 / 123)

### 8.2 Communication Overhead

**All-to-All Scatter/Gather**:
- Per bt=8 tile: 8 tokens × 16 KB = **128 KB** per device
- With ep_size=32: 32 devices × 128 KB = **4 MB total**
- ICI bandwidth (v7x): ~300 GB/s per link
- Latency: **~13 μs** (optimistic, ignoring contention)

**Metadata AllReduce**:
- Per bt tile: Allgather `num_experts=256` counts across 32 devices
- Data: 256 × 4 bytes × 32 = **32 KB**
- Rounds: log₂(32) = **5 rounds** (recursive doubling)
- Latency: **~5-10 μs** (network + sync)

**Total comm per bt tile**: **~20-30 μs**

### 8.3 Synchronization Overhead

The kernel uses **full mesh barriers**:
```python
def sync_barrier():
    for i in range(num_devices):
        pltpu.semaphore_signal(barrier_sem, device_id=i)
    pltpu.semaphore_wait(barrier_sem, num_devices)
```

**Frequency**:
- Before/after metadata allreduce: **2× per bt tile**
- Latency: **~2-5 μs per barrier** (depends on stragglers)

**Total sync overhead**: **~4-10 μs per bt tile**

---

## 9. Performance Estimates

### 9.1 Per-Layer Latency (256 tokens, 1 MoE layer)

**Routed Experts** (256 experts, ep_size=32):
- Number of bt tiles: 256 / 8 = **32 tiles**
- Per tile:
  - Metadata allreduce: **~10 μs**
  - A2A scatter: **~13 μs**
  - Expert FFN (8 local experts):
    - Weight load: **~160 μs** (dominant)
    - Compute: **~2 μs** (hidden)
  - A2A gather: **~13 μs**
  - Sync barriers: **~10 μs**
  - **Total per tile**: **~206 μs**
- **Total routed**: 32 × 206 = **~6.6 ms**

**Shared Expert** (parallel):
- Weight load: **~60 μs** (96 MB @ 1.6 TB/s)
- Compute: **~44 μs** (8.6 GFLOPs @ 197 TFLOPs)
- **Total shared**: **~104 μs** (hidden by routed)

**Per-layer latency**: **~6.6 ms**

### 9.2 Full Model (76 MoE layers)

- Total latency: 76 × 6.6 ms = **~502 ms**
- Plus dense layers (4): **~10 ms** (estimated)
- Plus attention (80 layers): **~80 ms** (estimated, MLA)
- **Total model latency**: **~592 ms** (0.59 seconds)

**Throughput**:
- Tokens/second: 256 / 0.592 = **432 tokens/s**
- Per-device: 432 / 32 = **13.5 tokens/s/device**

### 9.3 Hardware Utilization

**MXU Utilization**:
- Achieved FLOPs: 215 GFLOPs / 6.6 ms = **32.6 TFLOPs**
- Peak: 197 TFLOPs × 32 devices = **6.3 PFLOPs**
- Utilization: 32.6T / 6.3P = **0.52%** (extremely low)

**HBM Bandwidth Utilization**:
- Weight traffic: 196.7 GB / 6.6 ms = **29.8 TB/s**
- Peak: 1.6 TB/s × 32 devices = **51.2 TB/s**
- Utilization: 29.8 / 51.2 = **58%** (memory-bound)

---

## 10. Optimization Opportunities

### 10.1 Weight Prefetching

**Current**: Prefetch `bd1_id+1` during `bd1_id` compute  
**Issue**: Compute too short to hide load latency

**Proposal**: **Multi-expert prefetching**
- Prefetch expert `e+1` while computing expert `e`
- Requires: Reordering expert loop to process in sequence
- Benefit: **~160 μs latency hiding per expert**

### 10.2 Expert Batching

**Current**: Process experts independently  
**Issue**: Low token count per expert (poor reuse)

**Proposal**: **Batch multiple experts**
- Group experts with similar token counts
- Load weights for batch, then process all tokens
- Benefit: **Amortize weight load over more tokens**

### 10.3 Token Packing

**Current**: `bt=8` outer tile, `bts=256` staging tile  
**Issue**: Small bt leads to many A2A rounds

**Proposal**: **Increase bt to 32-64**
- Reduce number of A2A rounds: 32 → 8-4
- Reduce metadata allreduce overhead
- Benefit: **~4-8× reduction in comm overhead**

### 10.4 Quantization

**Current**: BF16 weights (2 bytes/param)  
**Proposal**: **INT8/INT4 quantization**
- Weight size: 384 MB → 192 MB (INT8) or 96 MB (INT4)
- Load time: 160 μs → 80 μs (INT8) or 40 μs (INT4)
- Benefit: **~2-4× speedup** (if dequant cost is low)

### 10.5 Expert Parallelism Tuning

**Current**: ep_size=32 (all devices)  
**Issue**: High communication overhead, low tokens/expert

**Proposal**: **Reduce ep_size to 8-16**
- More tokens per expert: 8 → 16-32
- Better weight reuse
- Trade-off: Higher weight memory per device
- Benefit: **~2-4× better MXU utilization**

---

## 11. Comparison with Prefill

| Metric | Decode (bs=256) | Prefill (bs=1, seq=8192) |
|--------|-----------------|--------------------------|
| Tokens/expert | ~2-8 | ~32-128 |
| Weight reuse | ~2× | ~32-128× |
| Arithmetic intensity | ~1 FLOPs/Byte | ~32-128 FLOPs/Byte |
| MXU utilization | ~0.5% | ~25-50% |
| Bottleneck | Memory (weights) | Compute (balanced) |
| Latency/layer | ~6.6 ms | ~50-100 ms |

**Key Insight**: Decode is **fundamentally memory-bound** due to low token count per expert. Prefill benefits from high token count and better weight reuse.

---

## 12. Conclusion

### Summary

The fused MoE kernel for Ling 2.6 1T in decode mode (bs=256) exhibits:

1. **Severe memory bottleneck**: AI ≈ 1 FLOPs/Byte << 123 (roofline)
2. **Poor MXU utilization**: ~0.5% (compute idle most of the time)
3. **High communication overhead**: ~20-30 μs per bt tile (30-40% of total)
4. **Low weight reuse**: ~2× per layer (vs. 256× for shared expert)

### Recommendations

**Short-term** (kernel-level):
1. Increase `bt` to 32-64 (reduce A2A rounds)
2. Implement multi-expert prefetching (hide weight load)
3. Tune `bts` and `btc` for better pipeline balance

**Medium-term** (system-level):
1. Reduce `ep_size` to 8-16 (improve weight reuse)
2. Implement INT8 quantization (reduce weight traffic)
3. Explore expert batching (amortize loads)

**Long-term** (architecture):
1. Increase batch size (more tokens/expert)
2. Continuous batching (mix decode with prefill)
3. Speculative decoding (increase effective batch size)

### Expected Impact

With optimizations:
- **2-4× speedup** from quantization + prefetching
- **2-3× speedup** from ep_size tuning + batching
- **Combined**: **4-12× speedup** (target: ~50-150 ms/layer)
- **MXU utilization**: 0.5% → **5-10%** (still memory-bound, but improved)

---

**Document Version**: 1.0  
**Date**: 2026-05-10  
**Author**: Performance Analysis (AI Assistant)
