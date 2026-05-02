# MoE Kernel: Per-Stage Input Shapes & Dtypes

> **Kernel**: `kernels/fused_moe.py` → `_fused_moe_impl.py:fused_ep_moe`
> **Config**: `fusedMoE_ling2.6.yaml` — Ling 2.6 1T Decode
> **Date**: 2026-05-02
> **Status**: Draft

---

## 1. Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_tokens` | 256 | Batch size (decode: 1 token/seq) |
| `num_experts` | 256 | Total routed experts |
| `top_k` | 8 | Active experts per token |
| `hidden_size` | 8192 | Model hidden dimension |
| `intermediate_size` | 2048 | Expert intermediate (SwiGLU) |
| `se_intermediate_size` | 2048 | Shared expert intermediate |
| `dtype` | `bfloat16` | Activation/compute dtype |
| `weight_dtype` | `bfloat16` | Weight storage dtype |
| `act_fn` | `silu` | SwiGLU activation |
| `ep_size` | 4 | Expert-parallel degree |
| `tpu_type` | v7x | |
| `tpu_topology` | 2x2x1 | |

### Block Tiling Config (tuned for v7x)

| Tile | Value | Meaning |
|------|-------|---------|
| `bt` | 8 | Outer token tile (per-device output) |
| `bts` | 32 | Per-expert token staging tile in VMEM |
| `btc` | 8 | Token compute tile inside GEMM |
| `bf` | 2048 | `intermediate_size` block |
| `bfc` | 8 | `intermediate_size` compute sub-tile |
| `bd1` | 2048 | `hidden_size` block in w1/w3 |
| `bd1c` | 2048 | `hidden_size` compute sub-tile in w1/w3 |
| `bd2` | 2048 | `hidden_size` block in w2 |
| `bd2c` | 2048 | `hidden_size` compute sub-tile in w2 |
| `bse` | 2048 | Shared expert `intermediate_size` block |

### Derived Sizes

```
t_packing         = 2           (32-bit / 16-bit for bf16 packing)
local_num_tokens  = 64          (num_tokens / ep_size)
local_num_experts = 64          (num_experts / ep_size)
h_per_t_packing   = 4096        (hidden_size / t_packing)
bd1_per_t_packing = 1024        (bd1 / t_packing)
bd2_per_t_packing = 1024        (bd2 / t_packing)
bd1c_per_t_packing = 1024       (bd1c / t_packing)
bd2c_per_t_packing = 1024       (bd2c / t_packing)
num_bt            = 8           (local_num_tokens / bt)
a2a_max_tokens    = 32          (align_to(bt * ep_size, bts))
padded_num_experts = 256        (align_to(num_experts, 128))
padded_top_k      = 128         (align_to(top_k, 128))
num_bd1           = 4           (hidden_size / bd1)
num_bd2           = 4           (hidden_size / bd2)
num_bf            = 1           (intermediate_size / bf)
se_total_blocks   = 1           (se_intermediate_size / bse)
expert_buffer_count = 64        (min(local_num_experts, HBM_budget_slots))
```

---

## 2. HBM Inputs (before Pallas kernel)

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `tokens_hbm` | `(64, 2, 4096)` | bf16 | Local token shard, packed minor dim |
| `w1_hbm` | `(64, 8192, 2048)` | bf16 | Gate proj, expert-sharded |
| `w2_hbm` | `(64, 2048, 8192)` | bf16 | Down proj, expert-sharded |
| `w3_hbm` | `(64, 8192, 2048)` | bf16 | Up proj, expert-sharded |
| `topk_weights_hbm` | `(64, 128)` | f32 | Padded to 128 |
| `topk_ids_hbm` | `(64, 128)` | i32 | Padded to 128, -1 sentinel |
| `a2a_s_x2_hbm` | `(64, 32, 2, 4096)` | bf16 | A2A scatter buffers (per-expert-slots) |
| `a2a_s_acc_x2_hbm` | `(64, 32, 2, 4096)` | bf16 | A2A gather-in-progress accum |
| `a2a_g_hbm` | `(256, 8, 2, 4096)` | bf16 | A2A gather output (per global expert) |
| `w1_shared_hbm` | `(8192, 2048)` | bf16 | Shared expert Gate |
| `w2_shared_hbm` | `(2048, 8192)` | bf16 | Shared expert Down |
| `w3_shared_hbm` | `(8192, 2048)` | bf16 | Shared expert Up |
| `output_hbm` | `(64, 8192)` | bf16 | Per-device output |

---

## 3. Pipeline Stages

The kernel runs in an outer loop over `bt` tiles (`num_bt = 8` iterations). Each iteration processes one tile of `bt = 8` tokens. The diagram below follows one `bt` iteration.

```
  BT Tile (bt=8 tokens)
       │
       ▼
┌─────────────────┐
│ 1. Fetch TopK   │  topk_weights, topk_ids: HBM → VMEM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. T2E Routing  │  token→expert mask → expert_sizes, expert_starts
│    + AllReduce  │  Metadata all-gather across ep_size=4 devices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. A2A Scatter  │  Route tokens to expert devices via remote DMA
│    (Batch)      │  One slot per local expert (64 scatter buffers)
└────────┬────────┘
         │
         ▼  ┌──────────────────────────────────────┐
┌──────────────────┐                                │
│ 4. Expert Loop   │ for each local_e_id in 0..63:  │
│                  │                                │
│  ┌─────────────┐ │                                │
│  │4a. SE Slice │ │  Shared Expert FFN interleaved │
│  ├─────────────┤ │                                │
│  │4b. Wait Recv│ │  Wait scatter DMA for expert   │
│  ├─────────────┤ │                                │
│  │4c. FFN1     │ │  Gate+Up GEMM + SiLU activation│
│  │4d. FFN2     │ │  Down GEMM                     │
│  ├─────────────┤ │                                │
│  │4e. A2A Gath │ │  Start gather send back        │
│  ├─────────────┤ │                                │
│  │4f. SE Slice │ │  Next shared expert block      │
│  └─────────────┘ │                                │
└────────┬─────────┘                                │
         │                                          │
         ▼                                          │
┌─────────────────┐                                 │
│ 5. Wait All     │  Wait all scatter sends         │
│    Scatter Sends│  Wait all gather receives       │
└────────┬────────┘                                 │
         │                                          │
         ▼                                          │
┌─────────────────┐                                 │
│ 6. Acc & Store  │  Accumulate gathered tokens     │
│    Output       │  Weight by topk_weights         │
│                 │  Add shared expert result       │
│                 │  Store → output_hbm             │
└─────────────────┘                                 │
```

---

## 4. Stage Details

### Stage 1: Fetch TopK

Load top-k weights and IDs from HBM into VMEM scratch buffers.

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `topk_weights_hbm` (src) | `(64, 128)` | f32 | HBM |
| `b_topk_weights_x2_vmem` (dst) | `(2, 8, 128)` | f32 | VMEM |
| `topk_ids_hbm` (src) | `(64, 128)` | i32 | HBM |
| `b_topk_ids_x2_vmem` (dst) | `(2, 8, 128)` | i32 | VMEM |

**Operation**: DMA copy, `bt=8` rows at a time, double-buffered (`x2` ping-pong).

---

### Stage 2: Token-to-Expert Routing + AllReduce Metadata

For each token in the `bt` tile, map its `top_k=8` expert IDs to a per-expert token count.

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `b_topk_ids_x2_vmem` (src) | `(2, 8, 8)` | i32 | VMEM |
| `t2e_routing_x2_smem` (dst) | `(2, 8, 128)` | i32 | SMEM |
| `d2e_count_x2_smem` | `(2, 4, 1, 256)` | i32 | SMEM |
| `expert_offsets_x2_smem` | `(2, 2, 256)` | i32 | SMEM |
| `expert_starts_x2_smem` | `(2, 1, 256)` | i32 | SMEM |
| `expert_sizes_x2_smem` | `(2, 1, 256)` | i32 | SMEM |

**Operation**:
1. Create routing mask: `(8, 8, 1) == (1, 1, 256)` → per-expert token counts per device
2. All-gather `d2e_count` across 4 devices (recursive doubling in `log2(4)=2` rounds)
3. Compute global `expert_sizes` and local `expert_starts` via prefix-sum

**Key dynamic shape**: `expert_sizes[e_id]` — number of tokens routed to global expert `e_id` in this `bt` tile.

---

### Stage 3: A2A Scatter

Route each token to the device that holds its assigned expert. Uses batch scatter path (`expert_buffer_count = 64 >= local_num_experts = 64`).

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `tokens_hbm` (src) | `(64, 2, 4096)` | bf16 | HBM |
| `a2a_s_x2_hbm` (dst) | `(64, 32, 2, 4096)` | bf16 | HBM |
| `a2a_s_sends_x2_smem` | `(64,)` | i32 | SMEM |

**Operation**:
- For each of the `bt=8` tokens, for each `k` in `top_k=8`, determine if the expert `e_id` is local or remote
- **Local**: HBM→HBM async copy to `a2a_s_x2_hbm[e_sem_id, offset]`
- **Remote**: HBM→HBM async remote copy to peer device's `a2a_s_x2_hbm[e_sem_id, offset]`
- Track `send_sz` per expert slot for later wait

**Per-expert receive size**: `expert_sizes[global_e_id]` — up to `bt * ep_size = 32` tokens.

---

### Stage 4a: Shared Expert FFN (interleaved)

The shared expert applies a dense SwiGLU FFN to all tokens. Executed in slices interleaved with routed expert compute to hide latency.

**FFN1 (Gate+Up)**:

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `b_se_tokens_vmem` | `(2, 2, 8, 2, 1024)` | bf16 | VMEM |
| `b_se_w1_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | VMEM |
| `b_se_w3_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | VMEM |
| `b_se_acc_vmem` (accumulator) | `(2, 8, 8192)` | f32 | VMEM |

**Operation**: For each `bd1` slice (4 slices of 2048):
- `(bt=8, bd1=2048) @ (bd1=2048, bse=2048)` → gate_acc/up_acc `(8, 2048)`
- Accumulate across `bd1` slices: `gate_res, up_res` → `silu(gate_res) * up_res` → `act(8, 2048)`

**FFN2 (Down)**:

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `act` (in registers) | `(8, 2048)` | f32 | — |
| `b_se_w2_x2_vmem` | `(2, 2, 2048, 1024)` | bf16 | VMEM |
| `b_se_acc_vmem` (write) | `(2, 8, 8192)` | f32 | VMEM |

**Operation**: For each `bd2` slice (4 slices of 2048):
- `(8, 2048) @ (2048, bd2=2048)` → `(8, 2048)` chunk
- Accumulate into `b_se_acc_vmem[out_buf_id, :, offset:offset+2048]`

---

### Stage 4c: Routed Expert FFN1 (Gate+Up Projection)

For one local expert that has `dyn_sz` tokens:

**Token staging (HBM → VMEM)**:

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `a2a_s_x2_hbm` (src) | `(64, 32, 2, 4096)` | bf16 | HBM |
| `b_stage_x2_vmem` (dst) | `(2, 32, 2, 1024)` | bf16 | VMEM |

Staged in `bts=32` tiles along the token dimension, `bd1=2048` slices along hidden.

**GEMM compute**:

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `b_stage_x2_vmem` (t) | `(32, 2, 1024)` | bf16 | VMEM |
| `b_w1_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | VMEM |
| `b_w3_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | VMEM |
| `b_acc_vmem` (acc1/acc3) | `(2, 32, 2048)` | f32 | VMEM |

**Operation** (tippled over `btc=8`, `bd1c=2048`, `bfc=8`):
```
for t_tile in 0..ceil(dyn_sz/32):          # token tiles (bts=32)
  for bd1_slice in 0..4:                   # hidden slices
    for token_tile in 0..ceil(dyn_sz/8):   # btc=8 compute tiles
      for bfc_tile in 0..256:              # intermediate slices (bfc=8)
        t(8, 2048) @ w1(2048, 8) → acc1(8, 8)  # in f32
        t(8, 2048) @ w3(2048, 8) → acc3(8, 8)
```

**Key shapes**:
- `num_token_tiles = ceil(dyn_sz / bts) = ceil(dyn_sz / 32)` (typically 1 for decode with dyn_sz ≤ 32)
- Gate result: `acc1(dyn_sz, 2048)` in f32
- Up result: `acc3(dyn_sz, 2048)` in f32

---

### Stage 4d: Routed Expert FFN2 (Down Projection + Activation)

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `b_acc_vmem` (acc1) | `(2, 32, 2048)` | f32 | VMEM |
| `b_acc_vmem` (acc3) | `(2, 32, 2048)` | f32 | VMEM |
| `b_w2_x2_vmem` | `(2, 2, 2048, 1024)` | bf16 | VMEM |
| `a2a_s_acc_stage_x3_vmem` (res) | `(3, 32, 2, 1024)` | bf16 | VMEM |

**Operation**:
```
act = silu(acc1) * acc3                      # (dyn_sz, 2048) SwiGLU
for token_tile in 0..ceil(dyn_sz/8):         # btc=8
  for bd2_slice in 0..4:                     # bd2=2048 slices
    for bfc_tile in 0..256:                  # bfc=8 intermediate tiles
      act(8, 8) @ w2(8, 1024) → res(8, 1024)  # f32 → bf16 write
```

Accumulated into `a2a_s_acc_stage_x3_vmem` (triple-buffered), then written back to `a2a_s_acc_x2_hbm`.

---

### Stage 4e: A2A Gather

Send computed expert outputs back to the token's home device.

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `a2a_s_acc_x2_hbm` (src) | `(64, 32, 2, 4096)` | bf16 | HBM |
| `a2a_g_hbm` (dst) | `(256, 8, 2, 4096)` | bf16 | HBM |

**Operation**: For each token routed to this expert, remote-copy the expert's output chunk back to `a2a_g_hbm[e_id, offset]` on the token's home device. Uses `expert_offsets[1, e_id]` to track gather position.

---

### Stage 5: Wait All Communication

- `wait_a2a_scatter_send_batch()`: wait for all 64 DM send semaphores
- `wait_a2a_gather_recv_all()`: wait for all gather receives

---

### Stage 6: Accumulate & Store Output

**Load gathered tokens**:

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `a2a_g_hbm` (src) | `(256, 8, 2, 4096)` | bf16 | HBM |
| `a2a_g_acc_vmem` (dst) | `(2, 8, 4, 2, 4096)` | bf16 | VMEM |

`acc_bt = gcd(bt, 16) = 4` — loads 4 tokens at a time, double-buffered.

**Weighted accumulation**:

```
for k in 0..8:                                # top_k experts per token
  output(4, 8192) += a2a_g[expert, token](4, 8192) * topk_weight(token, k)
```

Plus shared expert result:
```
output(4, 8192) += b_se_acc_vmem[out_buf, offset:offset+4, :]
```

| Buffer | Shape | Dtype | Location |
|--------|-------|-------|----------|
| `b_topk_weights_x2_vmem` | `(2, 8, 128)` | f32 | VMEM |
| `a2a_g_acc_vmem` | `(2, 8, 4, 2, 4096)` | bf16 | VMEM |
| `b_se_acc_vmem` | `(2, 8, 8192)` | f32 | VMEM |
| `b_output_x2_vmem` (dst) | `(2, 8, 8192)` | bf16 | VMEM |

**Store to HBM**:

| Buffer | Shape | Dtype |
|--------|-------|-------|
| `b_output_x2_vmem` (src) | `(2, 8, 8192)` | bf16 |
| `output_hbm` (dst) | `(64, 8192)` | bf16 |

---

## 5. Memory Summary

### VMEM Allocation (per Pallas kernel instance)

| Scratch Buffer | Shape | Dtype | Bytes |
|---------------|-------|-------|-------|
| `b_topk_weights_x2_vmem` | `(2, 8, 128)` | f32 | 8,192 |
| `b_topk_ids_x2_vmem` | `(2, 8, 128)` | i32 | 8,192 |
| `b_output_x2_vmem` | `(2, 8, 8192)` | bf16 | 262,144 |
| `b_w1_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | 16,777,216 |
| `b_w3_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | 16,777,216 |
| `b_w2_x2_vmem` | `(2, 2, 2048, 1024)` | bf16 | 16,777,216 |
| `b_acc_vmem` | `(2, 32, 2048)` | f32 | 524,288 |
| `b_stage_x2_vmem` | `(2, 32, 2, 1024)` | bf16 | 262,144 |
| `a2a_s_acc_stage_x3_vmem` | `(3, 32, 2, 1024)` | bf16 | 393,216 |
| `a2a_g_acc_vmem` | `(2, 8, 4, 2, 4096)` | bf16 | 1,048,576 |
| `b_se_tokens_vmem` | `(2, 2, 8, 2, 1024)` | bf16 | 131,072 |
| `b_se_w1_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | 16,777,216 |
| `b_se_w3_x2_vmem` | `(2, 2, 1024, 2048)` | bf16 | 16,777,216 |
| `b_se_w2_x2_vmem` | `(2, 2, 2048, 1024)` | bf16 | 16,777,216 |
| `b_se_acc_vmem` | `(2, 8, 8192)` | f32 | 524,288 |
| **Total VMEM** | | | **~87 MB** |

VMEM budget: 96 MB (set via `vmem_limit_bytes`).

### SMEM Allocation

| Scratch Buffer | Shape | Dtype | Bytes |
|---------------|-------|-------|-------|
| `t2e_routing_x2_smem` | `(2, 8, 128)` | i32 | 8,192 |
| `d2e_count_x2_smem` | `(2, 4, 1, 256)` | i32 | 8,192 |
| `expert_offsets_x2_smem` | `(2, 2, 256)` | i32 | 4,096 |
| `expert_starts_x2_smem` | `(2, 1, 256)` | i32 | 2,048 |
| `expert_sizes_x2_smem` | `(2, 1, 256)` | i32 | 2,048 |
| `a2a_s_sends_x2_smem` | `(64,)` | i32 | 256 |
| **Total SMEM** | | | **~25 KB** |

---

## 6. Key Data Flow Transformations

```
Tokens:      (64, 8192) bf16  ──packed──▶  (64, 2, 4096) bf16
                                                    │
                              ┌──────────────────────┼──────────────────────┐
                              │  A2A Scatter                                  │
                              │  Routes each token's top_k=8 copies          │
                              │  to expert devices                           │
                              ▼                                              │
Per-Expert Input:  (dyn_sz, 2, 4096) bf16    dyn_sz ∈ [0, 32]
                              │
                              │  FFN1: (dyn_sz, 8192) @ (8192, 2048) → (dyn_sz, 2048) f32  [Gate]
                              │         (dyn_sz, 8192) @ (8192, 2048) → (dyn_sz, 2048) f32  [Up]
                              │  Act:   silu(gate) * up → (dyn_sz, 2048) f32
                              │  FFN2:  (dyn_sz, 2048) @ (2048, 8192) → (dyn_sz, 8192) bf16
                              │
                              │  A2A Gather
                              ▼
Per-Token Expert Out:  (256, 8, 2, 4096) bf16    (8 experts × 1 token each per global expert slot)
                              │
                              │  Weighted sum + Shared Expert
                              ▼
Output:      (64, 8192) bf16  ◀──unpacked──  (64, 2, 4096) bf16
```

---

## 7. A2A Ring Buffer Architecture

With `expert_buffer_count = 64` (equal to `local_num_experts`), the kernel uses the **batch scatter path**:

- Each local expert gets a dedicated HBM buffer slot in `a2a_s_x2_hbm[slot, :, :, :]`
- All 64 scatter DMAs are issued in one pass (no buffer-reuse barriers needed)
- Each expert's compute loop waits only on its own `recv_x2_sems[local_e_id]`
- Memory: 64 slots × 2 buffers × 32 tokens × 8192 elem × 2 bytes = **~64 MB** for A2A scratch

This eliminates buffer-reuse synchronization between expert iterations at the cost of higher HBM scratch usage (fits within the 3% HBM budget on v7x-96GB = ~2.9 GB).
