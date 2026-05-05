---
title: "RFC-0026: Fused EP-MoE Kernel 分阶段性能分析框架"
status: draft
author: xl
date: 2026-05-05
reviewers: []
---

# RFC-0026: Fused EP-MoE Kernel 分阶段性能分析框架

> 基于 `kernels/_fused_moe_impl.py` 源码的逐阶段 Roofline 理论分析、LLO 验证与消融实验设计
>
> Target: TPU v7x, Pallas kernel

---

## 0. TPU v7x 硬件规格

### 0.1 单芯片硬件参数

| 组件 | 规格 | 说明 |
|------|------|------|
| **HBM 容量** | 192 GB | 片外高带宽存储 |
| **HBM 带宽** | 3690 GB/s | 双向聚合带宽 |
| **VMEM 容量** | 64 MiB | 片上 Scratchpad (软件管理) |
| **SPR** | 4096 个 | 32-bit 标量寄存器 |
| **VPR** | 32 个 × 4 KB | 向量寄存器, 每个 8×128×32bit |
| **MXU** | 2 × MXU | 矩阵乘法单元 (Dual MXU) |
| **MXU 峰值算力** | 2307 TFLOPS (BF16) | 包含 multiply + accumulate |
| **VPU** | 1 个 | 向量处理单元 (elementwise/reduce) |
| **DMA 引擎** | 1 个 | HBM ↔ VMEM 数据搬运 |
| **对齐要求** | 128 元素 | Block 维度必须被 128 整除 |
| **Ridge Point** | ~625 FLOPs/byte | MXU_peak / HBM_BW |

### 0.2 数据流层次

```text
┌──────────────────────────────────────────────────────────┐
│                        HBM (192 GB)                       │
│   权重 (W1, W2, W3), tokens, A2A scratch buffers          │
│                     ↕ 3690 GB/s (DMA)                     │
├──────────────────────────────────────────────────────────┤
│                    VMEM (64 MiB)                          │
│   Tile buffers, 双缓冲权重, 中间激活, 路由元数据           │
│                     ↕ 极高带宽                            │
├──────────────────────────────────────────────────────────┤
│              VPR (32×4KB) / SPR (4096×4B)                │
│   MXU 累加器, VPU 计算临时值                              │
│                     ↕ 1 cycle                             │
├──────────────────────────────────────────────────────────┤
│           MXU (2307 TFLOPS)  │  VPU (elementwise)        │
└──────────────────────────────────────────────────────────┘
```

### 0.3 ICI 互联参数 (多芯片)

| 拓扑 | 单链路带宽 | 说明 |
|------|-----------|------|
| v7x 2D Torus | ~200 GB/s per link | 用于 All2All scatter/gather |
| 全双工 | 双向同时传输 | A2A 可与计算重叠 |

### 0.4 关键符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `T` | 全局 token 数 (num_tokens) | 64–8192 |
| `E` | 全局专家数 (num_experts) | 64–256 |
| `K` | top_k (每 token 激活专家数) | 2–8 |
| `H` | hidden_size | 4096–8192 |
| `I` | intermediate_size (每专家 FFN 中间维度) | 1024–4096 |
| `SE_I` | se_intermediate_size (共享专家 FFN 中间维度) | 1024–4096 |
| `D` | ep_size (设备数 / Expert Parallelism 度) | 1–256 |
| `T_L` | local_num_tokens = T / D | |
| `E_L` | local_num_experts = E / D | |
| `bt` | 外层 token tile | 8–128 |
| `bts` | 内层 token staging tile (expert_ffn 内) | 8–128 |
| `btc` | 计算 token tile (MXU GEMM 的 M 维度) | 8–64 |
| `bf` | intermediate_size tile | 128–2048 |
| `bfc` | intermediate_size 计算 tile | 128–2048 |
| `bd1` | hidden_size tile (FFN1 K 维度) | 128–2048 |
| `bd1c` | hidden_size 计算 tile (FFN1) | 128–2048 |
| `bd2` | hidden_size tile (FFN2 N 维度) | 128–2048 |
| `bd2c` | hidden_size 计算 tile (FFN2) | 128–2048 |
| `bse` | 共享专家 intermediate tile | 128–1024 |
| `t_p` | t_packing = 32 / dtype_bits (BF16=2) | 1–4 |
| `B_w` | 每权重元素字节数 (BF16=2, INT8=1) | 1–4 |
| `B_t` | 每 token 元素字节数 (BF16=2) | 2–4 |

---

## 1. Stage 1: Metadata 计算 (Gate, TopK, Permute, AllReduce)

### 1.1 功能描述

在每个 `bt` tile 的开头, 从 HBM 加载 top-k 路由结果, 计算 token-to-expert 映射, 并通过跨设备 AllReduce 交换路由元数据, 使每个设备知道每个专家将接收多少 token 以及从哪些设备接收.

### 1.2 对应源码

```text
run_bt():
  ├── start_fetch_topk(bt_id)           # HBM → VMEM: topk_weights, topk_ids
  ├── wait_fetch_topk(bt_id)            # 等待 DMA 完成
  ├── 计算 t2e_routing                  # VPU: broadcast compare + reduce
  ├── 计算 expert_sizes                 # VPU: mask reduction
  └── all_reduce_metadata()             # ICI: 跨设备 allgather + prefix sum
```

### 1.3 输入 / 输出

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `topk_weights_hbm` | `(T_L, K)` | F32 | `T_L × K × 4` |
| **Input** | `topk_ids_hbm` | `(T_L, K)` | I32 | `T_L × K × 4` |
| **Output** | `t2e_routing` | `(bt, padded_K)` | I32 | `bt × pad128(K) × 4` |
| **Output** | `expert_sizes` | `(1, padded_E)` | I32 | `pad128(E) × 4` |
| **Output** | `expert_starts` | `(1, padded_E)` | I32 | `pad128(E) × 4` |
| **Output** | `d2e_count` | `(D, 1, padded_E)` | I32 | `D × pad128(E) × 4` |
| **Output** | `expert_offsets` | `(2, padded_E)` | I32 | `2 × pad128(E) × 4` |

### 1.4 计算步骤分解

#### Step 1.1: Fetch TopK (DMA)

```text
T_fetch_topk = (T_L × K × 4 × 2) / HBM_BW
             = (2 × T_L × K × 4) / 3.69e12   [秒]
```

两次 DMA (weights + ids), 每次 `T_L × K × 4` bytes.

#### Step 1.2: 计算路由掩码 (VPU)

```python
# 源码: run_bt() 内
expert_iota = broadcasted_iota(I32, (1, 1, padded_E), 2)
routing_expanded = expand_dims(t2e_routing[:, :K], axis=2)
mask = (routing_expanded == expert_iota).astype(I32)        # (bt, K, padded_E)
expert_sizes = sum(mask, axis=(0,1), keepdims=True)         # (1, padded_E)
```

FLOPs:

```text
FLOPs_routing = bt × K × pad128(E)           # broadcast compare
             + bt × K × pad128(E)            # reduce sum
             = 2 × bt × K × pad128(E)
```

#### Step 1.3: AllReduce Metadata (ICI 通信)

两种路径:

**Power-of-2 设备数** (recursive doubling):

```text
Rounds = log2(D)
每 round 传输: pad128(E) × 4 bytes (单次 remote copy)
T_allreduce = Rounds × (T_latency + pad128(E) × 4 / ICI_BW + T_barrier)
```

**非 Power-of-2** (ring allgather):

```text
Steps = D - 1
T_allreduce = Steps × (T_latency + pad128(E) × 4 / ICI_BW)
```

加上本地 prefix-sum 计算 `starts` 和全局 `sizes`:

```text
FLOPs_prefix_sum = D × pad128(E) × 2    # reduce + conditional accumulate
```

### 1.5 Roofline 公式

```text
Bytes_HBM_stage1 = 2 × T_L × K × 4                    # topk weights + ids read

FLOPs_stage1     = 2 × bt × K × pad128(E)              # routing mask
                 + D × pad128(E) × 2                    # prefix sum

Bytes_ICI_stage1 = log2(D) × pad128(E) × 4             # allreduce (power-of-2)
                   + D × 2 × sync_barrier_bytes          # barrier messages

T_compute_stage1 = FLOPs_stage1 / VPU_peak              # VPU-bound (无 MXU)
T_hbm_stage1     = Bytes_HBM_stage1 / HBM_BW
T_ici_stage1     = Bytes_ICI_stage1 / ICI_BW + log2(D) × T_barrier

T_stage1 = max(T_compute_stage1, T_hbm_stage1) + T_ici_stage1
```

> **瓶颈特征**: 该阶段以 **ICI 延迟 (AllReduce)** 为主要瓶颈, 计算量极小 (纯 VPU), HBM 数据量也很小. 对于 `D=1` (单设备), AllReduce 退化为 noop, 该阶段几乎可忽略.

### 1.6 消融实验设计

| 实验 | 消融 Flag | 预期影响 | 验证目标 |
|------|----------|---------|---------|
| 禁用 AllReduce | `disable_all_reduce_metadata=True` | 消除 ICI 通信 | 量化 AllReduce 占比 |
| 禁用 Barrier | `disable_sync_barrier=True` | 消除 barrier 同步 | 量化 barrier 开销 |
| 变化 `bt` | 调整 block_config.bt | 改变每次 routing 的 token 数 | bt 对 metadata 阶段的影响 |

---

## 2. Stage 2: All2All Dispatch (Scatter)

### 2.1 功能描述

根据 Stage 1 计算的路由信息, 将本地 token 通过 DMA (本地) 或 ICI remote copy (远程) 分发到每个专家对应的设备上的 `a2a_s_x2_hbm` 缓冲区. Kernel 提供两条路径:

- **Batch Scatter**: 当 `expert_buffer_count >= local_num_experts` 时, 一次遍历所有 token 完成分发 (无 buffer 重用 barrier)
- **Pipelined Scatter**: 当 buffer 不足时, 逐专家分发, 每轮重用 buffer slot

### 2.2 对应源码

```text
Batch path:   start_a2a_scatter_batch(bt_sem_id, bt_start)
Pipelined:    start_a2a_scatter(bt_sem_id, e_sem_id, local_e_id, bt_start)
              wait_a2a_scatter_recv(bt_sem_id, e_sem_id, local_e_id)
              wait_a2a_scatter_send(bt_sem_id, e_sem_id, local_e_id)
```

### 2.3 输入 / 输出

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `tokens_hbm` | `(T_L, t_p, H/t_p)` | BF16 | `T_L × H × B_t` |
| **Input** | routing metadata (SMEM) | — | I32 | (已在 SMEM) |
| **Output** | `a2a_s_x2_hbm` | `(E_buf, a2a_max_T, t_p, H/t_p)` | BF16 | per-expert 动态 |

其中 `E_buf = expert_buffer_count`, `a2a_max_T = align(bt × D, bts)`.

### 2.4 数据搬运量分析

每个 token 被路由到 `K` 个专家, 每个副本 `H × B_t` bytes:

```text
Total_bytes_scatter = bt × K × H × B_t
```

其中:

- **本地份额**: 约 `bt × K × H × B_t / D` (1/D 概率路由到本设备)
- **远程份额**: 约 `bt × K × H × B_t × (D-1)/D`

### 2.5 Roofline 公式

**HBM 读 (源 tokens)**:

```text
Bytes_HBM_read_scatter = bt × H × B_t
```

> 注意: 每个 token 虽被路由到 K 个专家, 但源数据只读一次 (DMA scatter 的 src_ref 重叠).

**HBM 写 (目标 a2a_s buffer)**:

```text
Bytes_HBM_write_local  = bt × K × H × B_t / D          # 本地 expert 接收
Bytes_HBM_write_remote = 0                              # 远程写由 ICI 直接落入对端 HBM
```

**ICI 传输**:

```text
Bytes_ICI_scatter = bt × K × H × B_t × (D-1)/D         # 远程发送量
```

> 在均匀路由假设下, 每个设备发送 `bt × K × H × B_t × (D-1)/D` bytes.

**延迟模型**:

```text
T_scatter_dma    = Bytes_HBM_read_scatter / HBM_BW
T_scatter_ici    = Bytes_ICI_scatter / ICI_BW_effective
T_scatter_serial = bt × K × T_per_token_routing         # fori_loop 串行遍历每 token

T_stage2 = max(T_scatter_dma, T_scatter_ici) + T_scatter_serial
```

> **瓶颈特征**: 对于小 `bt` (decode), 该阶段以 **串行路由循环延迟** 为瓶颈 (每 token 的条件分支 + DMA 启动开销). 对于大 `bt` (prefill), 以 **ICI 带宽** 为瓶颈.

### 2.6 消融实验设计

| 实验 | 消融 Flag | 预期影响 | 验证目标 |
|------|----------|---------|---------|
| 禁用 A2A | `disable_a2a=True` | 消除全部 scatter/gather 通信 | 量化 A2A 占 kernel 总时间比例 |
| Batch vs Pipelined | 调整 `_A2A_HBM_FRACTION` | 切换 scatter 路径 | 两种路径的延迟对比 |
| 变化 D | 改变 ep_size | 改变远程/本地比例 | D 对 scatter 延迟的缩放关系 |

---

## 3. Stage 3: Expert Compute (FFN1 + Activation + FFN2)

### 3.1 功能描述

这是 kernel 的核心计算阶段. 对每个本地专家, 从 `a2a_s_x2_hbm` 读取该专家接收的 token, 执行两阶段 FFN:

1. **FFN1 (Gate/Up projection)**: `acc1 = tokens @ W1`, `acc3 = tokens @ W3`
2. **Activation**: `act = activation(acc1) * acc3` (SwiGLU/GeGLU)
3. **FFN2 (Down projection)**: `result = act @ W2`

结果写回 `a2a_s_acc_x2_hbm` 供 Stage 4 gather.

### 3.2 对应源码

```text
expert_ffn(bt_sem_id, e_sem_id, local_e_id):
  ├── for bf_id in 0..num_bf-1:
  │   ├── run_gate_up_slices(bf_id):       # FFN1
  │   │   ├── for bd1_id in 0..num_bd1-1:
  │   │   │   ├── load W1[bd1], W3[bd1] tile → VMEM (双缓冲)
  │   │   │   ├── for token_tile_id:
  │   │   │   │   ├── stage tokens tile from a2a_s_x2_hbm → VMEM
  │   │   │   │   └── dynamic_ffn1(): MXU matmul + accumulate
  │   │   │   └── (prefetch next bd1 weights)
  │   │   └── (prefetch W2 for FFN2)
  │   └── run_down_slices(bf_id):          # FFN2
  │       └── for bd2_id in 0..num_bd2-1:
  │           ├── load W2[bd2] tile → VMEM (双缓冲)
  │           ├── for token_tile_id:
  │           │   ├── activation(acc1, acc3) → act (VPU, fused)
  │           │   ├── dynamic_ffn2(): MXU matmul + accumulate
  │           │   └── store result tile → a2a_s_acc_x2_hbm (三缓冲)
  │           └── (prefetch next bd2 / next expert weights)
  └── (重复 num_bf 次)
```

### 3.3 输入 / 输出

**Per-expert** (设 `M` 为该专家接收的 token 数, 期望值 `M_avg = bt × K / E_L`):

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | tokens tile (from a2a_s) | `(M, t_p, H/t_p)` | BF16 | `M × H × B_t` |
| **Input** | W1 | `(H, I)` | BF16/INT8 | `H × I × B_w` |
| **Input** | W3 | `(H, I)` | BF16/INT8 | `H × I × B_w` |
| **Input** | W2 | `(I, H)` | BF16/INT8 | `I × H × B_w` |
| **Output** | FFN result (to a2a_s_acc) | `(M, t_p, H/t_p)` | BF16 | `M × H × B_t` |

### 3.4 FLOPs 分解

**Per-expert**:

```text
FLOPs_FFN1_gate = 2 × M × H × I            # tokens @ W1
FLOPs_FFN1_up   = 2 × M × H × I            # tokens @ W3
FLOPs_act       = 3 × M × I                 # SiLU(gate) * up (sigmoid + mul + mul)
FLOPs_FFN2      = 2 × M × I × H            # act @ W2

FLOPs_per_expert = 2 × (2 × M × H × I) + 2 × M × I × H + 3 × M × I
                 = 6 × M × H × I + 3 × M × I
                 ≈ 6 × M × H × I            # activation 项可忽略
```

**所有本地专家的总 FLOPs**:

```text
Total_FLOPs_stage3 = E_L × FLOPs_per_expert
                   = E_L × 6 × M_avg × H × I
                   = E_L × 6 × (bt × K / E_L) × H × I
                   = 6 × bt × K × H × I
```

> **关键观察**: 总 FLOPs 与 `E_L` 无关, 仅取决于 `bt × K × H × I`. 这是因为总 token-expert 对数 = `bt × K`, 每个 pair 的计算量 = `6HI`.

### 3.5 HBM 数据搬运量

**权重读取** (per-expert, 每个 bt tile 全量读取):

```text
Bytes_W_per_expert = (H × I + H × I + I × H) × B_w
                   = 3 × H × I × B_w

Total_Bytes_W = E_L × 3 × H × I × B_w
```

> **注意**: `num_bt` 个 bt tile 循环中, 每个 bt tile 都需要重新读取所有权重 (因为 VMEM 无法容纳全部权重). 所以 `num_bt > 1` 时, 权重读取量翻倍.

**Token 读写** (所有专家汇总):

```text
Bytes_tokens_read  = bt × K × H × B_t       # 从 a2a_s 读
Bytes_tokens_write = bt × K × H × B_t       # 写回 a2a_s_acc
```

**总 HBM 搬运**:

```text
Bytes_HBM_stage3 = num_bt × E_L × 3 × H × I × B_w        # 权重 (主导项)
                 + 2 × bt × K × H × B_t                   # tokens (读+写)
```

### 3.6 Roofline 公式

```text
AI_stage3 = Total_FLOPs_stage3 / Bytes_HBM_stage3
          = (6 × bt × K × H × I) / (num_bt × E_L × 3 × H × I × B_w + 2 × bt × K × H × B_t)

# 简化 (忽略 token I/O, 权重主导):
AI_stage3 ≈ (6 × bt × K) / (num_bt × E_L × 3 × B_w)
           = (2 × bt × K) / (num_bt × E_L × B_w)
```

**Decode 场景** (`bt=T_L`, `num_bt=1`):

```text
AI_decode ≈ (2 × T_L × K) / (E_L × B_w)
```

- 例: T_L=64, K=8, E_L=64, B_w=2 → AI = 2×64×8 / (64×2) = 8 FLOPs/byte
- Ridge point = 625 → **严重 HBM bandwidth bound**

**Prefill 场景** (`bt=128`, `num_bt=T_L/bt`):

```text
AI_prefill ≈ (2 × bt × K) / (E_L × B_w)
```

- 例: bt=128, K=8, E_L=64, B_w=2 → AI = 2×128×8 / (64×2) = 16 FLOPs/byte
- 仍然 HBM bandwidth bound, 但比 decode 好 2×

**延迟模型**:

```text
T_compute_stage3 = Total_FLOPs_stage3 / MXU_peak
T_hbm_stage3     = Bytes_HBM_stage3 / HBM_BW

T_stage3 = max(T_compute_stage3, T_hbm_stage3)
         ≈ T_hbm_stage3                          # 几乎总是 HBM-bound
```

**Pipeline 重叠分析**:

Kernel 使用双缓冲 (权重 ping-pong) + 三缓冲 (FFN2 output staging) 来重叠 DMA 与计算:

```text
T_stage3_pipelined = sum_over_experts(
    max(T_weight_load_e, T_compute_e)
) + T_pipeline_drain

# 理想情况 (完美重叠):
T_stage3_ideal = max(
    sum_e(T_weight_load_e),     # DMA 总时间
    sum_e(T_compute_e)          # MXU 总时间
)
```

### 3.7 Per-tile MXU/DMA 细粒度分析

每个 `(expert, bf_id, bd1_id)` 迭代的 micro-ops:

```text
┌──────────────────────────────────────────────────────────────┐
│ FFN1 Inner Tile: (btc, bd1c) × (bd1c, bfc)                  │
│                                                              │
│ DMA:  Load W1[bd1c, bfc] = bd1c × bfc × B_w bytes           │
│       Load W3[bd1c, bfc] = bd1c × bfc × B_w bytes           │
│       Load tokens[btc, bd1c] = btc × bd1c × B_t bytes       │
│                                                              │
│ MXU:  matmul(btc, bd1c, bfc) → 2 × btc × bd1c × bfc FLOPs  │
│       × 2 (W1 + W3)                                         │
│       = 4 × btc × bd1c × bfc FLOPs                          │
│                                                              │
│ AI_tile = 4 × btc × bd1c × bfc                              │
│         / (2 × bd1c × bfc × B_w + btc × bd1c × B_t)         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ FFN2 Inner Tile: (btc, bfc) × (bfc, bd2c)                    │
│                                                              │
│ DMA:  Load W2[bfc, bd2c] = bfc × bd2c × B_w bytes           │
│       Load/Store acc[btc, bd2c] = btc × bd2c × B_t bytes    │
│                                                              │
│ VPU:  activation(acc1[btc, bfc], acc3[btc, bfc])             │
│       = 3 × btc × bfc FLOPs                                 │
│                                                              │
│ MXU:  matmul(btc, bfc, bd2c) → 2 × btc × bfc × bd2c FLOPs  │
│                                                              │
│ AI_tile = 2 × btc × bfc × bd2c                              │
│         / (bfc × bd2c × B_w + btc × bd2c × B_t)             │
└──────────────────────────────────────────────────────────────┘
```

### 3.8 消融实验设计

| 实验 | 消融 Flag | 预期影响 | 验证目标 |
|------|----------|---------|---------|
| 禁用 FFN1 计算 | `disable_dynamic_ffn1=True` | 消除 FFN1 matmul, 保留 DMA | 量化 FFN1 compute vs DMA 平衡 |
| 禁用 FFN2 计算 | `disable_dynamic_ffn2=True` | 消除 FFN2 matmul + activation | 量化 FFN2 compute vs DMA 平衡 |
| 禁用权重加载 | `disable_weight_load=True` | 消除所有权重 DMA | 量化 DMA 占比 (纯计算时间) |
| 禁用 token 读 | `disable_a2a_s_tile_read=True` | 消除 token staging DMA | token staging 开销 |
| 禁用结果写 | `disable_a2a_s_acc_tile_write=True` | 消除 FFN2 output store | output staging 开销 |
| FFN1+FFN2 同时禁 | 两个 flag 同时开 | 保留仅 DMA | 验证纯 DMA pipeline 延迟 |
| 变化 bf/bd1/bd2 | 调整 block_config | 改变 tile 大小 | 找到 DMA/compute 平衡点 |
| 变化 btc | 调整 block_config.btc | 改变 MXU M 维度 | btc 对 MXU 利用率的影响 |

---

## 4. Stage 4: All2All Combine (Gather)

### 4.1 功能描述

将各专家计算结果从 `a2a_s_acc_x2_hbm` 通过 DMA/ICI 收集回每个 token 的原始设备上的 `a2a_g_hbm` 缓冲区, 为 Stage 6 的加权求和做准备. 这是 Stage 2 的反向操作.

### 4.2 对应源码

```text
start_a2a_gather(bt_sem_id, e_sem_id, local_e_id):
  ├── for recv_id in 0..D-1:
  │   ├── if local: async_copy (VMEM ← a2a_s_acc → a2a_g)
  │   └── if remote: async_remote_copy → peer's a2a_g
  └── expert_offsets 更新

wait_a2a_gather_recv_all(bt_sem_id):
  └── for e_id in 0..E-1:
      └── if d2e_count[my_id, e_id] > 0: wait recv sem
```

### 4.3 输入 / 输出

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `a2a_s_acc_x2_hbm` | `(E_buf, a2a_max_T, t_p, H/t_p)` | BF16 | 动态 |
| **Output** | `a2a_g_hbm` | `(E, bt, t_p, H/t_p)` | BF16 | `E × bt × H × B_t` |

### 4.4 数据搬运量分析

与 Stage 2 对称:

```text
Total_bytes_gather = bt × K × H × B_t

Bytes_ICI_gather   = bt × K × H × B_t × (D-1)/D    # 远程传输
Bytes_local_gather = bt × K × H × B_t / D           # 本地 DMA
```

### 4.5 Roofline 公式

```text
T_gather_ici   = Bytes_ICI_gather / ICI_BW_effective
T_gather_local = Bytes_local_gather / HBM_BW

T_stage4 = max(T_gather_ici, T_gather_local) + T_gather_serial
```

> **关键**: Stage 4 在实际 kernel 中与 Stage 3 **流水线重叠** — 每个专家完成 FFN 后立即启动 gather, 下一个专家的 FFN 与上一个专家的 gather 并行. 因此 Stage 4 的延迟大部分被 Stage 3 隐藏.

### 4.6 重叠分析

```text
# Pipelined path: 逐专家 scatter → FFN → gather 流水线
# Batch path: 一次性 scatter → 逐专家 FFN+gather → 一次性等 gather

T_stage3_4_batch = T_scatter_batch
                 + sum_e(max(T_ffn_e, T_gather_e_prev))
                 + T_gather_drain

T_stage3_4_pipelined = sum_e(
    T_scatter_e + max(T_ffn_e, T_gather_e_prev) + T_scatter_send_wait_e
)
```

### 4.7 消融实验设计

| 实验 | 消融 Flag | 预期影响 | 验证目标 |
|------|----------|---------|---------|
| 禁用 A2A | `disable_a2a=True` | 消除全部 scatter+gather | Stage 2+4 的合计占比 |
| 单设备 | `ep_size=1` | 消除所有 ICI 通信 | 纯本地 DMA copy 的 gather 开销 |

---

## 5. Stage 5: Shared Expert Compute

### 5.1 功能描述

与路由专家并行执行的共享专家 (Shared Expert) FFN, 所有 token 都通过共享专家, 其输出在 Stage 6 与路由专家输出相加. 共享专家的计算被切分为 `se_total_blocks = ceil(SE_I / bse)` 个块, 交错插入到 Stage 3 的专家循环中, 以最大化计算与通信的重叠.

### 5.2 对应源码

```text
run_shared_expert_slice(block_id, bt_id, bt_sem_id, out_buf_id):
  ├── FFN1: for bd1_idx in 0..num_bd1-1:
  │   ├── load tokens slice → b_se_tokens_vmem (双缓冲)
  │   ├── load W1_shared[bd1, block_id] → b_se_w1_x2_vmem (双缓冲)
  │   ├── load W3_shared[bd1, block_id] → b_se_w3_x2_vmem (双缓冲)
  │   └── matmul: gate_acc += tokens @ W1_shared, up_acc += tokens @ W3_shared
  ├── apply scale (if quantized)
  ├── activation: act = activation(gate_acc, up_acc)
  └── FFN2: for bd2_idx in 0..num_bd2-1:
      ├── load W2_shared[block_id, bd2] → b_se_w2_x2_vmem (双缓冲)
      └── matmul: acc += act @ W2_shared → b_se_acc_vmem
```

### 5.3 输入 / 输出

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `tokens_hbm` (local) | `(bt, t_p, H/t_p)` | BF16 | `bt × H × B_t` |
| **Input** | W1_shared | `(H, SE_I)` | BF16/INT8 | `H × SE_I × B_w` |
| **Input** | W3_shared | `(H, SE_I)` | BF16/INT8 | `H × SE_I × B_w` |
| **Input** | W2_shared | `(SE_I, H)` | BF16/INT8 | `SE_I × H × B_w` |
| **Output** | `b_se_acc_vmem` | `(bt, H)` | F32 | `bt × H × 4` |

### 5.4 FLOPs 分解

```text
FLOPs_SE_FFN1 = 2 × bt × H × SE_I × 2       # W1 + W3
              = 4 × bt × H × SE_I
FLOPs_SE_act  = 3 × bt × SE_I                # activation
FLOPs_SE_FFN2 = 2 × bt × SE_I × H

Total_FLOPs_SE = 6 × bt × H × SE_I + 3 × bt × SE_I
               ≈ 6 × bt × H × SE_I
```

### 5.5 HBM 数据搬运量

```text
Bytes_SE_weights = 3 × H × SE_I × B_w                    # W1 + W3 + W2
Bytes_SE_tokens  = se_total_blocks × num_bd1 × bt × bd1 × B_t   # tokens re-read per block
                 = se_total_blocks × bt × H × B_t         # 每个 block 全量读 H
Bytes_SE_output  = bt × H × 4                             # F32 output to VMEM (not HBM)

Total_Bytes_HBM_SE = Bytes_SE_weights + Bytes_SE_tokens
```

### 5.6 Roofline 公式

```text
AI_SE = Total_FLOPs_SE / Total_Bytes_HBM_SE
      = (6 × bt × H × SE_I) / (3 × H × SE_I × B_w + se_total_blocks × bt × H × B_t)

T_compute_SE = Total_FLOPs_SE / MXU_peak
T_hbm_SE     = Total_Bytes_HBM_SE / HBM_BW

T_stage5 = max(T_compute_SE, T_hbm_SE)
```

> **关键**: SE 计算被切分为多个 slice, 交错在 Stage 3 的专家循环中执行. 理想情况下, SE 与路由专家的 A2A 等待时间完全重叠:

```text
# 交错调度模型:
per_expert_schedule:
  ├── se_before slices  (overlaps with scatter wait)
  ├── expert_ffn        (MXU compute)
  ├── start_gather
  └── se_after slices   (overlaps with gather + next scatter)

T_overlap_gain = min(T_stage5, T_scatter_wait + T_gather_overlap)
T_stage5_effective = max(0, T_stage5 - T_overlap_gain)
```

### 5.7 消融实验设计

| 实验 | 消融 Flag | 预期影响 | 验证目标 |
|------|----------|---------|---------|
| 禁用 SE | `disable_shared_expert=True` | 消除全部 SE 计算 | SE 的净延迟增量 (扣除重叠) |
| 变化 bse | 调整 block_config.bse | 改变 SE 切片粒度 | bse 对交错效率的影响 |
| SE alone | 禁 A2A + 禁路由专家 | 孤立 SE 计算 | SE 的裸延迟 |

---

## 6. Stage 6: MoE 结果 Top-K 加权求和 + 共享专家输出

### 6.1 功能描述

从 `a2a_g_hbm` 收集每个 token 的 `K` 个专家输出, 按 topk_weights 加权求和, 加上 Stage 5 的共享专家输出, 写入最终 `output_hbm`.

### 6.2 对应源码

```text
acc_and_store_output(bt_sem_id, out_buf_id):
  ├── for acc_tile in 0..num_acc_tiles-1:   # 双缓冲 pipeline
  │   ├── start_load_acc_bt(tile_start):     # 从 a2a_g_hbm 逐 token 加载
  │   │   └── for t_i in tile:
  │   │       └── for k in 0..K-1:
  │   │           └── DMA copy: a2a_g_hbm[e_id, offset] → a2a_g_acc_vmem[k, t_i]
  │   ├── wait_load_acc_bt()
  │   └── acc_gather_to_output():
  │       ├── for k in 0..K-1:
  │       │   └── output += a2a_g_acc_vmem[k] * topk_weights[k]   # VPU
  │       ├── if shared_expert: output += b_se_acc_vmem            # VPU add
  │       └── store → b_output_x2_vmem
  └── start_send_bo() → output_hbm                                # DMA write
```

### 6.3 输入 / 输出

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `a2a_g_hbm` | `(E, bt, t_p, H/t_p)` | BF16 | per-token: `K × H × B_t` |
| **Input** | `topk_weights` | `(bt, K)` | F32 | `bt × K × 4` |
| **Input** | `b_se_acc_vmem` | `(bt, H)` | F32 | `bt × H × 4` (已在 VMEM) |
| **Output** | `output_hbm` | `(T_L, H)` | BF16 | `T_L × H × B_t` |

### 6.4 FLOPs 分解

```text
FLOPs_weighted_sum = bt × K × H × 2         # multiply + accumulate per (token, expert, hidden)
FLOPs_se_add       = bt × H                  # add shared expert output
FLOPs_dtype_cast   = bt × H                  # F32 → BF16

Total_FLOPs_stage6 = bt × H × (2K + 2)
                   ≈ 2 × bt × K × H          # K 项主导
```

### 6.5 HBM 数据搬运量

```text
Bytes_HBM_read_stage6  = bt × K × H × B_t              # 读取 K 个专家输出
Bytes_HBM_write_stage6 = bt × H × B_t                   # 写最终输出

Total_Bytes_HBM_stage6 = bt × H × B_t × (K + 1)
```

> 注意: 每个 token 的 K 个专家输出分散在 `a2a_g_hbm` 的不同 expert 维度, DMA 是逐 token 逐 expert 的小块拷贝 (1 × t_p × H/t_p), 无法 coalesce.

### 6.6 Roofline 公式

```text
AI_stage6 = Total_FLOPs_stage6 / Total_Bytes_HBM_stage6
          = (2 × bt × K × H) / (bt × H × B_t × (K + 1))
          = 2K / (B_t × (K + 1))

# BF16: AI = 2K / (2(K+1)) = K/(K+1) ≈ 1 FLOPs/byte (K=8 → 0.89)
```

> **瓶颈特征**: 严重 **HBM bandwidth bound** (AI ≈ 1, 远低于 ridge point 625). 但该阶段数据量较小 (相比 Stage 3), 绝对延迟不高. 主要延迟来自 **小 DMA 串行化** — 逐 token 逐 expert 的 gather 是 `fori_loop` 串行发起.

### 6.7 延迟模型

```text
T_compute_stage6 = Total_FLOPs_stage6 / VPU_peak        # VPU-only (无 MXU)
T_hbm_stage6     = Total_Bytes_HBM_stage6 / HBM_BW
T_serial_dma     = bt × K × T_per_dma_launch            # 串行 DMA 启动开销

T_stage6 = max(T_hbm_stage6, T_serial_dma) + T_compute_stage6
```

### 6.8 消融实验设计

| 实验 | 消融方法 | 预期影响 | 验证目标 |
|------|---------|---------|---------|
| 变化 K | 调整 top_k | 线性改变 gather 数量 | K 对 Stage 6 的缩放 |
| 禁 SE | `disable_shared_expert=True` | 消除 SE add 分支 | SE add 的边际开销 |
| 变化 acc_bt | 调整 math.gcd(bt, 16) | 改变 gather tile 大小 | 小 tile DMA 效率 |

---

## 7. 端到端 Roofline 汇总

### 7.1 总延迟模型

```text
T_total = T_stage1
        + max(T_stage2, 0)                        # scatter (可能与 stage1 部分重叠)
        + max(T_stage3, T_stage4, T_stage5_eff)    # FFN + gather + SE (流水线重叠)
        + T_stage6
        + T_output_drain                           # 最后两个 bt tile 的 output DMA

# 简化 (pipeline 理想重叠):
T_total ≈ T_stage1 + T_stage3 + T_stage6
         + max(0, T_stage2 - T_stage3_overlap)     # scatter 未被隐藏的部分
         + max(0, T_stage5 - T_scatter_gather_gap)  # SE 未被隐藏的部分
```

### 7.2 各阶段 Roofline 特征汇总

| Stage | FLOPs | Bytes (HBM) | AI | 瓶颈类型 | 计算单元 |
|-------|-------|-------------|-----|---------|---------|
| 1. Metadata | `2btK·pad(E)` + `2D·pad(E)` | `2T_LK×4` | 极低 | ICI 延迟 | VPU |
| 2. Scatter | 0 (纯 DMA) | `btH·B_t` (read) | 0 | ICI BW / 串行 DMA | DMA+ICI |
| 3. Expert FFN | `6btKHI` | `E_L·3HI·B_w` + `2btKH·B_t` | `2btK/(E_L·B_w)` | **HBM BW** | MXU |
| 4. Gather | 0 (纯 DMA) | `btKH·B_t` (transfer) | 0 | ICI BW / 串行 DMA | DMA+ICI |
| 5. Shared Expert | `6btH·SE_I` | `3H·SE_I·B_w + ...` | 中等 | HBM BW | MXU |
| 6. Output Acc | `2btKH` | `btH·B_t·(K+1)` | `K/(K+1)` | HBM BW (小 DMA) | VPU |

### 7.3 关键观察

1. **Stage 3 (Expert FFN) 主导**: 占总计算量 >95%, 占总 HBM 搬运 >80%. 权重读取是主要瓶颈.

2. **AI 随 `bt × K / E_L` 增长**: 增大 `bt` 或 `K` 提高计算密度; 减少 `E_L` (通过增大 `D`) 同样有效, 但会增加 ICI 通信.

3. **Pipeline 重叠是关键**: SE 与 A2A 的重叠效率直接影响端到端延迟. `bse` 的选择决定了 SE slice 粒度, 过大导致交错不充分, 过小导致 DMA 启动开销增大.

4. **Decode vs Prefill 差异显著**:

   - Decode: `bt` 小 → AI 低 → 严格 HBM BW bound, 但 A2A 数据量小
   - Prefill: `bt` 大 → AI 提高但仍 BW bound, A2A 数据量大可能成为 ICI 瓶颈

---

## 8. 完整消融实验矩阵

### 8.1 消融 Flag 总表

kernel 内置的消融开关 (编译期 `static_argnames`):

| Flag | 默认值 | 影响阶段 | 作用 |
|------|--------|---------|------|
| `disable_a2a` | False | 2, 4 | 禁用全部 All2All 通信 |
| `disable_dynamic_ffn1` | False | 3 | 禁用 FFN1 matmul (保留 DMA) |
| `disable_dynamic_ffn2` | False | 3 | 禁用 FFN2 matmul + activation |
| `disable_weight_load` | False | 3, 5 | 禁用全部权重 DMA |
| `disable_a2a_s_tile_read` | False | 3 | 禁用 token staging read (a2a_s → VMEM) |
| `disable_a2a_s_acc_tile_write` | False | 3 | 禁用 FFN output write (VMEM → a2a_s_acc) |
| `disable_shared_expert` | False | 5 | 禁用共享专家计算 |
| `disable_all_reduce_metadata` | False | 1 | 禁用跨设备 AllReduce (仅 ep_size=1 或 disable_a2a 时安全) |
| `disable_sync_barrier` | False | 1 | 禁用全局 barrier 同步 |

### 8.2 推荐消融实验序列

按从粗到细的顺序执行:

#### Level 0: Baseline

```python
# 完整 kernel, 无消融
fused_ep_moe(mesh, tokens, w1, w2, w3, topk_weights, topk_ids, top_k, ...)
```

#### Level 1: 粗粒度阶段隔离

```python
# Exp 1.1: 禁 A2A → 隔离 Stage 3+5+6
fused_ep_moe(..., disable_a2a=True, disable_all_reduce_metadata=True, disable_sync_barrier=True)

# Exp 1.2: 禁 SE → 隔离 Stage 1+2+3+4+6
fused_ep_moe(..., disable_shared_expert=True)

# Exp 1.3: 禁 A2A + 禁 SE → 纯 Expert FFN + Output
fused_ep_moe(..., disable_a2a=True, disable_shared_expert=True,
             disable_all_reduce_metadata=True, disable_sync_barrier=True)
```

#### Level 2: Stage 3 内部隔离

```python
# Exp 2.1: 纯 DMA (禁全部计算)
fused_ep_moe(..., disable_dynamic_ffn1=True, disable_dynamic_ffn2=True,
             disable_a2a=True, disable_shared_expert=True,
             disable_all_reduce_metadata=True, disable_sync_barrier=True)

# Exp 2.2: 纯 compute (禁全部 DMA)
fused_ep_moe(..., disable_weight_load=True, disable_a2a_s_tile_read=True,
             disable_a2a_s_acc_tile_write=True,
             disable_a2a=True, disable_shared_expert=True,
             disable_all_reduce_metadata=True, disable_sync_barrier=True)

# Exp 2.3: 仅 FFN1
fused_ep_moe(..., disable_dynamic_ffn2=True,
             disable_a2a=True, disable_shared_expert=True,
             disable_all_reduce_metadata=True, disable_sync_barrier=True)

# Exp 2.4: 仅 FFN2
fused_ep_moe(..., disable_dynamic_ffn1=True,
             disable_a2a=True, disable_shared_expert=True,
             disable_all_reduce_metadata=True, disable_sync_barrier=True)
```

#### Level 3: Pipeline 效率分析

```python
# Exp 3.1: 变化 bt (固定其他参数)
for bt in [8, 16, 32, 64, 128]:
    fused_ep_moe(..., block_config=FusedMoEBlockConfig(bt=bt, ...))

# Exp 3.2: 变化 bse (SE 交错粒度)
for bse in [128, 256, 512, 1024]:
    fused_ep_moe(..., block_config=FusedMoEBlockConfig(..., bse=bse))

# Exp 3.3: 变化 bf (intermediate tile)
for bf in [256, 512, 1024, 2048]:
    fused_ep_moe(..., block_config=FusedMoEBlockConfig(..., bf=bf))
```

### 8.3 Profile 指标采集

每个实验需采集:

| 指标 | 来源 | 用途 |
|------|------|------|
| 端到端延迟 (ms) | `jax.block_until_ready` + timer | 主要指标 |
| HBM 读写量 (GB) | XLA profiler / LLO IR | 验证理论公式 |
| MXU 利用率 (%) | XLA profiler | Stage 3 效率 |
| ICI 利用率 (%) | XLA profiler | Stage 2+4 通信效率 |
| VMEM 峰值占用 | LLO IR 分析 | 是否接近 64 MiB 上限 |
| Pipeline bubble ratio | `T_total - T_critical_path` / `T_total` | 流水线效率 |

---

## 9. LLO IR 分析指南

### 9.1 从 LLO 验证 HBM 搬运量

```bash
# 获取 LLO IR (使用 profile-kernel skill 或手动 dump)
# 搜索 DMA 指令, 统计各阶段的实际字节数

# Stage 3 权重加载:
grep -c "dma.*w1_hbm\|dma.*w2_hbm\|dma.*w3_hbm" kernel.llo

# Stage 2/4 A2A:
grep -c "dma.*a2a_s\|remote_dma.*a2a" kernel.llo
```

### 9.2 从 LLO 验证 Pipeline 重叠

检查 DMA 和 MXU 指令的交错模式:

```text
# 理想: DMA(weight_n+1) 与 MXU(compute_n) 并行
# 不良: MXU stall 等待 DMA 完成
```

### 9.3 从 Trace 验证阶段边界

使用 XLA profiler trace 识别各阶段:

```text
sync_barrier              → Stage 1 开始
start_a2a_scatter_batch   → Stage 2 开始
expert_ffn first call     → Stage 3 开始
start_a2a_gather first    → Stage 4 开始 (与 Stage 3 交错)
acc_and_store_output      → Stage 6 开始
start_send_bo             → Stage 6 结束
```

---

## 10. 数值验证示例

### 10.1 DeepSeek-V3-like 配置 (Decode)

| 参数 | 值 |
|------|------|
| T=256, E=256, K=8, H=8192, I=2048 | |
| D=4, BF16, bt=64 | |

```text
T_L = 256/4 = 64, E_L = 256/4 = 64
M_avg = 64 × 8 / 64 = 8 tokens/expert

Stage 3 FLOPs = 6 × 64 × 8 × 8192 × 2048 = 51.5 TFLOPS
Stage 3 Weight bytes = 64 × 3 × 8192 × 2048 × 2 = 6.44 GB
AI = 51.5e12 / 6.44e9 = 8.0 FLOPs/byte

T_compute = 51.5e12 / 2307e12 = 22.3 μs
T_hbm     = 6.44e9 / 3690e9 = 1745 μs
T_stage3  ≈ 1745 μs (HBM BW bound, MXU 利用率 ≈ 1.3%)
```

### 10.2 Prefill 配置

| 参数 | 值 |
|------|------|
| T=2048, E=256, K=8, H=8192, I=2048 | |
| D=4, BF16, bt=128 | |

```text
T_L = 2048/4 = 512, E_L = 64
num_bt = 512/128 = 4
M_avg = 128 × 8 / 64 = 16 tokens/expert

Stage 3 FLOPs = 6 × 128 × 8 × 8192 × 2048 = 103 TFLOPS  (per bt tile)
Stage 3 Weight bytes = 64 × 3 × 8192 × 2048 × 2 = 6.44 GB (per bt tile)
Total (4 tiles) = 412 TFLOPS / 25.8 GB

AI = 103e12 / 6.44e9 = 16.0 FLOPs/byte (per tile)

T_compute_total = 412e12 / 2307e12 = 178 μs
T_hbm_total     = 25.8e9 / 3690e9 = 6990 μs
T_stage3 ≈ 6990 μs (HBM BW bound, MXU 利用率 ≈ 2.5%)
```

> **结论**: Decode 和 Prefill 均严格 HBM BW bound. MXU 利用率极低 (< 3%), 主要瓶颈是逐专家权重读取. 优化方向: 增大 `bt×K/E_L` (通过 EP degree, batch size, 或权重量化减少 B_w).

---

## 附录 A: 符号速查表

| 符号 | 公式 | 说明 |
|------|------|------|
| `pad128(x)` | `ceil(x/128) × 128` | 128 对齐 |
| `align_to(x, a)` | `ceil(x/a) × a` | 向上对齐 |
| `cdiv(x, a)` | `ceil(x/a)` | 向上取整除法 |
| `t_p` | `32 / dtype_bits` | 类型打包因子 |
| `M_avg` | `bt × K / E_L` | 每专家平均 token 数 |
| `num_bt` | `T_L / bt` | 外层循环次数 |
| `num_bf` | `I / bf` | intermediate 分块数 |
| `num_bd1` | `H / bd1` | hidden FFN1 分块数 |
| `num_bd2` | `H / bd2` | hidden FFN2 分块数 |
| `se_total_blocks` | `ceil(SE_I / bse)` | SE intermediate 分块数 |
| `a2a_max_T` | `align(bt × D, bts)` | A2A buffer 最大 token 数 |

## 附录 B: 消融 Flag 组合安全表

| 组合 | 安全? | 约束 |
|------|-------|------|
| `disable_a2a=True` alone | Yes (ep_size=1) | ep_size>1 时需同时禁 metadata + barrier |
| `disable_all_reduce_metadata=True` alone | **No** | 必须 `disable_a2a=True` 或 `ep_size=1` |
| `disable_sync_barrier=True` alone | **No** | 必须 `disable_a2a=True` 或 `ep_size=1` |
| `disable_dynamic_ffn1=True` + `disable_dynamic_ffn2=True` | Yes | 保留 DMA pipeline, 输出全零 |
| `disable_weight_load=True` alone | Yes | matmul 使用未初始化权重 |
| `disable_shared_expert=True` alone | Yes | 无副作用 |
| 全部禁用 (所有 flag=True) | Yes (ep_size=1) | 输出全零, 仅测 kernel overhead |

---

*文档版本: v1.0 | 创建日期: 2026-05-05 | 基于 `kernels/_fused_moe_impl.py` 源码分析*
