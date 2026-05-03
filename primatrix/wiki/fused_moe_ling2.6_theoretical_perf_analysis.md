# Fused MoE Kernel 理论性能分析 — Ling 2.6 1T Decode

> 基于 `fusedMoE_ling2.6.yaml` 配置与 LLO IR dump 的逐阶段理论分析
>
> TPU v7x, 2x2x1 topology (4 chips), BF16, ep_size=4

---

## 1. 配置参数总览

### 1.1 模型参数

| 参数 | 值 | 说明 |
|------|------|------|
| `num_tokens` | 256 (global) | Decode batch size |
| `num_experts` | 256 | 路由专家数 |
| `top_k` | 8 | 每 token 激活专家数 |
| `hidden_size` | 8192 | 隐藏层维度 |
| `intermediate_size` | 2048 | 专家 FFN 中间维度 |
| `se_intermediate_size` | 2048 | 共享专家中间维度 |
| `dtype` | BF16 | 激活/权重精度 |
| `act_fn` | SiLU | SwiGLU 门控激活 |
| `ep_size` | 4 | Expert Parallelism 分片数 |

### 1.2 编译后 Block Config

从 LLO kernel 名称
`fused-moe-k_8-bt_64_64_64-bf_1024_1024-bd1_2048_2048-bd2_2048_2048-shared_expert_bse_256`
解析得到:

| Block 参数 | 值 | 含义 |
|------------|------|------|
| `bt` | 64 | 外层 token 分块 (routing/output 维度) |
| `bts` | 64 | 内层 token staging 分块 (expert_ffn 内 HBM<->VMEM 搬运) |
| `btc` | 64 | 计算分块 (MXU GEMM 的 M 维度) |
| `bf` | 1024 | intermediate_size 分块 |
| `bfc` | 1024 | intermediate_size 计算分块 |
| `bd1` | 2048 | hidden_size 分块 (FFN1 K 维度) |
| `bd1c` | 2048 | hidden_size 计算分块 (FFN1) |
| `bd2` | 2048 | hidden_size 分块 (FFN2 N 维度) |
| `bd2c` | 2048 | hidden_size 计算分块 (FFN2) |
| `bse` | 256 | 共享专家 intermediate 分块 |

### 1.3 派生维度

| 派生量 | 公式 | 值 |
|--------|------|------|
| `local_num_tokens` | 256 / 4 | **64** |
| `local_num_experts` | 256 / 4 | **64** |
| `num_bt` | 64 / 64 | **1** (外层循环次数) |
| `num_bf` | 2048 / 1024 | **2** |
| `num_bd1` | 8192 / 2048 | **4** |
| `num_bd2` | 8192 / 2048 | **4** |
| `t_packing` | 64 / 32 | **2** (BF16 打包因子) |
| `h_per_t_packing` | 8192 / 2 | **4096** |
| `bd1_per_t_packing` | 2048 / 2 | **1024** |
| `bd2_per_t_packing` | 2048 / 2 | **1024** |
| `a2a_max_tokens` | align(64×4, 64) | **256** |
| `expert_buffer_count` | min(64, HBM_budget) | **64** (= local_num_experts) |
| `se_total_blocks` | ceil(2048 / 256) | **8** |

> **关键推论**: `expert_buffer_count == local_num_experts`，因此内核走 **Batch Scatter 路径** — 一次性 scatter 所有 token，而非逐专家 scatter。
>
> **相比 bt=32 的关键变化**: `num_bt` 从 2 降至 1 — 外层循环只执行一次，权重读取总量减半！

---

## 2. 执行流程与阶段划分

整个 kernel 的执行以 `bt` 为外循环单位 (`num_bt=1`)，只有 1 个 bt tile 的执行流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│ run_bt(bt_id=0)                                                 │
│                                                                 │
│  ┌──── Stage 0 ────┐                                            │
│  │ Prefetch topk    │  (前一 bt tile 结束时已发起)                │
│  │ Prefetch SE tok  │                                            │
│  └────────┬─────────┘                                            │
│           ▼                                                      │
│  ┌──── Stage 1 ────────────────────────────────────────────┐    │
│  │ Wait topk → 计算 t2e_routing, expert_sizes              │    │
│  │ all_reduce_metadata() → 跨设备交换路由元数据             │    │
│  │ sync_barrier()                                           │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           ▼                                                      │
│  ┌──── Stage 2 ────────────────────────────────────────────┐    │
│  │ start_a2a_scatter_batch()                                │    │
│  │   64 tokens × 8 experts → 分发至本地/远程设备            │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           ▼                                                      │
│  ┌──── Stage 3+4 ──────────────────────────────────────────┐    │
│  │ for local_e_id in 0..63:  (核心循环)                     │    │
│  │   ├─ run_shared_expert_slice()  [交错 SE FFN]           │    │
│  │   ├─ wait_a2a_scatter_recv(expert_id)                   │    │
│  │   ├─ expert_ffn(expert_id):                             │    │
│  │   │   ├─ for bf_id in 0..1:                             │    │
│  │   │   │   ├─ FFN1: for bd1_id in 0..3:                  │    │
│  │   │   │   │   Load W1,W3 tile → VMEM → MXU matmul      │    │
│  │   │   │   └─ FFN2: for bd2_id in 0..3:                  │    │
│  │   │   │       Load W2 tile → act() → MXU matmul → HBM  │    │
│  │   │   └─                                                │    │
│  │   ├─ start_a2a_gather(expert_id)                        │    │
│  │   └─ run_shared_expert_slice()  [交错 SE FFN]           │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           ▼                                                      │
│  ┌──── Stage 5 ────────────────────────────────────────────┐    │
│  │ wait_a2a_scatter_send_batch()                            │    │
│  │ wait_a2a_gather_recv_all()                               │    │
│  │ sync_barrier()                                           │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           ▼                                                      │
│  ┌──── Stage 6 ────────────────────────────────────────────┐    │
│  │ acc_and_store_output():                                  │    │
│  │   Load 8 expert outputs per token → weighted sum         │    │
│  │   + SE output → 写回 output HBM                         │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           ▼                                                      │
│  ┌──── Stage 7 ────┐                                            │
│  │ start_send_bo() │  DMA output VMEM → HBM                     │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

**依赖关系汇总:**

```
Stage 0 (prefetch) ─┐
                     ├─→ Stage 1 (routing) ─→ Stage 2 (scatter) ─→ Stage 3 (expert FFN)
sync_barrier ────────┘                                               │     ↑↓ interleaved
                                                          Stage 4 (SE FFN)─┘
                                                                     │
                                                         Stage 5 (wait comm) ─→ Stage 6 (output acc)
                                                                                      │
                                                                              Stage 7 (output store)
```

---

## 3. 逐阶段详细分析

### 3.0 初始化 (Kernel Prologue)

**操作**: `sync_barrier()` + `start_fetch_and_wait_se_scales()` (BF16 无量化, SE scale 加载为 no-op)

| 指标 | 值 |
|------|------|
| HBM 读取 | 0 |
| 通信 | 全局 barrier (4 设备) |
| 估计耗时 | ~5 μs (barrier latency) |

**LLO 观测**: 利用文件前 ~100 个 bundle 全部为 SALU-only (标量地址计算 + operand pointer spill to SMEM)，MXU/VALU 完全空闲。

---

### 3.1 Stage 1: 路由与元数据交换

**操作**:
1. 从 HBM 读取 `topk_weights` 和 `topk_ids` (各 bt×top_k)
2. 计算 `t2e_routing`: 将 topk_ids 展开为 (bt, padded_top_k) 矩阵
3. 计算 `expert_sizes`: 按专家 ID 聚合 token 计数
4. `all_reduce_metadata()`: 跨 4 设备递归倍增 allgather 专家路由元数据

**数据大小**:

| 数据 | Shape | 大小 |
|------|-------|------|
| topk_weights | (64, 8) F32 | 2 KB |
| topk_ids | (64, 8) S32 | 2 KB |
| expert_sizes (per device) | (1, 256) S32 | 1 KB |
| all_reduce 通信量 | 4 设备 × 1 KB | 4 KB total |

**计算强度分析**:

| 指标 | 值 |
|------|------|
| HBM 读取 | 4 KB |
| VPU 计算 | ~130K ops (mask 生成 + reduce) |
| ICI 通信 | ~4 KB (recursive doubling, 2 rounds for 4 devices) |
| Compute unit | VPU only (无 MXU) |
| **瓶颈** | **通信延迟** (barrier + allgather latency ~10-20 μs) |

**LLO 观测 (Critical Path)**: `region13` 显示 critical path length = 201，主要是
`vrot.slane`, `vbcast.lane`, `vpop`, 和比较/选择操作链 — 对应路由矩阵生成 (`_fused_moe_impl.py:2661-2666`)。

---

### 3.2 Stage 2: All-to-All Scatter

**操作**: `start_a2a_scatter_batch()` — 一次性遍历 bt=64 个 token，将每个 token 发送至其 top_k=8 个目标专家所在设备。

**数据大小**:

| 数据 | 计算 | 大小 |
|------|------|------|
| 单 token 大小 | t_packing × h_per_t_packing × 2B | **16 KB** |
| Scatter 总量 (上界) | 64 tokens × 8 copies × 16 KB | **8 MB** |
| 本地 DMA 占比 | local_experts / total_experts = 64/256 | 25% (~2 MB) |
| 远程 ICI 占比 | 75% | ~6 MB |

**通信模式**:

```
Device 0 ──token[i]──→ Device 0 (local, 25%)
         ──token[i]──→ Device 1 (ICI)
         ──token[i]──→ Device 2 (ICI)    } 75% remote
         ──token[i]──→ Device 3 (ICI)
```

| 指标 | 值 |
|------|------|
| HBM 读取 (source tokens) | 64 × 16 KB = **1 MB** |
| HBM 写入 (dest buffers) | ~8 MB (distributed across devices) |
| ICI 传输 | ~6 MB per device |
| Compute | 0 (pure data movement) |
| **瓶颈** | **ICI latency** + DMA setup overhead |
| 估计耗时 | ~30-50 μs |

> **注意**: Scatter 使用 `lax.fori_loop` (非展开) 遍历 64 个 token，每个 token 内展开 8 个 top_k 路由。DMA 发起后不等待完成——接收端在 `wait_a2a_scatter_recv()` 中等待。

---

### 3.3 Stage 3: 专家 FFN 计算 (核心瓶颈)

这是内核的**绝对瓶颈阶段**。对 64 个本地专家逐一执行 FFN 计算。

#### 3.3.1 每专家预期 Token 数

| 计算 | 值 |
|------|------|
| 每 bt tile 总 token-expert 分配数 | 64 × 8 = 512 (本设备) |
| 跨 4 设备总分配数 | 4 × 512 = 2,048 |
| 每专家平均 token 数 (256 experts) | 2048 / 256 = **8** |
| 本地 64 专家预期总 token 数 | 64 × 8 = 512 |
| 零 token 专家概率 (Poisson lambda=8) | e^(-8) = **0.034%** (~0/64) |

> Decode 场景下每专家处理 ~8 个 token，`btc=64` 的 GEMM tile 以完整 64 行执行 — M 维度有效利用率 8/64 = **12.5%** (与 bt=32 时相同)。

#### 3.3.2 FFN1: Gate + Up 投影 (per expert)

每专家对每个 `(bf_id, bd1_id)` 组合执行:

```
token_tile[btc, bd1c_per_t_packing] × W1_tile[bd1c_per_t_packing, bfc] → gate[btc, bfc]
token_tile[btc, bd1c_per_t_packing] × W3_tile[bd1c_per_t_packing, bfc] → up[btc, bfc]
```

**权重加载 (per bd1 slice)**:

| 数据 | Shape | 大小 |
|------|-------|------|
| W1 tile | (t_packing, bd1_per_tp, bf) = (2, 1024, 1024) BF16 | **4 MB** |
| W3 tile | 同上 | **4 MB** |
| Token tile | (bts, t_packing, bd1_per_tp) = (64, 2, 1024) BF16 | **256 KB** |
| 小计 (per bd1 slice) | | **8.25 MB** |

**遍历维度**:

| 循环 | 次数 | 累计权重数据 |
|------|------|-------------|
| per (bf_id, bd1_id) | 1 | 8 MB |
| × num_bd1 = 4 | 4 | 32 MB |
| × num_bf = 2 | 2 | **64 MB** |

**MXU FLOPs (per expert, 含 padding)**:

```
Per (bf_id, bd1_id, p_id):
  W1 matmul: 2 × btc × bd1c_per_tp × bfc = 2 × 64 × 1024 × 1024 = 134.22 MFLOPS
  W3 matmul: 同上 = 134.22 MFLOPS

Per (bf_id, bd1_id):  × t_packing = 2 × 268.44M = 536.87 MFLOPS
Per bf_id:            × num_bd1 = 4 × 536.87M = 2,147.48 MFLOPS
Per expert:           × num_bf = 2 × 2,147.48M = 4,294.97 MFLOPS = 4.29 GFLOPS
```

#### 3.3.3 FFN2: Down 投影 (per expert)

激活函数后，对每个 `(bf_id, bd2_id)` 执行:

```
act[btc, bfc] × W2_tile[bfc, bd2c_per_t_packing] → output[btc, bd2c_per_t_packing]
```

**权重加载 (per bd2 slice)**:

| 数据 | Shape | 大小 |
|------|-------|------|
| W2 tile | (t_packing, bf, bd2_per_tp) = (2, 1024, 1024) BF16 | **4 MB** |
| FFN2 result staging | (bts, t_packing, bd2_per_tp) = (64, 2, 1024) BF16 | **256 KB** |

**遍历维度**: 同 FFN1: num_bd2=4 × num_bf=2 = 8 slices → **32 MB** 权重

**MXU FLOPs (per expert, 含 padding)**:

```
Per (bf_id, bd2_id, p_id):  2 × 64 × 1024 × 1024 = 134.22 MFLOPS
Per (bf_id, bd2_id):  × t_packing = 268.44 MFLOPS
Per bf_id:            × num_bd2 = 4 × 268.44M = 1,073.74 MFLOPS
Per expert:           × num_bf = 2 × 1,073.74M = 2,147.48 MFLOPS = 2.15 GFLOPS
```

#### 3.3.4 VPU 激活 (per expert)

SiLU(gate) * up 在 FFN2 内部计算:

| 操作 | FLOPs | Compute Unit |
|------|-------|-------------|
| sigmoid(alpha × gate) | ~3 × btc × bf = 196,608 | VPU |
| gate × sigmoid | btc × bf = 65,536 | VPU |
| (up+1) × glu 或 silu×up | btc × bf = 65,536 | VPU |
| **小计 (per bf block)** | **~328K** | VPU |
| **Per expert (× 2 bf)** | **~656K** | VPU |

> VPU 开销相对 MXU 可忽略 (656K vs 6.44G = 0.01%)。

#### 3.3.5 单专家汇总

| 指标 | 值 |
|------|------|
| 权重 HBM 读取 | W1: 32 MB + W3: 32 MB + W2: 32 MB = **96 MB** |
| Token HBM 读取 | ~4 MB (from a2a_s_hbm scratch) |
| Result HBM 写入 | ~4 MB (to a2a_s_acc_hbm scratch) |
| MXU FLOPs (executed) | FFN1: 4,295M + FFN2: 2,148M = **6,442 MFLOPS** |
| MXU FLOPs (useful, dyn_sz~=8) | 6,442 × 8/64 = **805 MFLOPS** |
| Arithmetic Intensity (executed) | 6,442M / 96M = **67.1 FLOPs/byte** |
| Arithmetic Intensity (useful) | 805M / 96M = **8.4 FLOPs/byte** |
| **瓶颈判定** | **HBM BW** (AI << ridge point 625) |

#### 3.3.6 全部 64 专家汇总 (per bt tile)

| 指标 | Per Expert | × 64 Experts |
|------|-----------|-------------|
| 权重 HBM 读取 | 96 MB | **6,144 MB (6 GB)** |
| Token 读取 | ~4 MB | ~256 MB |
| Result 写入 | ~4 MB | ~256 MB |
| MXU FLOPs (executed) | 6.44 GFLOPS | **412 GFLOPS** |
| MXU FLOPs (useful) | 0.81 GFLOPS | **51.5 GFLOPS** |
| 总 HBM 流量 | ~104 MB | **~6,656 MB** |

**时间估算 (per bt tile)**:

| 路径 | 计算 | 时间 |
|------|------|------|
| HBM 带宽限制 | 6,656 MB / 3,690 GB/s | **1.80 ms** |
| MXU 计算限制 (executed) | 412 GFLOPS / 2,307 TFLOPS | **0.179 ms** |
| MXU 计算限制 (useful) | 51.5 GFLOPS / 2,307 TFLOPS | **0.022 ms** |
| **HBM / Compute 比** | | **10.1x** (极度访存受限) |

---

### 3.4 Stage 4: 共享专家 FFN (交错执行)

共享专家与路由专家**交错执行** — 每处理一个路由专家前后各调用一次 `run_shared_expert_slice()`。
由于 `se_total_blocks=8` 而循环 128 次 (2 × 64)，SE 实际在前 4 个专家迭代中完成计算。

#### 3.4.1 每 SE Block 数据

| 数据 | 遍历 | 大小 |
|------|------|------|
| W1_shared slice | num_bd1=4 × (t_packing × bd1_per_tp × bse) BF16 | 4 × 1 MB = **4 MB** |
| W3_shared slice | 同上 | **4 MB** |
| W2_shared slice | num_bd2=4 × (t_packing × bse × bd2_per_tp) BF16 | 4 × 1 MB = **4 MB** |
| Token slices | num_bd1=4 × (bt × t_packing × bd1_per_tp) BF16 | 4 × 256 KB = 1 MB |
| **Per SE block 权重** | | **12 MB** |

#### 3.4.2 每 SE Block FLOPs

| 操作 | FLOPs |
|------|-------|
| FFN1 (gate+up) | 2 × num_bd1 × t_packing × (2 × bt × bd1_per_tp × bse) = 2×4×2×(2×64×1024×256) = 537M |
| Activation | ~82K (negligible) |
| FFN2 (down) | num_bd2 × t_packing × (2 × bt × bse × bd2_per_tp) = 4×2×(2×64×256×1024) = 268M |
| **Per SE block** | **805 MFLOPS** |

#### 3.4.3 全部 8 SE Blocks 汇总

| 指标 | Per Block | × 8 Blocks |
|------|-----------|-----------|
| 权重 HBM 读取 | 12 MB | **96 MB** |
| Token HBM 读取 | 1 MB | 8 MB |
| MXU FLOPs | 805 MFLOPS | **6,440 MFLOPS** |
| Arithmetic Intensity | 805M / 12M = 67.1 | 同 |

**时间估算**: 96 MB / 3,690 GB/s = **26 μs** (HBM-limited)

> **关键**: SE FFN 与路由专家 FFN **并行执行** — SE 在专家 0-3 的 FFN 间隙中完成，不增加关键路径。
> SE 权重加载 (96 MB) 与路由专家权重加载 (6,144 MB) 共享 HBM 带宽，增加约 1.5% 总带宽压力。

---

### 3.5 Stage 5: All-to-All Gather + 通信等待

**操作**: 收集每个专家的 FFN 输出，发送回原始设备。

Gather 在每个专家 FFN 完成后立即启动 (`start_a2a_gather`)，与后续专家的 FFN 计算流水线化。
最终 `wait_a2a_gather_recv_all()` 等待所有 gather 完成。

| 指标 | 值 |
|------|------|
| 每专家 gather 数据 | dyn_sz × token_size = 8 × 16 KB = 128 KB |
| 64 专家总 gather 量 | ~8 MB per device |
| ICI 传输 (75% remote) | ~6 MB |
| 本地 DMA | ~2 MB |
| 估计耗时 | ~30-50 μs (largely overlapped with expert FFN) |

> Gather 是 scatter 的反向操作。由于与专家 FFN 流水线化，大部分通信时间被 compute 覆盖。
> `sync_barrier()` 确保所有设备完成 gather 后才进入输出聚合阶段。

---

### 3.6 Stage 6: 输出聚合与写回

**操作**: `acc_and_store_output()`

1. 从 `a2a_g_hbm` 加载每个 token 的 top_k=8 个专家输出
2. 使用 `topk_weights` 加权求和
3. 加上共享专家输出 (从 `b_se_acc_vmem` 读取, F32)
4. 转换为 BF16 并写入 `output_hbm`

**数据大小**:

| 数据 | Shape | 大小 |
|------|-------|------|
| Expert outputs (read) | bt × top_k × hidden_size × 2B | 64 × 8 × 8192 × 2 = **8 MB** |
| topk_weights (already in VMEM) | bt × top_k × 4B | 2 KB |
| SE output (VMEM, F32) | bt × hidden_size × 4B | 2 MB |
| Final output (write) | bt × hidden_size × 2B | 1 MB |

**计算**: bt × top_k × hidden_size = 64 × 8 × 8192 = **4 MFLOPS** (VPU 加权求和)

| 指标 | 值 |
|------|------|
| HBM 读取 | ~8 MB (expert outputs from a2a_g_hbm) |
| HBM 写入 | 1 MB (final output) |
| VPU FLOPs | ~4 MFLOPS |
| **瓶颈** | HBM read (8 MB / 3,690 GB/s = 2 μs) |
| 估计耗时 | ~5-10 μs (DMA latency dominated) |

> 输出聚合使用 acc_bt 级别的双缓冲流水线 (`run_acc_pipeline`)，每次加载 acc_bt 个 token 的 top_k 输出进行加权聚合。

---

## 4. 全局性能预算

### 4.1 Per BT Tile 汇总

| 阶段 | HBM Read | HBM Write | MXU FLOPS | ICI Comm | Est. Time |
|------|----------|-----------|-----------|----------|-----------|
| 0. Barrier + Init | 0 | 0 | 0 | barrier | 5 μs |
| 1. Routing/Metadata | 4 KB | 0 | 0 | 4 KB allreduce | 15 μs |
| 2. A2A Scatter | 1 MB | 8 MB | 0 | 6 MB ICI | 40 μs |
| **3. Expert FFN (×64)** | **6,144 MB** | **256 MB** | **412 GFLOPS** | **0** | **1,800 μs** |
| 4. SE FFN (×8 blocks) | 96 MB | 0 | 6.4 GFLOPS | 0 | (overlapped) |
| 5. Gather + Sync | 0 | 8 MB | 0 | 6 MB ICI | 40 μs |
| 6. Output Acc | 8 MB | 1 MB | 4 MFLOPS | 0 | 10 μs |
| 7. Output Store | 0 | 1 MB | 0 | 0 | <1 μs |
| **Total (per bt tile)** | **~6,249 MB** | **~274 MB** | **~418 GFLOPS** | **~12 MB** | **~1,910 μs** |

### 4.2 全 Kernel 汇总 (1 bt tile)

| 指标 | 值 |
|------|------|
| 总 HBM 读取 | 1 × 6,249 = **~6,249 MB = 6.1 GB** |
| 总 HBM 写入 | 1 × 274 = **~274 MB** |
| 总 MXU FLOPs (executed) | 1 × 418 = **~418 GFLOPS** |
| 总 MXU FLOPs (useful) | 1 × 51.5 = **~51.5 GFLOPS** |
| 总 ICI 通信 | 1 × 12 = **~12 MB** |
| **HBM-limited 时间** | 6,523 MB / 3,690 GB/s = **1.77 ms** |
| **Compute-limited 时间** | 418 GFLOPS / 2,307 TFLOPS = **0.18 ms** |
| **理论下界** | **~1.8 ms** |
| **实测时间** | **149.02 ms** |

> **相比 bt=32**: HBM 读取从 12,490 MB 降至 ~6,249 MB (减半)，理论下界从 ~3.5 ms 降至 ~1.8 ms。`num_bt` 从 2 降至 1，权重只需读取一遍。

### 4.3 数据构成分析

```
HBM 读取构成 (~6,249 MB):
  ┌─────────────────────────────────────────────────────────┐
  │ 路由专家权重 (W1+W2+W3)  6,144 MB   ████████████████  98.3%
  │ 共享专家权重              96 MB      █                  1.5%
  │ Token/Routing/Output      9 MB       ▏                  0.1%
  └─────────────────────────────────────────────────────────┘
```

> **核心发现**: 98.3% 的 HBM 流量来自路由专家权重加载。每个 bt tile 加载全部 64 个本地专家的完整权重 (6 GB)，因为：
> 1. Decode 场景下每专家仅处理 ~8 个 token
> 2. `num_bt=1` 意味着权重只加载一次 (相比 bt=32 的两次已减半)
> 3. 256 expert × top_k=8 使得几乎所有本地专家都收到 token (P(0)=0.034%)

---

## 5. Roofline 分析

### 5.1 TPU v7x Hardware Ceiling

| 参数 | 值 |
|------|------|
| HBM 带宽 | 3,690 GB/s |
| MXU 峰值 (BF16) | 2,307 TFLOPS (dual MXU) |
| Ridge Point | 2,307T / 3,690G = **625 FLOPs/byte** |
| VMEM 容量 | 64 MiB |

### 5.2 Kernel 工作点

| Metric | Executed | Useful (dyn_sz~=8) |
|--------|----------|-------------------|
| Total FLOPs | 418 GFLOPS | 51.5 GFLOPS |
| Total Bytes | 6.2 GB | 6.2 GB |
| **Arithmetic Intensity** | **67.1 FLOPs/byte** | **8.4 FLOPs/byte** |
| AI / Ridge Point | 10.7% | 1.3% |
| **Regime** | **Memory-bound** | **Extremely memory-bound** |

### 5.3 Roofline 图 (概念)

```
TFLOPS
  │
  │  2307 ─────────────────────────────────────── Peak MXU ──
  │      ╱
  │    ╱
  │  ╱                                Ridge Point = 625
  │╱                                       │
  │                                        ▼
  │                    ┌─── HBM BW Ceiling ─┐
  │                  ╱ │                     │
  │                ╱   │                     │
  │              ╱     │                     │
  │            ╱       │                     │
  │          ╱         │                     │
  │        ╱           │  ★ Kernel (executed, AI=67.1)
  │      ╱             │  │                  │
  │    ╱               │  │                  │
  │  ╱   ☆ Kernel (useful, AI=8.4)          │
  │╱      │                                  │
  ├───────┼────────────┼───────────────────┤───→ FLOPs/Byte
  0      8.4          67.1                625
```

- ★ **Executed**: 67.1 FLOPs/byte → HBM ceiling 下的理论吞吐 = 3,690 × 67.1 = 248 TFLOPS (MXU 峰值的 10.7%)
- ☆ **Useful**: 8.4 FLOPs/byte → 有效计算吞吐 = 3,690 × 8.4 = 31 TFLOPS (MXU 峰值的 1.3%)

> **诊断**: 本 kernel 深度处于 HBM BW-bound 区域。提升性能的唯一途径是减少 HBM 访问量或增大 batch size。

---

## 6. 流水线与双缓冲分析

### 6.1 权重双缓冲 (bd1/bd2 维度)

```
bd1_id:    0           1           2           3
      ┌──────────┬──────────┬──────────┬──────────┐
DMA:  │Load W1[0]│Load W1[1]│Load W1[2]│Load W1[3]
      │Load W3[0]│Load W3[1]│Load W3[2]│Load W3[3]
      └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
MXU:       │  Compute  │  Compute  │  Compute  │
           │   bd1=0   │   bd1=1   │   bd1=2   │
           └───────────┴───────────┴───────────┘
Buffer:  sem_id=0    sem_id=1    sem_id=0    sem_id=1   (ping-pong)
```

**分析**: 每个 W1/W3 tile = 4 MB，DMA 时间 = 4 MB / 3,690 GB/s = 1,084 ns。
MXU 计算时间 (executed) = 537M FLOPs / 2,307 TFLOPS = 233 ns。
**DMA:Compute 比 = 4.7:1** — DMA 仍慢于 MXU, 双缓冲有效但无法消除等待。相比 bt=32 的 9.3:1 已显著改善。

### 6.2 Token Staging 双缓冲 (bts 维度)

```
tile_id:    0               1               2
       ┌──────────────┬──────────────┬──────────────┐
DMA:   │Stage tile[0] │Stage tile[1] │Stage tile[2] │
       └──┬───────────┴──┬───────────┴──┬───────────┘
MXU:      │  FFN1 tile[0]│  FFN1 tile[1]│
          └──────────────┴──────────────┘
```

每 token tile = 256 KB，DMA = 256 KB / 3,690 GB/s = 69 ns (远快于权重加载)。Token staging 不在关键路径上。

### 6.3 A2A 累加三缓冲

FFN2 使用 `a2a_s_acc_stage_x3_vmem` 实现 **三缓冲流水线**:

```
Triple buffer: buf_compute / buf_store / buf_load (rotating)

Tile:     0           1           2           3
     ┌─────────┬─────────┬─────────┬─────────┐
Load:│ buf[0]  │ buf[2]  │ buf[1]  │ buf[0]  │
     └──┬──────┴──┬──────┴──┬──────┴──┬──────┘
Comp:   │ buf[0]  │ buf[2]  │ buf[1]  │ buf[0]
        └──┬──────┴──┬──────┴──┬──────┴──┬────┘
Store:     │ (init)  │ buf[0]  │ buf[2]  │ buf[1]
           └─────────┴─────────┴─────────┴────────
```

对于 `should_init_ffn2=True` (bf_id=0), 无需预加载旧值, 直接覆盖写入。
对于后续 bf_id, 需先加载已有部分和再累加。

### 6.4 跨专家权重预取

内核在当前专家的最后一个 bd2 slice 处理期间, 预取下一个专家的 W1/W3:

```
Expert N:                    Expert N+1:
... │ FFN2 last bd2 │        │ FFN1 first bd1 │ ...
    │               │        │                │
    └─┬─ Prefetch W1[N+1] ─→ │ Wait W1[N+1]  │
      └─ Prefetch W3[N+1] ─→ │ Wait W3[N+1]  │
```

这减少了专家切换时的 cold-start 延迟, 但首个权重 tile (4 MB) 的加载时间 (1,084 ns) 相对专家总处理时间 (~28 μs) 仍很小。

### 6.5 共享专家交错

SE 计算插入在路由专家循环的间隙中:

```
Expert loop iteration:
┌──────────────────────────────────────────────────┐
│  SE_slice(2i)  │ wait_scatter │ expert_ffn(i)    │ start_gather │ SE_slice(2i+1) │
│  (~26 μs / 8)  │              │  (~28 μs)        │              │  (~26 μs / 8)  │
└──────────────────────────────────────────────────┘

i=0:  SE[0] (3.3μs)  │ wait │ expert[0] (28μs) │ gather │ SE[1] (3.3μs)
i=1:  SE[2] (3.3μs)  │ wait │ expert[1] (28μs) │ gather │ SE[3] (3.3μs)
i=2:  SE[4] (3.3μs)  │ wait │ expert[2] (28μs) │ gather │ SE[5] (3.3μs)
i=3:  SE[6] (3.3μs)  │ wait │ expert[3] (28μs) │ gather │ SE[7] (3.3μs)
i=4:  SE[no-op]       │ wait │ expert[4] (28μs) │ gather │ SE[no-op]
...
i=63: SE[no-op]       │ wait │ expert[63](28μs) │ gather │ SE[no-op]
```

> SE 在前 4 个专家迭代中完成全部 8 blocks 的计算。SE 权重加载 (96 MB) 与前 4 个专家的权重加载 (4 × 96 MB) 竞争 HBM 带宽, 可能增加这 4 个迭代的时间约 25%。

---

## 7. LLO 调度分析

### 7.1 编译结果概述

从 `benchmark_results/ir_dumps/llo/` 中分析编译 ID `1777793912529267903` 的内核:

| 指标 | 值 |
|------|------|
| Kernel 全名 | `fused-moe-k_8-bt_64_64_64-bf_1024_1024-bd1_2048_2048-bd2_2048_2048-shared_expert_bse_256.1` |
| 总静态 bundle 数 | 36,121 |
| 带注释指令数 | 135,642 |
| 带源码位置数 | 262 |
| LLO pass 数 | 71 passes (00-71) |

### 7.2 VLIW 功能单元利用率

从 `69-final_hlo-static-per-bundle-utilization.txt` 统计:

| 功能单元 | 含 MXU 指令的 bundle 数 | 占总 bundle 比例 |
|----------|----------------------|-----------------|
| MXU (>=1 active) | 3,005 | **8.3%** |
| MXU=2 (dual active) | ~2,800 | ~7.8% |
| VALU (>=1 active) | 多数 MXU bundle 同时有 VALU=3-4 | — |
| SALU only | ~30,000+ | >83% |

**解读**: 83%+ 的静态 bundle 不含 MXU 指令 — 主要是:
- 标量地址计算 (SALU) 和循环控制
- DMA 发起/等待指令
- Semaphore 同步操作
- SMEM/VMEM 指针操作

### 7.3 MXU 计算区域特征

在 MXU 活跃的 bundle 中 (lines 1310-1378):

```
MXU  XLU  VALU  VPOP  EUP  VLOAD  VLOAD:FILL  VSTORE  VSTORE:SPILL  SALU
 2    0    3     0     0    0      0           2       2             0    ← Gate/Up matmul
 2    0    4     0     0    0      0           1       1             0    ← 满载 VALU
 2    0    4     0     0    2      0           1       1             0    ← 权重预取重叠
 1    0    4     0     0    3      2           0       0             0    ← DMA 密集
```

**特征**:
- 双 MXU 同时激活 (MXU=2), 说明 matmul 正确利用了双 MXU 配置
- VALU 饱和 (3-4/4), 处理激活函数和数据格式转换
- VSTORE:SPILL 出现 (1-2), 表明 VMEM 压力导致少量 spill — 但比例低
- 无 HBM starvation 标记 (静态分析)

### 7.4 VMEM 分配概况

从 `02-original.txt` (初始 LLO) 中观察到的主要 VMEM 分配:

| 分配 | Shape | 大小 | 推断用途 |
|------|-------|------|---------|
| allocation8 | bf16[2,8,16,2,4096] | 4 MB | FFN accumulator (reshaped b_acc_vmem) |
| allocation11 | bf16[2,64,8192] | 2 MB | 输出双缓冲 (b_output_x2_vmem) |
| allocation12-14 | 3 × bf16[2,2,1024,1024] | 3 × 8 MB | W1/W3/W2 权重双缓冲 |
| allocation19-21 | 3 × bf16[2,2,1024,1024] | 3 × 8 MB | SE W1/W3/W2 权重双缓冲 |
| allocation15 | f32[2,256,1,1024] | 2 MB | FFN 内部累加器 |
| allocation22 | f32[2,64,8192] | 4 MB | SE 输出累加器 (F32) |
| **VMEM 总占用** | | **~56 MB** | 占 64 MiB 的 **87%** |

> VMEM 利用率 87%, 相比 bt=32 时的 34% 显著增加。更大的 tile 尺寸占用了更多 VMEM 空间。

### 7.5 Critical Path 分析

从 `23-critical-path.txt` 观察:

1. **Entry block** (region0): 972 members, critical path length=12
   - 主要是地址计算 (`sld`, `sshift`, `sadd`, `ssub`)
   - 对应 `_fused_moe_impl.py:587` (axis index 计算)

2. **Routing block** (region13): critical path length=201
   - `vrot.slane`, `vbcast.lane`, `vpop`, `vselect`, `vadd` 操作链
   - 对应路由矩阵生成 (lines 2661-2666)

3. **Expert scatter** (region4609-4614): critical path length=451
   - 8 轮重复的 expert ID 加载 → modular index → stride descriptor → `dma.general`
   - 对应 `_fused_moe_impl.py:970-1010` (top_k=8 展开)

4. **MXU compute** (region4620-4625): `vmatprep` + `vmatpush1` 指令
   - BF16 矩阵乘法的 MXU 流水线
   - 对应 SE 内层循环 (line 2570)

### 7.6 Strix Bundle-Source Mapping

使用 `strix analyze-bundles` 对 36,121 个 bundle 进行源码位置映射，识别出以下关键热点:

| Source | Slots | Ops | Role |
|--------|-------|-----|------|
| L1680-1684 | 31,871 | vrot.slane, vcombine, vcmask | Token tile 加载 |
| L1691-1695 | 14,336 | vmatprep, vmatpush1 (MXU0+MXU1) | W1 gate matmul |
| L1709-1713 | 14,336 | vmatprep, vmatpush1 (MXU0+MXU1) | W3 up matmul |
| L2565-2567 | 13,707 | vld, vrot.slane, vcmask | W2 token 加载 |
| L1885 | 7,686 | vmatprep, vmatpush1 | W2 down matmul |
| L1693/L1711 | 7,168 ea. | vld | 权重 tile 加载 |
| L1907 | 5,888 | vadd.f32, vpack.c.bf16 | SiLU 激活 |

**Strix 静态 profiler 结论**: AI = 59.5 FLOPs/Byte, Balanced/Mixed bottleneck。

> Token tile 加载 (L1680-1684) 占用了最多的 bundle slot (31,871)，说明 DMA 调度和数据搬运在 LLO 指令层面占据主导地位。MXU matmul 相关的 bundle (W1/W3/W2 合计 ~36K slots) 虽然总量可观，但分散在多个源码位置，反映了权重 tile 化加载的结构。

### 7.7 实测性能 vs 理论下界

| 指标 | 值 |
|------|------|
| 理论下界 (HBM-limited) | ~1.8 ms |
| **实测时间** | **149.02 ms** |
| **差距倍数** | **83x** |

**有效 HBM 带宽**: 6,249 MB / 149 ms = **42 GB/s** (峰值 3,690 GB/s 的 **1.1%**)

**差距根因分析**:

1. **Per-expert DMA setup 开销**: 每个专家需要独立发起多轮 DMA 请求 (W1/W3/W2 各 num_bd1/num_bd2 个 tile)，每次 DMA 发起有固定延迟 (descriptor 构建、semaphore 操作)。64 个专家 × 每专家 ~24 个 DMA 请求 = ~1,536 次 DMA 发起，固定开销累积显著。

2. **同步 barrier 开销**: `sync_barrier()`, `wait_a2a_scatter_recv()`, `wait_a2a_gather_recv_all()` 等同步原语在 4 设备拓扑上引入不可忽略的等待时间。每个专家循环迭代都包含 scatter recv 等待。

3. **顺序循环控制**: 64 个专家通过 `lax.fori_loop` 顺序执行，循环控制逻辑 (SALU 指令链) 在每次迭代产生开销。LLO 中 >83% 的 bundle 为 SALU-only 佐证了这一点。

4. **大量小 DMA 传输**: 尽管 tile 尺寸从 1MB 增大到 4MB，但仍存在大量 sub-optimal DMA 传输 (token tile 256KB、routing metadata 数 KB 等)，DMA 引擎无法充分发挥流水线效率。

---

## 8. 瓶颈诊断与优化方向

### 8.1 核心瓶颈

```
┌─────────────────────────────────────────────┐
│  #1 瓶颈: HBM 带宽 (权重加载)               │
│  ─────────────────────────────────           │
│  AI = 8.4 FLOPs/byte (useful)               │
│  Ridge Point = 625 FLOPs/byte               │
│  → MXU 有效利用率 ~1.3%                      │
│  → 98.3% HBM 流量来自权重                    │
│  → 每 bt tile 加载 6 GB 权重处理 ~512 tokens │
│                                             │
│  #2 瓶颈: 实测 149ms vs 理论 1.8ms (83×)    │
│  → 有效 HBM BW 仅 42 GB/s (1.1% peak)      │
│  → DMA setup + 同步开销是主要差距来源         │
└─────────────────────────────────────────────┘
```

### 8.2 MoE Decode 固有挑战

这是 MoE decode 的**结构性瓶颈**, 非内核实现问题:

| 因素 | 影响 |
|------|------|
| 每专家 ~8 tokens (稀疏路由) | 每 96 MB 权重仅做 805 MFLOPS 有效计算 |
| btc=64 > dyn_sz~=8 | MXU M-dim 填充率 12.5% (浪费 87.5% 计算) |
| 256 专家 / 4 设备 = 64 local | 64 次完整权重加载, 无法跨 bt tile 复用 |
| top_k=8 + 256 experts | 几乎所有本地专家都收到 token (P(0)=0.034%) |

### 8.3 潜在优化方向

| 方向 | 预期收益 | 可行性 |
|------|---------|--------|
| **INT8/FP8 量化** | 权重减半→HBM流量减半→~0.9ms | 高 (已有 quant_block_k 支持) |
| **增大 batch size** | batch 增 → AI 增 → 更接近 ridge point | 取决于推理场景 |
| **Expert caching (跨 bt tile)** | 避免重复加载热门专家权重 | 需要额外 VMEM/HBM 缓存策略 |
| **Expert 合并/压缩** | 减少 expert 数或合并冷门专家 | 模型级别修改 |
| **减小 ep_size** (增加 local batch) | 更多 token/expert → 更高 AI | 权重内存需求增加 |
| **Prefill-Decode 分离** | Prefill 大 batch 高 MXU 利用率 | 系统级优化 |

### 8.4 量化收益预估

若使用 INT8 量化 (quant_block_k=128):

| 指标 | BF16 | INT8 | 变化 |
|------|------|------|------|
| 权重大小 (per expert) | 96 MB | 48 MB + 1.5 MB scale | -50% |
| 总权重读取 (1 bt tile) | 6.1 GB | 3.1 GB | -49% |
| HBM-limited 时间 | 1.77 ms | ~0.9 ms | -48% |
| Scale 计算开销 | 0 | +~5% MXU | 可忽略 |

---

## 9. 阶段依赖关系汇总图

```
Time ──────────────────────────────────────────────────────────────→

BT tile 0 (唯一):
  ┌─Barrier─┐ ┌─Routing─┐ ┌─Scatter─┐ ┌──── Expert FFN ×64 ────────────────────┐ ┌Wait┐ ┌─Acc─┐ ┌Out┐
  │  5 μs   │ │  15 μs  │ │  40 μs  │ │           1,800 μs                     │ │40μs│ │10μs │ │1μs│
  └─────────┘ └─────────┘ └─────────┘ └─────────────────────────────────────────┘ └────┘ └─────┘ └───┘
                                        ├─SE[0]─SE[1]─..─SE[7]─(no-op×56)─────┤  (SE overlapped)
                                        ├──── Gather (pipelined) ──────────────┤

Total: ~1,910 μs ≈ 1.9 ms (理论下界 ~1.8 ms)

实测: 149.02 ms — 差距主要来自 per-expert DMA setup 开销与同步 barrier
```

---

## 10. TPU Performance Model 仿真结果

使用 `tpu-perf-model` CLI 对 kernel 进行 ComputeStep 建模与仿真。

### 10.1 Per-Expert FFN 仿真 (核心循环)

对单个专家的 FFN 管线建模: gate matmul → up matmul → SiLU activation → gated mul → down matmul。

**输入 ComputeStep 定义**: `steps_fused_moe_per_expert.json`

| Step | Op Type | FLOPs | HBM I/O | T(HBM) | T(Compute) | T(Step) | Bottleneck | AI |
|------|---------|-------|---------|--------|------------|---------|------------|-----|
| ffn1_gate_matmul | matmul/MXU | 2.15G | 32.06 MB | 8,689 ns | 930 ns | 9,619 ns | **HBM_BW** | 67.1 |
| ffn1_up_matmul | matmul/MXU | 2.15G | 32.06 MB | 8,689 ns | 930 ns | 9,619 ns | **HBM_BW** | 67.1 |
| silu_activation | elem/VPU | 197K | **0** (fused) | 0 | 8.5 ns | 8.5 ns | COMPUTE | inf |
| gated_mul | elem/VPU | 66K | 256 KB (fused) | 69 ns | 2.8 ns | 69 ns | HBM_BW | 0.3 |
| ffn2_down_matmul | matmul/MXU | 2.15G | 32.06 MB | 8,689 ns | 930 ns | 9,619 ns | **HBM_BW** | 67.1 |
| **单专家合计** | | **6.44G** | **96 MB** | **26,067 ns** | **2,793 ns** | **28,860 ns** | **HBM_BW** | **67.1** |

**关键发现**:

1. **DMA:Compute 比 = 9.34:1** — 每个 matmul tile 的 DMA 时间 (8.69 μs) 是计算时间 (0.93 μs) 的 **9.3 倍** (相比 bt=32 的 19.92:1 改善一倍)
2. **Fusion 有效**: SiLU activation 完全在 VMEM/REG 中执行, 节省 1 MB HBM 访问
3. **MXU 效率 9.66%** — 峰值 2,307 TFLOPS 中利用 ~223 TFLOPS (per-expert), 相比 bt=32 的 4.77% 提升一倍
4. **单 tile 即为全专家**: 由于 M=64 < MXU tile 128, 整个专家权重在一个 tile 内完成, 无法 double-buffer

### 10.2 Pipeline Balance 分析

```
每步 DMA vs Compute 时间:
                                    DMA         Compute     Ratio
ffn1_gate_matmul [64,8192]×[8192,2048]:  8,689 ns    930 ns     9.34×
ffn1_up_matmul   [64,8192]×[8192,2048]:  8,689 ns    930 ns     9.34×
silu_activation  [64,2048]:                 69 ns      9 ns     7.67×  (fused)
gated_mul        [64,2048]:                 69 ns      1 ns    69.00×  (fused)
ffn2_down_matmul [64,2048]×[2048,8192]:  8,689 ns    930 ns     9.34×

Pipeline Balance:  ██████████████████ DMA ██████████████████  ██ Compute ██
                   ← 90% time in DMA wait →                   ← 10% compute →
```

### 10.3 全 BT Tile 推算 (64 专家 + 共享专家)

| 组件 | Per Unit | Count | Total Time |
|------|----------|-------|------------|
| 路由专家 FFN | 28.86 μs | ×64 | **1,850 μs** |
| 共享专家 FFN | 28.86 μs | ×1 | 29 μs (overlapped) |
| Routing + Scatter | — | — | ~55 μs |
| Gather + Output Acc | — | — | ~50 μs |
| **1 BT tile 合计** | | | **~1,955 μs = ~1.96 ms** |

> **相比 bt=32**: 2 个 bt tile 共 ~3.95 ms → 1 个 bt tile ~1.96 ms, 减少约 50%。

> **注意**: 28.86 μs/expert 是无 double-buffer 的串行时间。实际内核对 bd1/bd2 维度实施 double-buffering (ping-pong 权重加载),
> 使得有效时间更接近 max(DMA, DMA_prev_compute) = 27.8 μs/expert。64 × 27.8 μs = 1,780 μs,
> 与 Section 4.2 的理论下界 1.77 ms 吻合。

### 10.4 Micro-Op Critical Path 分析

从 micro-op 级别仿真获得的 critical path:

```
ffn1_up_load_q → vmem_to_reg_q → mxu → writeback → reg_to_vmem
    → silu_vmem_to_reg → silu_vpu → silu_reg_to_vmem
    → gated_mul_vmem_to_reg → gated_mul_vpu → gated_mul_reg_to_vmem → gated_mul_store
    → ffn2_load_q → vmem_to_reg_q → mxu → writeback → reg_to_vmem → store
```

**Stall 分布**:

| Stall Type | Count | 占比 | 原因 |
|------------|-------|------|------|
| WAIT_DATA | 20 | 41% | 等待 DMA 完成 (权重从 HBM 搬入 VMEM) |
| WAIT_VMEM | 11 | 22% | VMEM slot 被其他 DMA 占用 |
| WAIT_REG | 11 | 22% | VPR 被其他操作占用 |
| WAIT_UNIT | 7 | 14% | MXU/VPU 被前序操作占用 |

> **核心瓶颈确认**: 41% 的 stall 来自 WAIT_DATA — 等待 HBM→VMEM 的权重 DMA 完成。
> 这与 Section 8.1 的 HBM BW-bound 诊断完全一致。

### 10.5 VPR Register 压力

| 指标 | 值 |
|------|------|
| 峰值 VPR 使用 | 25,167 VPR (at t=24 ns, ffn2 writeback) |
| 权重在 REG 中驻留 | W1: VPR[128..8319] (8,192 VPR), W2: VPR[16848..25039] (8,192 VPR) |
| Spill 次数 | **0** |
| Activation 融合 VPR 开销 | VPR[16704..16799] (~96 VPR) |

> VPR 使用量极高 (W1/W3/W2 各占 8,192 VPR), 但由于 matmul 的 K 维度一次性加载整个权重 tile,
> 这些 VPR 在 MXU 计算后立即释放。无 spill 说明 VMEM→REG 管线无压力。

### 10.6 聚合模型对比 (Per-Expert vs Aggregated)

| 模型 | Total FLOPs | Total HBM | AI | Bottleneck | T(total) |
|------|-------------|-----------|-----|-----------|----------|
| **Per-Expert ×64** (正确) | 412G | 6.1 GB | **67.1** | **HBM_BW** | **1.85 ms** |
| Aggregated [4096,131072] (错误) | 26.4T | 8.2 GB | 3236 | COMPUTE | 11.8 ms |

> **警告**: 将 64 个独立专家聚合为一个 [4096, 131072] 大 matmul 会产生误导性结论:
> - 聚合 AI=3236 (超 ridge point) → 看似 compute-bound
> - 但实际 AI=67.1 (per-expert, M=64) → 仍然 memory-bound
>
> 原因: 聚合假设所有专家共享权重矩阵, 但实际每个专家有独立权重 (需独立加载)。
> 正确建模必须以 per-expert 为单位。

---

## 11. Gate + Up 两种计算方式的 HBM 访问量对比

当前 kernel 对 Gate (W1) 和 Up (W3) 投影的计算有两种可选策略，它们在 HBM 访问量上有本质差异。

### 11.1 符号定义

| 符号 | 含义 | Ling 2.6 值 |
|------|------|-------------|
| $T_{\text{local}}$ | 每芯片 token-expert pair 总数 $= \sum_e n_e$ | 64 (简化) |
| $E_{\text{local}}$ | 每芯片 expert 数 = E / EP | 64 |
| $k$ | Top-K routing | 8 |
| $H$ | Hidden dimension | 8192 |
| $I$ | Intermediate dimension | 2048 |
| $bt$ | Batch-tile 大小 | 64 |
| $B$ | 每元素字节数 (BF16) | 2 |
| $n_{bt}$ | $T_{\text{local}} / bt$, bt-tile 轮数 | 1 (= 64/64) |

### 11.2 方案 A: BT-tile 扫描全部 expert（Weight-streaming）

每个 bt-tile 将 token 加载进 VMEM，然后 `fori_loop` 遍历全部 $E_{\text{local}}$ 个 expert，逐个 load Gate / Up 权重执行 matmul。**Token 在 bt-tile 内保持 VMEM 常驻，跨 expert 复用**。

**Gate (W1) HBM 读取**:

$$\text{HBM}_{W1}^{(A)} = n_{bt} \times E_{\text{local}} \times H \times I \times B$$

每个 bt-tile 将所有 expert 的 Gate 权重从 HBM load 一遍 — 共 $n_{bt}$ 遍。

**Up (W3) HBM 读取**: 结构与 Gate 相同

$$\text{HBM}_{W3}^{(A)} = n_{bt} \times E_{\text{local}} \times H \times I \times B$$

**Token HBM 读取**: 每 bt-tile load 一次，跨 expert 复用

$$\text{HBM}_{\text{tokens}}^{(A)} = T_{\text{local}} \times H \times B$$

**合计**:

$$\boxed{\text{HBM}_{\text{Gate+Up}}^{(A)} = \frac{T_{\text{local}}}{bt} \times E_{\text{local}} \times 2 H I B \;+\; T_{\text{local}} \cdot H \cdot B}$$

代入 Ling 2.6 数值 ($n_{bt} = 1$):

| 项 | 公式 | 字节 |
|----|------|------|
| 权重 | $1 \times 64 \times 2 \times 8192 \times 2048 \times 2$ | **4 GiB** |
| Token | $64 \times 8192 \times 2$ | **1 MiB** |
| **合计** | | **= 4 GiB** |

> **关键改善**: 相比 bt=32 时 $n_{bt}=2$ 导致权重 8 GiB，bt=64 使 $n_{bt}=1$，方案 A 权重读取降至 4 GiB — **与方案 B 完全相同**！

### 11.3 方案 B: Token 预映射 + Token-stationary

先完成 routing 分组，将 token 按 expert 归属 scatter。对每个有 token 的 expert，将其 $\text{count}_e$ 个 token 固定在 VMEM 中，**只加载该 expert 的权重一次**。

**Gate (W1) HBM 读取**: 每个 active expert 权重只 load **一次**

$$\text{HBM}_{W1}^{(B)} = E_{\text{active}} \times H \times I \times B$$

**Up (W3) HBM 读取**:

$$\text{HBM}_{W3}^{(B)} = E_{\text{active}} \times H \times I \times B$$

**Token HBM 读取**: 每个 token 被路由到 $k$ 个 expert，每次需 load 进对应 expert 的 VMEM batch — 总共 load $k$ 次

$$\text{HBM}_{\text{tokens}}^{(B)} = \sum_{e} \text{count}_e \times H \times B = T_{\text{local}} \times k \times H \times B$$

**合计**:

$$\boxed{\text{HBM}_{\text{Gate+Up}}^{(B)} = E_{\text{active}} \times 2 H I B \;+\; T_{\text{local}} \cdot k \cdot H \cdot B}$$

代入 Ling 2.6 数值 ($E_{\text{active}} \approx E_{\text{local}} = 64$):

| 项 | 公式 | 字节 |
|----|------|------|
| 权重 | $64 \times 2 \times 8192 \times 2048 \times 2$ | **4 GiB** |
| Token | $64 \times 8 \times 8192 \times 2$ | **8 MiB** |
| **合计** | | **= 4 GiB** |

### 11.4 核心差异分析

两种方案的 FLOPs **完全相同**（相同的矩阵乘法），差别纯粹在 HBM 数据搬运量。

**权重维度**: 方案 A 的权重被重复加载 $n_{bt}$ 次，方案 B 只加载 1 次

$$\frac{\text{HBM}_{\text{weights}}^{(A)}}{\text{HBM}_{\text{weights}}^{(B)}} = n_{bt} = \frac{T_{\text{local}}}{bt}$$

**Token 维度**: 方案 B 的 token 被重复加载 $k$ 次（每个路由 expert 各一次），方案 A 只加载 1 次

$$\frac{\text{HBM}_{\text{tokens}}^{(B)}}{\text{HBM}_{\text{tokens}}^{(A)}} = k$$

但由于权重体量远大于 token（权重:token = 4000:1），**token 侧的差异可以忽略**，总 HBM 由权重主导。

> **bt=64 的关键结论**: 当 $n_{bt} = 1$ 时，方案 A 与方案 B 的 HBM 访问量**完全相同** (均为 4 GiB)。此时方案 A 的简单控制流更优，无需切换到方案 B。

### 11.5 不同 Token 规模下的对比

| $T_{\text{local}}$ | $n_{bt}$ (bt=64) | 方案 A (Gate+Up) | 方案 B (Gate+Up) | 权重加载倍数 |
|-----|------|--------|--------|------|
| 64 | 1 | 4 GiB | 4 GiB | **1x** (当前配置) |
| 128 | 2 | 8 GiB | 4 GiB | **2x** |
| 256 | 4 | 16 GiB | 4 GiB | **4x** |
| 512 | 8 | 32 GiB | 4 GiB | **8x** |

> **结论**: 当前 kernel 采用方案 A（bt-tile sweep all experts）。bt=64 配置下 $n_{bt} = 1$，方案 A 与方案 B 的 HBM 权重访问量完全相同 — 这是 bt 增大到 64 的核心收益之一。
>
> 当 $T_{\text{local}}$ 因负载不均或 batch 增大而超过 64 时，$n_{bt} > 1$ 将再次引入冗余权重读取，此时方案 B 的优势将重新显现。
>
> 但方案 B 需要额外开销:
> 1. **Token scatter/permute** — 按 expert 归属重排 token（需 all-to-all 或 local scatter）
> 2. **Variable-length expert batch** — 每个 expert 的 token 数不等，需要 padding 或动态 shape 处理
> 3. **跨 expert token 重复** — 每个 token load $k$ 次（但在此配置下 8 MiB vs 4 GiB 权重，影响 < 0.2%）

### 11.6 负载均衡敏感性

方案 A 对路由负载均衡**显著更敏感**，方案 B 则天然不受影响。

**注意**: 方案 A 的代码中**会跳过零 token 的 expert**（不会为没有路由 token 的 expert 加载权重）。负载敏感性的根源不在于此，而在于**热点 expert 导致单设备 token 量膨胀，进而触发所有 expert 权重的反复加载**。

**方案 A 的负载敏感性来源 — $n_{bt}$ 膨胀**:

在 Expert Parallelism 中，all-to-all scatter 将每个 token 发送到其路由 expert 所在的设备。当路由出现热点时：

1. 热点 expert 所在的设备接收**远多于均值**的 token
2. 该设备的有效 $T_{\text{local}}$ 膨胀
3. $n_{bt} = T_{\text{local}} / bt$ 随之增大
4. 每个 bt-tile 仍需遍历 active expert → **每个 active expert 的权重被加载 $n_{bt}$ 次**

$$\text{HBM}_{\text{weights}}^{(A)} = n_{bt} \times E_{\text{active}} \times 2HIB = \frac{T_{\text{local}}}{bt} \times E_{\text{active}} \times 2HIB$$

$T_{\text{local}}$ 不再是常量，而是取决于路由分布 — 热点设备的 $T_{\text{local}}$ 线性放大 $n_{bt}$，从而线性放大**所有** active expert 的权重加载次数。

**数值示例** (Ling 2.6, EP=4, bt=64):

| 路由质量 | 热设备 $T_{\text{local}}$ | $n_{bt}$ | 权重读取 (Gate+Up) | 相对均匀倍数 |
|---------|-------------------------|----------|-------------------|-------------|
| 均匀 | 64 | 1 | 4 GiB | 1x |
| 中度热点 | 128 | 2 | 8 GiB | **2x** |
| 极端热点 | 256 | 4 | 16 GiB | **4x** |

整个 kernel 的延迟由最慢的设备决定（需 `sync_barrier` 同步），因此热设备的 $n_{bt}$ 膨胀直接决定全局延迟：

$$T_{\text{kernel}} = \max_d \left( \frac{T_{\text{local}}^{(d)}}{bt} \right) \times E_{\text{active}} \times \frac{2HIB}{\text{HBM-BW}}$$

**方案 B 不受负载均衡影响**:

方案 B 中每个 expert 的权重**恰好加载一次**，无论该设备收到多少 token：

$$\text{HBM}_{\text{weights}}^{(B)} = E_{\text{active}} \times 2HIB$$

token 量增加只影响 matmul 的 M 维度（更多计算），但在此配置下 kernel 极度 HBM-bound（AI << ridge point），增加的计算时间可被权重 DMA 完全覆盖。token 自身的 HBM 开销 ($T_{\text{local}} \times k \times H \times B$) 相对权重可忽略（8 MiB vs 4 GiB）。

| 特性 | 方案 A | 方案 B |
|------|--------|--------|
| 权重加载次数/expert | $n_{bt} = T_{\text{local}} / bt$ (随 token 量线性增长) | **1** (常量) |
| 热点设备代价 | $n_{bt}$ 膨胀 → 所有 expert 权重反复加载 | 不变 — 更多 token 只增加可被覆盖的计算 |
| 全局延迟决定因素 | $\max_d(n_{bt}^{(d)})$ — 受最热设备支配 | $E_{\text{active}}$ — 与路由分布无关 |
| 负载不均匀时趋势 | **线性恶化** | **不受影响** |

> **总结**: 方案 A 的 HBM 代价与 $n_{bt}$ 成正比，而 $n_{bt}$ 由设备级 token 量决定。热点 expert 使 token 集中到少数设备 → $n_{bt}$ 膨胀 → 该设备上**所有** active expert 的权重被反复加载，延迟线性增长。全局 barrier 同步使最慢设备成为瓶颈。
>
> 方案 B 每个 expert 权重恰好加载一次，$\text{HBM}_{\text{weights}}$ 与 token 量完全解耦。热点带来的额外 token 只增加计算量（可被 DMA 覆盖），不触发权重重复加载。方案 B 对负载均衡天然免疫。
>
> **bt=64 的改善**: 在均匀负载下，bt=64 使 $n_{bt}=1$，方案 A 与方案 B 等价。只有在负载不均导致 $T_{\text{local}} > 64$ 时，方案 B 的优势才重新体现。

### 11.7 方案 B 单 Expert VMEM 容量分析

当 $E_{\text{local}} = 1$（每设备仅承载一个 expert）时，方案 B 的 token-stationary 策略将 token 全量驻留 VMEM，权重按 tile 流式加载。VMEM 容量决定了单次可处理的最大 token 数。

#### 11.7.1 VMEM 预算模型

VMEM 中同时驻留的数据分为三类：

$$\text{VMEM}_{\text{total}} = \underbrace{T \times H \times B}_{\text{token (stationary)}} + \underbrace{W_{\text{buf}}}_{\text{weight buf}} + \underbrace{T \times A_{\text{pt}}}_{\text{accumulators}}$$

其中 $B = 2$ (BF16 字节数)。

#### 11.7.2 FFN1 峰值（Gate + Up, binding constraint）

FFN1 阶段对每个 $(bf, bd1)$ tile 执行 Gate 和 Up 两个 matmul，VMEM 中需同时容纳：

| 组件 | Shape | 大小 | 说明 |
|------|-------|------|------|
| Token 输入 | $[T, H]$ BF16 | $T \times H \times 2$ | 常驻，跨所有 $(bf, bd1)$ 复用 |
| W1 双缓冲 | $[bd1, bf]$ BF16 ×2 | $2 \times 2048 \times 1024 \times 2 = 8$ MiB | Gate 权重 ping-pong |
| W3 双缓冲 | $[bd1, bf]$ BF16 ×2 | $8$ MiB | Up 权重 ping-pong |
| Gate 累加器 | $[T, bf]$ F32 | $T \times 1024 \times 4$ | 跨 bd1 累加 |
| Up 累加器 | $[T, bf]$ F32 | $T \times 1024 \times 4$ | 跨 bd1 累加 |

代入 Ling 2.6 参数 ($H = 8192$, $bf = 1024$, $bd1 = 2048$)：

$$\text{VMEM}_{\text{FFN1}} = T \times \underbrace{(8192 \times 2 + 1024 \times 4 + 1024 \times 4)}_{24 \text{ KiB/token}} + \underbrace{16 \text{ MiB}}_{\text{weight buffers}}$$

约束 $\text{VMEM}_{\text{FFN1}} \leq 64$ MiB：

$$T \times 24 \text{ KiB} + 16 \text{ MiB} \leq 64 \text{ MiB}$$

$$\boxed{T_{\max}^{(\text{FFN1})} = \frac{48 \text{ MiB}}{24 \text{ KiB}} = 2{,}048}$$

#### 11.7.3 FFN2 峰值验证

FFN2 阶段 token 数据仍驻留 VMEM（下一轮 bf 的 FFN1 需要复用），同时需要容纳激活输入和输出累加器：

| 组件 | Shape | 大小 |
|------|-------|------|
| Token 输入 | $[T, H]$ BF16 | $T \times 16$ KiB (常驻) |
| Activation 输入 | $[T, bf]$ BF16 | $T \times 2$ KiB |
| W2 双缓冲 | $[bf, bd2]$ BF16 ×2 | $8$ MiB |
| Output 累加器 | $[T, bd2]$ F32 | $T \times 8$ KiB |

$$\text{VMEM}_{\text{FFN2}} = T \times 26 \text{ KiB} + 8 \text{ MiB} \leq 64 \text{ MiB}$$

$$T_{\max}^{(\text{FFN2})} = \frac{56 \text{ MiB}}{26 \text{ KiB}} = 2{,}203$$

FFN1 更紧 (2,048 < 2,203)。FFN1 的 3,072 → 2,048 减少是因为更大的权重 buffer (16 MiB vs 4 MiB) 和更大的 per-token 累加器 (24 KiB vs 20 KiB)。

#### 11.7.4 结论

$$\boxed{T_{\max} = \left\lfloor \frac{\text{VMEM} - W_{\text{buf}}}{H \times B + A_{\text{pt}}} \right\rfloor = \left\lfloor \frac{64 \text{ MiB} - 16 \text{ MiB}}{16 \text{ KiB} + 8 \text{ KiB}} \right\rfloor = 2{,}048 \text{ tokens}}$$

VMEM 分配验证 ($T = 2{,}048$)：

| 组件 | 大小 | 占比 |
|------|------|------|
| Token $[2048, 8192]$ BF16 | 32 MiB | 50.0% |
| Gate + Up 累加器 F32 | 16 MiB | 25.0% |
| W1 + W3 双缓冲 | 16 MiB | 25.0% |
| **合计** | **64 MiB** | **100%** |

> 此时 VMEM 恰好 100% 占满。$T = 2{,}048$ 是 128 对齐的（$2048 = 16 \times 128$），满足 MXU block 对齐要求。
>
> 对比方案 A 的 Ling 2.6 配置 ($E_{\text{local}} = 64$, $bt = 64$)：方案 A 每 bt-tile 处理 64 个 token，因为需要遍历 64 个 expert。方案 B 在 $E_{\text{local}} = 1$ 时可一次性处理 2,048 个 token — **单轮 token 吞吐量提升 32x**，同时每 expert 权重仅加载一次（$n_{bt} = 1$），彻底消除权重重复加载。
>
> 相比 bt=32 配置的 $T_{\max} = 3{,}072$, bt=64 配置下 $T_{\max}$ 降至 2,048，因为更大的 tile 尺寸使权重 buffer 从 4 MiB 增至 16 MiB。

### 11.8 Decode 场景下 $T_{\text{local}}$ 上界分析

直觉上，decode 场景 batch size 较小，令 $bt = \text{batch-size}$ 即可使 $n_{bt} = 1$，消除权重重复加载。但这一推断混淆了 **unique token 数**与 **token-expert pair 总数**。

#### 11.8.1 $T_{\text{local}}$ 的精确定义

$T_{\text{local}}$ 是单设备上 token-expert pair 的总数：

$$T_{\text{local}} = \sum_{e \in E_{\text{local}}} n_e$$

其中 $n_e$ 是路由到 expert $e$ 的 token 数。同一个 token 如果被路由到本卡的 $m$ 个 expert，在 $T_{\text{local}}$ 中被计入 $m$ 次。

注意 $T_{\text{local}} \neq T_{\text{unique}}$（本卡上的去重 token 数）。

#### 11.8.2 上界推导

当 $E_{\text{local}} \geq k$ 时，每个 token 的全部 $k$ 个 top-$k$ 选择都可能落在本卡的 expert 上：

$$\boxed{T_{\text{local}}^{\max} = \text{batch-size} \times k}$$

**数值示例** ($E_{\text{local}} = 16$, batch size $= 256$, $k = 8$)：

由于 $16 \geq 8$，每个 token 的 8 个 expert 选择均可落在本卡的 16 个 expert 中：

$$T_{\text{local}}^{\max} = 256 \times 8 = 2{,}048$$

#### 11.8.3 对方案 A weight reload 的影响

即使将 $bt$ 设为 batch size：

$$n_{bt} = \frac{T_{\text{local}}}{bt} = \frac{2{,}048}{256} = 8$$

每个 active expert 的权重仍被加载最多 $n_{bt} = 8$ 次。

| 场景 | $T_{\text{local}}$ | $bt$ | $n_{bt}$ | 权重加载倍数 |
|------|---------------------|------|----------|-------------|
| 均匀 ($k / \text{EP}$) | 512 | 256 | 2 | 2x |
| 中度热点 | 1,024 | 256 | 4 | 4x |
| 极端热点 | 2,048 | 256 | **8** | **8x** |

> **结论**: decode 场景下 $bt = \text{batch-size}$ **不能**消除方案 A 的 weight reload 问题。
> $T_{\text{local}}$ 由 token-expert pair 总数决定，其上界为 $\text{batch-size} \times k$，可远大于 batch size 本身。
> 方案 B（token-stationary）中每个 expert 权重仍只加载一次，不受 $T_{\text{local}}$ 膨胀影响。
>
> **bt=64 在当前配置下的优势**: 当 $T_{\text{local}} = 64$ (均匀负载) 时，$n_{bt} = 64/64 = 1$，方案 A 无冗余权重读取。但一旦负载不均导致 $T_{\text{local}} > 64$，方案 A 仍将产生冗余权重加载。

---

## 附录 A: LLO 文件索引

编译 ID: `1777793912529267903`
文件前缀: `fused-moe-k_8-bt_64_64_64-bf_1024_1024-bd1_2048_2048-bd2_2048_2048-shared_expert_bse_256.1-`

| Pass | 文件后缀 | 用途 |
|------|---------|------|
| 00 | `fingerprints.txt` | 编译指纹 |
| 02 | `original.txt` | 初始 LLO (253K 行) |
| 14 | `post-MXU-assigner.txt` | MXU 分配后 |
| 23 | `critical-path.txt` | 关键路径分析 |
| 46 | `schedule-analysis_packed-bundles-pre-ra.txt` | 寄存器分配前调度 |
| 63 | `schedule-analysis_packed-bundles-post-ra.txt` | 寄存器分配后调度 |
| 67 | `pre-delay_hlo-static-per-bundle-utilization.txt` | 延迟插入前利用率 |
| 69 | `final_hlo-static-per-bundle-utilization.txt` | 最终利用率 |
| 70 | `schedule-analysis_final_bundles.txt` | 最终 bundle 调度统计 |
| 71 | `final_bundles.txt` | 最终 VLIW 汇编 (36,121 bundles) |

## 附录 B: 计算公式速查

```python
# Per-expert weight size
W_expert = 3 * hidden_size * intermediate_size * sizeof(dtype)
         = 3 * 8192 * 2048 * 2 = 96 MB

# Total weight reads per bt tile (only 1 tile now!)
W_total = local_num_experts * W_expert = 64 * 96 = 6,144 MB

# MXU FLOPs per expert (executed, btc=64)
FLOPS_ffn1 = 2 * num_bf * num_bd1 * t_packing * 2 * btc * (bd1/t_packing) * bf
           = 2 * 2 * 4 * 2 * 2 * 64 * 1024 * 1024 = 4,295M
FLOPS_ffn2 = num_bf * num_bd2 * t_packing * 2 * btc * bf * (bd2/t_packing)
           = 2 * 4 * 2 * 2 * 64 * 1024 * 1024 = 2,148M
FLOPS_total = 6,442 MFLOPS per expert

# Arithmetic Intensity
AI_useful = (FLOPS_total * dyn_sz / btc) / W_expert
          = (6,442M * 8/64) / 96M = 8.4 FLOPs/byte

# HBM-limited time (1 tile only!)
T_hbm = (6,144 + 96) MB / 3,690 GB/s = 1.69 ms

# Compute-limited time (executed)
T_compute = (64 * 6,442M) / 2,307T = 0.18 ms
```
