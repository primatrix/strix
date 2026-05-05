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

### 0.1 单 TensorCore (Chiplet) 硬件参数

> TPU v7x 每颗芯片由 2 个 chiplet 组成, 每个 chiplet 包含 1 个 TensorCore + 2 个 SparseCore + 96 GB HBM. JAX 将每个 chiplet 视为独立设备. 以下参数为 **单 TensorCore (chiplet)** 规格, 即 Pallas kernel 可用资源.

| 组件 | 规格 | 说明 |
|------|------|------|
| **HBM 容量** | 96 GB | 每 chiplet 独立 HBM (芯片级 192 GB) |
| **HBM 带宽** | 3690 GB/s | 每 chiplet 双向聚合带宽 (芯片级 7380 GB/s) |
| **VMEM 容量** | 64 MiB | 片上 Scratchpad (软件管理) |
| **SPR** | 4096 个 | 32-bit 标量寄存器 |
| **VPR** | 32 个 × 4 KB | 向量寄存器, 每个 8×128×32bit |
| **MXU** | 2 × MXU | 矩阵乘法单元 (Dual MXU) |
| **MXU 峰值算力** | 1154 TFLOPS (BF16) | 单 TC; 芯片级 2307 TFLOPS |
| **VPU** | 1 个 | 向量处理单元 (elementwise/reduce) |
| **DMA 引擎** | 1 个 | HBM ↔ VMEM 数据搬运 |
| **对齐要求** | 128 元素 | Block 维度必须被 128 整除 |
| **Ridge Point** | ~313 FLOPs/byte | MXU_peak / HBM_BW |

### 0.2 数据流层次

```text
┌──────────────────────────────────────────────────────────┐
│                        HBM (96 GB / chiplet)                │
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
│           MXU (1154 TFLOPS)  │  VPU (elementwise)        │
└──────────────────────────────────────────────────────────┘
```

### 0.3 互联参数 (芯片内 + 芯片间)

| 互联类型 | 拓扑 | 带宽 | 说明 |
|----------|------|------|------|
| **D2D** (chiplet 间) | 片内直连 | 1200 GB/s | 6× ICI 单轴, chiplet 间 collective |
| **ICI** (芯片间) | 3D Torus | 200 GB/s/axis/方向 | 总双向 1200 GB/s/chip, 全双工 |
| **DCN** (机架间) | 数据中心网络 | 100 Gbps | Pod 间通信 |

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

---

## 1. Stage 1: Metadata 计算 (Gate, TopK, Permute, AllReduce)

### 1.1 功能描述

Stage 1 包含三个阶段:

**Phase A — Gate + TopK 计算 (Kernel 外, XLA 编译)**:
对每个设备的本地 token (`T_L` 个), 执行 Gate 线性投影 (`hidden_states @ W_gate`) 得到路由 logits, 经 score function (sigmoid/softmax) 和 grouped top-k 选择, 产生 `topk_weights` 和 `topk_ids`, 写入 HBM. 该步骤在 Pallas kernel 外部以普通 XLA 编译方式执行.

**Phase A' — Permute: Token-major → Expert-major 重排 (理论分析)**:
将 token 按 `topk_ids` 路由信息重排为 Expert:Tokens 连续布局, 使每个专家接收的 token 在内存中连续存储, 方便 AlltoAll 一次性批量发送. 当前 kernel 未单独实现此步骤 — permute 隐式融合在 Stage 2 的 `fori_loop` 串行 scatter 中.

**Phase B — In-kernel Metadata (Pallas kernel 内)**:
在 kernel 启动后, 一次性从 HBM 加载 Phase A 预计算的全部 T_L 个 token 的 top-k 路由结果, 计算所有 token 的 token-to-expert 映射, 并通过单次跨设备 AllReduce 交换路由元数据, 使每个设备知道每个专家将接收多少 token 以及从哪些设备接收.

### 1.2 对应源码

**Phase A: Gate + TopK** (参考 `sgl_jax/srt/layers/gate.py`):

```python
# GateLogit.__call__: 线性投影 + Score Function
logits = jnp.dot(hidden_states, W_gate,            # (T_L, H) × (H, E) → (T_L, E)
                 precision=jax.lax.Precision.HIGHEST)
scores = jax.nn.sigmoid(logits)                     # DeepSeek V3: sigmoid
# 其他 score_func: softmax (Qwen3-MoE), tanh (预留)

# TopK.__call__: Biased Grouped TopK (DeepSeek V3)
scores_biased = scores + correction_bias[None, :]   # 加 expert bias (可选)
scores_grouped = scores_biased.reshape(T_L, n_group, E // n_group)
group_scores = sum(top_k(scores_grouped, k=2)[0], axis=-1)  # top-2 per group
group_idx = top_k(group_scores, k=topk_group)[1]            # 选 top groups
mask = build_group_mask(group_idx, n_group)
topk_ids = top_k(where(mask, scores_biased, -inf), k=K)[1]
topk_weights = take_along_axis(scores, topk_ids, axis=1)    # 用 unbiased scores

# 重归一化 + 缩放
topk_weights /= sum(topk_weights, axis=-1, keepdims=True)
topk_weights *= routed_scaling_factor                        # DeepSeek V3: 2.5
```

> `W_gate` 为 `(H, E)` 的 float32 权重, 在每个 EP 设备上**全量复制** (不按 EP 切分). `correction_bias` 可选, 仅 DeepSeek V3 (`topk_method="noaux_tc"`) 启用. Score function 可选: `sigmoid` (DeepSeek V3), `softmax` (Qwen3-MoE).

**Phase A': Permute** (理论操作, 当前 kernel 未独立实现):

```text
# 目标: token-major → expert-major 连续布局
# 使 AlltoAll 可按 expert 批量发送, 消除 Stage 2 串行 DMA
permute_indices = compute_scatter_indices(topk_ids, expert_starts)
for t in 0..T_L-1:
    for k in 0..K-1:
        expert_id = topk_ids[t, k]
        offset = permute_indices[t, k]
        permuted_tokens[expert_id, offset, :] = hidden_states[t, :]
```

**Phase B: In-kernel Metadata** (源码: `kernels/_fused_moe_impl.py`):

```text
# 一次性处理所有 T_L 个 token 的元数据:
  ├── start_fetch_topk()                # HBM → VMEM: topk_weights, topk_ids (全部 T_L 个)
  ├── wait_fetch_topk()                 # 等待 DMA 完成
  ├── 计算 t2e_routing                  # VPU: broadcast compare + reduce (T_L tokens)
  ├── 计算 expert_sizes                 # VPU: mask reduction
  └── all_reduce_metadata()             # ICI: 跨设备 allgather + prefix sum (单次)
```

### 1.3 输入 / 输出

**Phase A (Gate + TopK) 输入 / 输出:**

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `hidden_states` | `(T_L, H)` | BF16 | `T_L × H × 2` |
| **Input** | `W_gate` | `(H, E)` | F32 | `H × E × 4` |
| **Input** | `correction_bias` (可选) | `(E,)` | F32 | `E × 4` |
| **Output→HBM** | `topk_weights` | `(T_L, K)` | F32 | `T_L × K × 4` |
| **Output→HBM** | `topk_ids` | `(T_L, K)` | I32 | `T_L × K × 4` |

> `W_gate` 在每个 EP 设备上全量复制 (不随 `D` 切分). DeepSeek-V3 配置: `W_gate` = 8192×256×4 = 8.0 MB.

**Phase A' (Permute) 输入 / 输出 (理论):**

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `hidden_states` | `(T_L, H)` | BF16 | `T_L × H × 2` |
| **Input** | `topk_ids` | `(T_L, K)` | I32 | (已在 HBM) |
| **Output** | `permuted_tokens` | `(E, max_M, H)` | BF16 | `T_L × K × H × 2` |

> Permute 将每个 token 复制 K 次到对应 expert 位置, 总写入量 = `T_L × K × H × B_t`.

**Phase B (In-kernel Metadata) 输入 / 输出:**

| 方向 | 张量 | Shape | Dtype | 字节数 |
|------|------|-------|-------|--------|
| **Input** | `topk_weights_hbm` | `(T_L, K)` | F32 | `T_L × K × 4` |
| **Input** | `topk_ids_hbm` | `(T_L, K)` | I32 | `T_L × K × 4` |
| **Output** | `t2e_routing` | `(T_L, padded_K)` | I32 | `T_L × pad128(K) × 4` |
| **Output** | `expert_sizes` | `(1, padded_E)` | I32 | `pad128(E) × 4` |
| **Output** | `expert_starts` | `(1, padded_E)` | I32 | `pad128(E) × 4` |
| **Output** | `d2e_count` | `(D, 1, padded_E)` | I32 | `D × pad128(E) × 4` |
| **Output** | `expert_offsets` | `(2, padded_E)` | I32 | `2 × pad128(E) × 4` |

### 1.4 计算步骤分解

#### Step 1.0: Gate + TopK + Permute (Phase A, XLA 编译)

**Step 1.0.1: Gate 线性投影 (MXU GEMM)**

```text
logits = hidden_states @ W_gate     # (T_L, H) × (H, E) → (T_L, E)

FLOPs_gate = 2 × T_L × H × E
```

HBM 数据搬运:

```text
Bytes_W_gate     = H × E × 4               # 读 W_gate (float32, 全量复制)
Bytes_input      = T_L × H × B_t           # 读 hidden_states
Bytes_output     = T_L × E × 4             # 写 logits (float32)

Total_Bytes_gate = H × E × 4 + T_L × H × B_t + T_L × E × 4
```

> 对于 Decode 场景 (`T_L` 小), `Bytes_W_gate` 项主导 (8.0 MB for DeepSeek-V3), 使 Gate GEMM 严格 HBM BW bound. AI ≈ `T_L / 2` FLOPs/byte, 远低于 ridge point 313.

**Step 1.0.2: Score Function (VPU)**

```text
# DeepSeek V3: sigmoid(x) = 1/(1+exp(-x))
FLOPs_score = 4 × T_L × E              # negate + exp + add + reciprocal

# Qwen3-MoE: softmax
FLOPs_score = 5 × T_L × E              # exp + sum + div (近似)
```

> 在 (T_L, E) 矩阵上的逐元素操作, 数据已在寄存器/VMEM, 无额外 HBM 访问.

**Step 1.0.3: Grouped TopK 专家选择 (VPU)**

```text
# DeepSeek V3: biased grouped topk
# 1. add bias:           T_L × E       additions
# 2. reshape + top-2:    T_L × E       comparisons (per-group top-2)
# 3. group score sum:    T_L × n_group additions
# 4. select top groups:  T_L × n_group comparisons
# 5. build mask:         T_L × E       conditionals
# 6. masked top-K:       T_L × E       comparisons
# 7. gather weights:     T_L × K       memory ops

FLOPs_topk ≈ 6 × T_L × E + T_L × K
```

**Step 1.0.4: 重归一化 + 缩放 (VPU)**

```text
FLOPs_renorm = 3 × T_L × K             # sum + div + scale
```

**Step 1.0.5: Permute — Token-major → Expert-major 重排 (理论分析)**

```text
# 目标: 将 hidden_states 按 topk_ids 重排为 expert-major 连续布局
# 效果: AlltoAll 可按 expert 批量 DMA, 消除 Stage 2 串行 fori_loop 开销
#
# 当前 kernel: permute 融合在 Stage 2 scatter 的 fori_loop 中
# 理论优化: 独立 permute → batch scatter

FLOPs_permute ≈ T_L × K                  # 索引计算 (negligible)

Bytes_permute_read  = T_L × H × B_t      # 读源 token (每 token 读一次)
Bytes_permute_write = T_L × K × H × B_t  # 写到 K 个 expert 位置

Total_Bytes_permute = T_L × H × B_t × (1 + K)
AI_permute ≈ 0                           # 纯 DMA scatter, 几乎无计算
T_permute = Total_Bytes_permute / HBM_BW
```

> 对于 DeepSeek-V3 (K=8, H=8192, B_t=2): `T_permute = T_L × 8192 × 2 × 9 / HBM_BW`.
> T_L=64 时 T_permute ≈ 2.6 μs, T_L=256 时 ≈ 10.2 μs. 若实现独立 Permute, 可用向量化 scatter 替代 Stage 2 的串行 DMA, 但需额外 HBM buffer.

**Phase A 总计:**

```text
FLOPs_phase_a_mxu = 2 × T_L × H × E                    # Gate GEMM (MXU, 主导)
FLOPs_phase_a_vpu = 10 × T_L × E + 3 × T_L × K         # Score + TopK + Renorm (VPU)

Bytes_HBM_phase_a = H × E × 4 + T_L × H × B_t          # 读 (W_gate + hidden_states)
                  + 2 × T_L × K × 4                     # 写 (topk_weights + topk_ids)

# Permute (理论, 若独立实现):
Bytes_HBM_permute = T_L × H × B_t × (1 + K)
```

#### Step 1.1: Fetch TopK (DMA)

```text
T_fetch_topk = (T_L × K × 4 × 2) / HBM_BW
             = (2 × T_L × K × 4) / 3.69e12   [秒]
```

两次 DMA (weights + ids), 每次 `T_L × K × 4` bytes.

#### Step 1.2: 计算路由掩码 (VPU)

```python
# 一次性处理所有 T_L 个 token
expert_iota = broadcasted_iota(I32, (1, 1, padded_E), 2)
routing_expanded = expand_dims(t2e_routing[:, :K], axis=2)
mask = (routing_expanded == expert_iota).astype(I32)        # (T_L, K, padded_E)
expert_sizes = sum(mask, axis=(0,1), keepdims=True)         # (1, padded_E)
```

FLOPs:

```text
FLOPs_routing = T_L × K × pad128(E)           # broadcast compare
             + T_L × K × pad128(E)            # reduce sum
             = 2 × T_L × K × pad128(E)
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

**Phase A (Gate + TopK):**

```text
FLOPs_gate        = 2 × T_L × H × E                    # MXU GEMM

Bytes_HBM_gate    = H × E × 4                           # W_gate 读 (主导项)
                  + T_L × H × B_t                        # hidden_states 读
                  + T_L × E × 4                          # logits 写

AI_gate           = 2 × T_L × H × E / Bytes_HBM_gate
                  ≈ T_L / 2                              # W_gate 主导时 (Decode)

T_gate            = max(FLOPs_gate / MXU_peak, Bytes_HBM_gate / HBM_BW)
                  ≈ Bytes_HBM_gate / HBM_BW              # 严格 HBM BW bound
```

> Gate GEMM 的 AI 在 Decode 场景极低 (AI ≈ `T_L/2`), 因为每次需读取完整 `W_gate` (DeepSeek-V3: 8.0 MB), 而 FLOPs 与 `T_L` 成正比. Score function 和 TopK 的 VPU 计算量 (~10 × T_L × E FLOPs) 可忽略.

**Phase A' (Permute, 理论):**

```text
Bytes_HBM_permute = T_L × H × B_t × (1 + K)             # 读 tokens + 写 K 份副本

T_permute         = Bytes_HBM_permute / HBM_BW
                  = T_L × H × B_t × (1 + K) / HBM_BW
```

> Permute 为纯 DMA 操作 (AI ≈ 0). 若独立实现, 其 HBM 开销与 Stage 2 scatter 部分重叠 — 即 Permute 预付的 HBM 搬运可减少 Stage 2 的串行 DMA 开销.

**Phase B (In-kernel Metadata, 一次性处理所有 T_L 个 token):**

```text
Bytes_HBM_phase_b = 2 × T_L × padded_K × 4              # topk weights + ids 读 (DMA)

FLOPs_phase_b     = 2 × T_L × K × pad128(E)              # routing mask
                  + D × pad128(E) × 2                    # prefix sum

Bytes_ICI_phase_b = log2(D) × pad128(E) × 4             # allreduce (power-of-2)
                  + D × 2 × sync_barrier_bytes            # barrier messages

T_compute_phase_b = FLOPs_phase_b / VPU_peak             # VPU-bound (无 MXU)
T_hbm_phase_b     = Bytes_HBM_phase_b / HBM_BW
T_ici_phase_b     = Bytes_ICI_phase_b / ICI_BW + log2(D) × T_barrier

T_phase_b         = max(T_compute_phase_b, T_hbm_phase_b) + T_ici_phase_b
```

**Stage 1 总延迟:**

```text
T_stage1 = T_gate + T_permute + T_phase_b

# 简化 (T_gate 被 W_gate 读取主导, T_phase_b 被 ICI AllReduce 主导):
T_stage1 ≈ (H × E × 4) / HBM_BW
          + T_L × H × B_t × (1 + K) / HBM_BW              # Permute (理论)
          + log₂(D) × T_ici_step
```

> **瓶颈特征**:
> - Phase A: **HBM BW bound** — 读取 W_gate (DeepSeek-V3: ~2.2 μs).
> - Phase A': **HBM BW bound** — 纯 scatter DMA (T_L=64 时 ~2.6 μs).
> - Phase B: **ICI 延迟 bound** — AllReduce payload 极小 (1 KB/round), 延迟由 ICI 启动开销主导. 一次性 allreduce 所有 token 的 expert_sizes, 无 tile 循环累积.
> - 对于 `D=1` (单设备), AllReduce 退化为 noop, 仅剩 Gate + Permute 开销.
>
> **关键观察**: 随 EP 规模增大, Stage 3 延迟因 `E_L` 减小而下降 (O(1/D)), 但 Stage 1 AllReduce 延迟增长 (O(log₂D)). 在 EP ≥ 128 时, Stage 1 占端到端 MoE 延迟比例可达 30%–76% (见 §1.7).

### 1.6 消融实验设计

固定 DeepSeek-V3-like 配置 (E=256, K=8, H=8192, I=2048), 扫描 EP 规模:

```python
for ep_size in [8, 32, 64, 128, 256]:
    for scenario in ["decode", "prefill"]:
        T = 512 if scenario == "decode" else 8192
        mesh = create_mesh(ep_size)
        # 分别测量: Gate GEMM, Permute (若实现), In-kernel metadata
        topk_weights, topk_ids = gate_and_topk(hidden_states, W_gate, ...)
        result = fused_ep_moe(mesh, tokens, w1, w2, w3,
                              topk_weights, topk_ids, top_k=8, ...)
        profile(result)  # 采集各阶段延迟
```

| 实验 | EP Size | 场景 | T_L | 验证目标 |
|------|---------|------|-----|---------|
| EP-8 Decode | D=8 | T=512 | 64 | 基线: AllReduce 3 轮 |
| EP-32 Decode | D=32 | T=512 | 16 | AllReduce 5 轮 |
| EP-64 Decode | D=64 | T=512 | 8 | AllReduce 6 轮 |
| EP-128 Decode | D=128 | T=512 | 4 | AllReduce 7 轮 |
| EP-256 Decode | D=256 | T=512 | 2 | AllReduce 8 轮 |
| EP-8 Prefill | D=8 | T=8192 | 1024 | Permute 主导 (40.9 μs) |
| EP-32 Prefill | D=32 | T=8192 | 256 | Permute + AR 均衡 |
| EP-64 Prefill | D=64 | T=8192 | 128 | AllReduce 6 轮 |
| EP-128 Prefill | D=128 | T=8192 | 64 | AllReduce 7 轮 |
| EP-256 Prefill | D=256 | T=8192 | 32 | AllReduce 8 轮 |

#### Exp: Pure JAX Stage 1 对照实验

> 观察 XLA 编译器对 Gate + TopK + Metadata + AllReduce 的自动融合效果.

```python
@jax.jit
def stage1_pure_jax(hidden_states, W_gate, correction_bias, E, K):
    """Stage 1 完全由 JAX 实现, 由 XLA 编译器自动优化融合."""
    # Phase A: Gate + TopK
    logits = jnp.dot(hidden_states, W_gate)
    scores = jax.nn.sigmoid(logits)
    topk_ids = jax.lax.top_k(scores, K)[1]
    topk_weights = jnp.take_along_axis(scores, topk_ids, axis=1)
    topk_weights /= topk_weights.sum(axis=-1, keepdims=True)

    # Phase B: Metadata (当前在 Pallas kernel 内, 此处改为 JAX)
    expert_mask = jax.nn.one_hot(topk_ids, E)           # (T_L, K, E)
    expert_sizes = expert_mask.sum(axis=(0, 1)).astype(jnp.int32)
    global_sizes = jax.lax.psum(expert_sizes, 'ep')      # AllReduce
    expert_starts = jnp.cumsum(global_sizes) - global_sizes

    return topk_weights, topk_ids, expert_starts, global_sizes

# 对照实验: 与 Pallas Phase B 比较
for ep_size in [8, 32, 64, 128, 256]:
    for T in [512, 8192]:
        # JAX 版本 (XLA 自动融合)
        jax_result = pjit(stage1_pure_jax, ...)(hidden_states, W_gate, ...)
        # Pallas 版本 (手写 kernel Phase B)
        pallas_result = fused_ep_moe(...)  # 仅 Stage 1
        profile_compare(jax_result, pallas_result)
```

| 验证目标 | 预期 |
|---------|------|
| XLA 融合程度 | Gate GEMM + Score + TopK 是否合并为单 XLA kernel |
| AllReduce 实现差异 | XLA psum vs Pallas 手写 allreduce 的延迟对比 |
| Metadata 计算开销 | XLA elementwise fusion vs Pallas VPU 手动管理 |
| 编译时间 | JAX jit 编译时间 vs Pallas kernel 编译时间 |

### 1.7 数值分析: EP 缩放对 Stage 1 耗时的影响

基于 DeepSeek-V3-like 配置: E=256, K=8, H=8192, I=2048, `W_gate` = float32 (8.0 MB).

> **假设**: `T_ici_step ≈ 2 μs` (ICI 单步延迟, 含 startup + barrier). 实际值需通过 §1.6 消融实验标定.
>
> **设计**: Stage 1 元数据一次性处理所有 T_L 个 token, 单次 AllReduce 交换 expert_sizes, 无 per-tile 重复.

#### 1.7.1 Decode 场景 (T=512)

| D | T_L | T_gate (μs) | T_permute (μs) | T_AR (μs) | **T_stage1** (μs) | T_stage3 (μs) | **S1/S3** |
|---|-----|------------|-----------------|-----------|-------------------|--------------|-----------|
| 8 | 64 | 2.6 | 2.6 | 6 | **11.8** | 832 | 1.4% |
| 32 | 16 | 2.3 | 0.6 | 10 | **12.9** | 208 | 6.2% |
| 64 | 8 | 2.3 | 0.3 | 12 | **14.6** | 104 | 14.0% |
| 128 | 4 | 2.3 | 0.2 | 14 | **16.5** | 52 | 31.7% |
| 256 | 2 | 2.3 | 0.1 | 16 | **18.4** | 26 | **70.8%** |

> **计算说明**:
> - `T_gate = (H×E×4 + T_L×H×2 + T_L×E×4) / HBM_BW` — W_gate 读取 (8.0 MB) 主导.
> - `T_permute = T_L × H × B_t × (1+K) / HBM_BW` — 一次性 DMA scatter 全部 T_L 个 token.
> - `T_AR = log₂(D) × T_ici_step` — 单次 AllReduce.
> - `T_stage3 ≈ E_L × 3 × H × I × B_w / HBM_BW` — Expert FFN 权重读取主导.

#### 1.7.2 Prefill 场景 (T=8192)

| D | T_L | T_gate (μs) | T_permute (μs) | T_AR (μs) | **T_stage1** (μs) | T_stage3 (μs) | **S1/S3** |
|---|-----|------------|-----------------|-----------|-------------------|--------------|-----------|
| 8 | 1024 | 7.1 | 40.9 | 6 | **54.0** | 6659 | 0.8% |
| 32 | 256 | 3.5 | 10.2 | 10 | **23.7** | 416 | 5.7% |
| 64 | 128 | 2.9 | 5.1 | 12 | **20.0** | 104 | 19.2% |
| 128 | 64 | 2.6 | 2.6 | 14 | **19.2** | 52 | 36.9% |
| 256 | 32 | 2.4 | 1.3 | 16 | **19.7** | 26 | **75.8%** |

> Prefill EP=8 的 T_permute 较高 (40.9 μs) 是因为 T_L=1024 个 token 需全量重排 (Permute 主导).

#### 1.7.3 各分项占比分析

```text
T_stage1 = T_gate + T_permute + T_AR

Decode (T=512):
           ┌─────────────────────────────────────────────────────┐
  EP=8     │████ Gate  │████ Perm │██████ AR                     │ ← Gate+Perm 与 AR 均衡
  EP=32    │██ Gate │█ Perm │██████████████ AR                   │
  EP=64    │██ Gate │P│████████████████ AR                       │
  EP=128   │█ G │P│██████████████████ AR                        │
  EP=256   │█ G│P│████████████████████ AR                       │ ← AR 主导
           └─────────────────────────────────────────────────────┘

Prefill (T=8192):
           ┌─────────────────────────────────────────────────────┐
  EP=8     │█ G│████████████████████████ Perm │██ AR             │ ← Permute 主导
  EP=32    │██ G│██████ Perm │██████████ AR                      │
  EP=64    │██ G│███ Perm │████████████ AR                       │
  EP=128   │██ G│██ Perm │██████████████ AR                      │
  EP=256   │█ G│█ P│████████████████████ AR                     │ ← AR 主导
           └─────────────────────────────────────────────────────┘
```

#### 1.7.4 关键观察

1. **Gate GEMM 延迟近乎恒定**: T_gate ≈ 2.3–7.1 μs, 由 `W_gate` 读取 (8.0 MB) 主导. 仅在 T_L > 256 时 hidden_states 读取才显著增加延迟.

2. **Permute 延迟与 T_L 成正比**: T_permute = `T_L × 8192 × 2 × 9 / HBM_BW`. Decode (T_L=2~64) 仅 0.1–2.6 μs; Prefill 大 T_L (T_L=1024) 高达 40.9 μs, 成为 Stage 1 主导分项.

3. **AllReduce 为单次调用, 无 tile 循环累积**: T_AR = `log₂(D) × T_ici_step`, 从 6 μs (D=8) 增长到 16 μs (D=256). AllReduce payload 仅 1 KB (pad128(E)×4), 完全由 ICI startup 延迟主导.

4. **Stage 1 在大 EP 下占比显著提升**:
   - EP ≤ 32: Stage 1 < 7% 端到端, 可忽略
   - EP = 64: Stage 1 ≈ 14–19%, 值得关注
   - EP = 128: Stage 1 ≈ 32–37%, 成为重要优化目标
   - EP = 256: Stage 1 ≈ **71–76%**, **成为主要瓶颈** (因 Stage 3 权重仅 96 MB / 26 μs)

5. **优化方向**:
   - **Gate 权重量化**: `W_gate` 从 float32 (8 MB) → BF16 (4 MB), T_gate 降低 ~40%
   - **AllReduce 优化**: 减少 `T_ici_step` (更高效的 barrier protocol); 探索非阻塞 AllReduce 与 Gate GEMM 重叠
   - **Permute 优化**: 大 T_L 场景 (Prefill) Permute 成为主导分项, 可探索 VMEM 驻留的向量化 scatter 替代 HBM-based 重排
   - **TODO: 利用 Group 局部性降低跳数**: Grouped TopK 强制每个 token 的 K 个 expert 集中在 `topk_group` 个 group 内 (DeepSeek V3: 256 expert / 8 group, 每 token 仅选 4 group). 若将 group 与 EP 设备拓扑对齐 (同一 group 的 expert 放在 ICI 拓扑相邻的设备上), AllReduce 和 All2All 的通信跳数可从 O(log₂D) 降至 O(log₂(D × topk_group / num_group)), 显著减少 Stage 1 AllReduce 和 Stage 2 Dispatch 延迟

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

### 2.7 DMA 启动开销估计值

以下分析使用的 DMA / ICI 启动开销为估计值, 需通过消融实验标定 (见 2.12 Exp D):

| 参数 | 估计值 | 说明 |
|------|--------|------|
| `T_dma_launch` | ~200 ns | 单次 HBM ↔ VMEM DMA 操作启动开销 |
| `T_ici_launch` | ~500 ns | 单次 ICI remote_copy 操作启动开销 (含两端 setup) |

### 2.8 Decode 场景 VMEM 可行性分析

**问题**: 大 EP 并行下, token 数据能否一次性驻留 VMEM, 避免 HBM scratch buffer 中转?

发送侧 VMEM 需求 = `T_L × H × B_t`, 接收侧 VMEM 需求 = `E_L × M_avg × H × B_t` (其中 `M_avg = T×K/E`).

**DeepSeek-V3-like 配置**: E=256, K=8, T=256, M_avg = 256×8/256 = 8 tokens/expert:

| D | T_L | E_L | 发送缓冲 (H=4096) | 发送缓冲 (H=8192) | 接收缓冲 (H=4096) | 接收缓冲 (H=8192) | 总计 (H=8192) | VMEM 占比 |
|---|-----|-----|-------------------|-------------------|-------------------|-------------------|--------------|----------|
| 4 | 64 | 64 | 512 KB | 1 MB | 4 MB | 8 MB | **9 MB** | 13.4% |
| 16 | 16 | 16 | 128 KB | 256 KB | 1 MB | 2 MB | **2.25 MB** | 3.4% |
| 64 | 4 | 4 | 32 KB | 64 KB | 256 KB | 512 KB | **576 KB** | 0.86% |

> **结论**: Decode 场景下, 即使 H=8192 + D=4 的最大 VMEM 需求也仅占 13.4%, **A2A 缓冲完全可放入 VMEM**. D≥16 时 VMEM 占用不到 4%.

#### VMEM 驻留的性能收益

当前数据流: `HBM → DMA → VMEM → ICI → 目标 HBM (a2a_s) → DMA → VMEM → MXU`

优化数据流: `HBM → DMA → VMEM → ICI → 目标 VMEM → MXU` (省去一次 HBM round-trip)

| D | 省去的 HBM 搬运量 | 节省时间 | Stage 3 耗时 | 相对收益 |
|---|------------------|---------|-------------|---------|
| 4 | 2×8 MB = 16 MB | 4.3 μs | 1745 μs | 0.25% |
| 16 | 2×2 MB = 4 MB | 1.1 μs | ~407 μs | 0.27% |
| 64 | 2×512 KB = 1 MB | 0.27 μs | ~109 μs | 0.25% |

> VMEM 驻留节省的 HBM round-trip 本身收益极小 (<0.3%). 真正价值在于**使 per-device 合并 DMA 成为可能** (见 2.9 节), 以及**消除 Stage 3 token staging 的小 DMA 串行开销**.

### 2.9 合并 DMA 收益分析 (Per-device Batch vs Per-expert Sequential)

#### 当前实现的开销结构

当前 `fori_loop` 实现中, 每个 token-expert 对发起一次独立的 DMA/remote_copy:

```text
total_ops = T_L × K
T_per_expert_seq = total_ops × (T_ici_launch + tok_bytes / ICI_BW)
```

每个 op 传输 `H × B_t` bytes, 启动开销固定 500 ns:

| H | tok_bytes | ICI 传输时间 | 总耗时/op | **启动开销占比** |
|------|-----------|------------|----------|----------------|
| 4096 | 8 KB | 41 ns | 541 ns | **92.4%** |
| 8192 | 16 KB | 82 ns | 582 ns | **85.9%** |

> **核心问题**: 逐 expert 的小 DMA 模式下, **85-92% 的时间浪费在 ICI launch overhead**, 实际数据传输仅占 8-15%.

#### 小 DMA 的有效 ICI 带宽

| 传输粒度 | 有效 ICI 带宽 | 峰值利用率 |
|---------|-------------|----------|
| 8 KB (H=4096, 单 token) | 14.8 GB/s | 7.4% |
| 16 KB (H=8192, 单 token) | 27.5 GB/s | 13.7% |
| 128 KB (8 tokens batch) | 110 GB/s | 55% |
| 1 MB (64 tokens batch) | 183 GB/s | 91.3% |
| 2 MB (128 tokens batch) | 191 GB/s | 95.4% |

#### Per-device 合并 DMA 方案

将同一目标设备的所有 expert 数据打包为一次 remote_copy:

```text
ops = D - 1 (remote) + 1 (local)
bytes_per_device = T_L × K / D × H × B_t    # 均匀路由假设
T_per_device_batch = (D-1) × (T_ici_launch + bytes_per_device / ICI_BW)
```

实现前提: token 已在 VMEM (见 2.8), 按目标 device 重排后发起单次 remote_copy.

#### H=8192 消融对比

| D | T_L | Pairs | Per-expert seq (μs) | Per-device batch (μs) | **Speedup** | A2A 占 Stage3 (before→after) |
|---|-----|-------|--------------------|--------------------|-------------|--------------------------|
| 4 | 64 | 512 | 298 | 33 | **9.0×** | 17% → 1.9% |
| 16 | 16 | 128 | 74.5 | 17.4 | **4.3×** | 18% → 4.3% |
| 64 | 4 | 32 | 18.6 | ~18.6 | **~1×** | 17% → 17% |

#### H=4096 消融对比

| D | T_L | Pairs | Per-expert seq (μs) | Per-device batch (μs) | **Speedup** | A2A 占 Stage3 (before→after) |
|---|-----|-------|--------------------|--------------------|-------------|--------------------------|
| 4 | 64 | 512 | 277 | 17.2 | **16.1×** | 16% → 1.0% |
| 16 | 16 | 128 | 69.3 | 12.4 | **5.6×** | 17% → 3.0% |
| 64 | 4 | 32 | 17.3 | ~17.3 | **~1×** | 16% → 16% |

> **关键发现**:
>
> 1. D=4~16 时, per-device 合并 DMA 带来 **4-16× A2A 加速**, 将 A2A dispatch 从 Stage 3 的 17% 降到 1-4%
> 2. D=64 时由于 token 极度稀疏 (每设备平均 0.5 个 pair), 每个 "batch" 已退化为单 token, 无合并收益
> 3. H 越小, 合并收益越大 (小传输的 launch 开销占比更高)

### 2.10 Prefill vs Decode 逐 Expert 发送耗时对比

**配置**: E=256, K=8, H=8192

#### Decode (bt = T_L, 全量一次)

| D | T_L=bt | Pairs/tile | Per-expert seq (μs) | Per-device batch (μs) | Stage 3 (μs) | A2A 占 Stage3 (before→after) |
|---|--------|-----------|--------------------|--------------------|-------------|--------------------------|
| 4 | 64 | 512 | 298 | 33 | 1745 | 17% → 1.9% |
| 16 | 16 | 128 | 74.5 | 17.4 | 407 | 18% → 4.3% |
| 64 | 4 | 32 | 18.6 | 18.6 | 109 | 17% → 17% |

#### Prefill (bt=128, num_bt=T_L/128)

| D | T_L | bt | Pairs/tile | Per-expert seq (μs) | Per-device batch (μs) | Stage 3/tile (μs) | A2A 占 Stage3 (before→after) |
|---|-----|-----|-----------|--------------------|--------------------|------------------|--------------------------|
| 4 | 512 | 128 | 1024 | 596 | 64.4 | 6990 | 8.5% → 0.9% |
| 16 | 128 | 128 | 1024 | 596 | 64.4 | 6990 | 8.5% → 0.9% |
| 64 | 32 | 32 | 256 | 149 | 34.8 | ~1745 | 8.5% → 2.0% |

> **观察**:
>
> 1. Decode 场景 A2A dispatch 占 Stage 3 比例 (~17%) 约为 Prefill (~8.5%) 的 2×, 因为 Decode 的 bt 小导致 Stage 3 权重加载时间短, A2A 相对占比更高
> 2. Per-device batching 在 Prefill 场景同样有效, 可将 A2A 从 8.5% 降到 <1%
> 3. D=64 Decode 是最不利场景: token 极度稀疏且 batching 无效, 同时 A2A 占比最高

### 2.11 单 Expert All2All 延迟扫描

**场景**: 一个 expert 从单个源设备接收 N 个 token, 对比逐 token DMA vs 单次批量 DMA.

#### H = 4096 (per_token = 8 KB)

| N tokens | 数据量 | 逐 token ICI (μs) | 批量 ICI (μs) | Speedup | 逐 token HBM (μs) | 批量 HBM (μs) |
|----------|-------|-------------------|-------------|---------|-------------------|-------------|
| 16 | 128 KB | 8.66 | 1.14 | 7.6× | 3.23 | 0.24 |
| 64 | 512 KB | 34.6 | 3.06 | 11.3× | 12.9 | 0.34 |
| 128 | 1 MB | 69.2 | 5.72 | 12.1× | 25.9 | 0.48 |
| 512 | 4 MB | 277 | 20.5 | 13.5× | 103 | 1.28 |
| 1024 | 8 MB | 554 | 40.4 | 13.7× | 206 | 2.37 |

#### H = 8192 (per_token = 16 KB)

| N tokens | 数据量 | 逐 token ICI (μs) | 批量 ICI (μs) | Speedup | 逐 token HBM (μs) | 批量 HBM (μs) |
|----------|-------|-------------------|-------------|---------|-------------------|-------------|
| 16 | 256 KB | 9.31 | 1.78 | 5.2× | 3.26 | 0.27 |
| 64 | 1 MB | 37.2 | 5.72 | 6.5× | 13.1 | 0.47 |
| 128 | 2 MB | 74.5 | 10.99 | 6.8× | 26.1 | 0.74 |
| 512 | 8 MB | 298 | 41.4 | 7.2× | 104 | 2.37 |
| 1024 | 16 MB | 596 | 82.4 | 7.2× | 207 | 4.55 |

> **收敛行为**: Speedup 随 N 增大收敛至 `T_ici_launch / T_transfer_per_token + 1`:
>
> - H=4096: 收敛至 500/41 + 1 ≈ **13.2×**
> - H=8192: 收敛至 500/82 + 1 ≈ **7.1×**
>
> 收敛值与 H 成反比 — H 越大, 每 token 传输时间越长, launch 开销相对占比越低, 合并收益越小.

### 2.12 Stage 2 消融实验详细方案

基于 2.8-2.11 理论分析, 设计以下消融实验:

#### Exp A: 逐 Expert vs 逐 Device DMA (核心实验)

```python
# A1: Baseline - 逐 expert sequential DMA (当前实现)
fused_ep_moe(..., ep_size=D)

# A2: Per-device batch DMA (需 kernel 修改)
# Stage 1 完成后, 按目标 device 重排 token, 每 device 一次 remote_copy
fused_ep_moe(..., ep_size=D, a2a_batch_mode="per_device")
```

预期: D=4~16 时 A2A dispatch 阶段 4-16× 加速.

#### Exp B: VMEM 直接缓冲 (仅 Decode)

```python
# B1: 当前 - A2A 通过 HBM scratch buffer (a2a_s_x2_hbm)
# B2: VMEM 驻留 - token 预加载到 VMEM, ICI 直接写入目标 VMEM
```

预期: 绝对收益较小 (<5 μs), 但消除了 Stage 3 的 token staging DMA 启动延迟.

#### Exp C: 单 Expert All2All 延迟扫描

```python
# 固定 ep_size=2 (最简单的跨设备场景), 扫描 token 数量
for n_tokens in [16, 64, 128, 512, 1024]:
    for H in [4096, 8192]:
        # 仅启用 Stage 2, 禁用 FFN + SE
        fused_ep_moe(
            tokens=randn(n_tokens, H),
            disable_dynamic_ffn1=True,
            disable_dynamic_ffn2=True,
            disable_shared_expert=True,
        )
```

关键指标: 每次 DMA 的有效 ICI 带宽, 验证 `T_ici_launch` 估计值 (500 ns).

#### Exp D: DMA / ICI Launch Overhead 标定

```python
# D1: 1 token × 1 expert (单次 DMA, 测最小延迟)
# D2: 1 token × N experts, N = 1,2,4,8,16 (N 次串行 DMA, 斜率 = per-DMA overhead)
# 线性拟合: T(N) = N × T_launch + N × tok_bytes / BW
# 从截距和斜率分别提取 T_launch 和有效 BW
```

> **Exp D 是所有后续分析的基础**: 上述理论分析使用 `T_dma_launch=200 ns`, `T_ici_launch=500 ns` 为估计值. 标定后可修正所有模型预测.

#### Stage 2 消融实验汇总

| 优化方向 | 理论收益 | 适用场景 | 优先级 |
|---------|---------|---------|-------|
| Per-device batch DMA | 4-16× A2A 加速, Stage3 占比 17%→1-4% | D=4~16, Decode+Prefill | **P0** |
| VMEM 直接缓冲 | <5 μs 绝对收益, 消除 staging 串行开销 | D≥4, Decode | P1 |
| D=64 稀疏优化 | 需替代方案 (路由策略/异步 DMA) | D≥64, Decode | P2 |
| Launch overhead 标定 | 修正所有理论预测 | 全场景 | **P0** |

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
- Ridge point = 313 → **严重 HBM bandwidth bound**

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
| **FFN1/FFN2 联合计算** | 见 §3.10.7 详细设计 | 消除 pipeline bubble | 达到 Memory Bound (η_mem → 100%) |

### 3.9 Compute Bound 临界 Token 数量推导

> 本节忽略实现细节 (tiling, num_bt 权重重读, VMEM 约束, 双缓冲), 仅从 SwiGLU FFN 算法本身推导 Stage 3 进入 Compute Bound 所需的最小 token 数量.

#### 3.9.1 算法定义

SwiGLU FFN per-expert 计算:

```text
Y = (SiLU(X @ W1) ⊙ (X @ W3)) @ W2

X ∈ [M, H],  W1 ∈ [H, I],  W3 ∈ [H, I],  W2 ∈ [I, H],  Y ∈ [M, H]
```

其中 `M` 为单个专家接收的 token 数.

#### 3.9.2 Per-Expert FLOPs

| 步骤 | 操作 | FLOPs |
|------|------|-------|
| FFN1 Gate | X @ W1 | 2MHI |
| FFN1 Up | X @ W3 | 2MHI |
| Activation | SiLU(gate) ⊙ up | 3MI |
| FFN2 Down | act @ W2 | 2MIH |
| **合计** | | **6MHI + 3MI ≈ 6MHI** |

#### 3.9.3 Per-Expert HBM 最小数据搬运量 (算法理论下限)

假设:
- 所有中间结果 (gate, up, act) 完美 fusion, 保留在片上 (VMEM/VPR)
- 每个专家的权重从 HBM 读取**恰好一次** (无 tiling 导致的重读)

| 数据 | 方向 | 字节数 |
|------|------|--------|
| W1 | Read | H × I × B_w |
| W3 | Read | H × I × B_w |
| W2 | Read | I × H × B_w |
| X (input tokens) | Read | M × H × B_t |
| Y (output tokens) | Write | M × H × B_t |
| **合计** | | **3HI·B_w + 2MH·B_t** |

> 与 §3.5 实现公式的关键区别: 实现中 `num_bt > 1` 时权重被重读 `num_bt` 次, 此处算法下限假设权重仅读一次. 这给出了 AI 的**理论上限**, 即进入 Compute Bound 的**最低 token 门槛**.

#### 3.9.4 Per-Expert Arithmetic Intensity

```text
AI = 6MHI / (3HI·B_w + 2MH·B_t)
   = 6MI / (3I·B_w + 2M·B_t)
```

> **关键观察**: H (hidden_size) 在分子分母中**完全约掉**, AI 仅取决于 M, I, B_w, B_t.

#### 3.9.5 Compute Bound 临界条件推导

Compute Bound 要求 AI ≥ R, 其中 R = Ridge Point = MXU_peak / HBM_BW = 1154 / 3690 ≈ **313 FLOPs/byte**.

```text
6MI / (3I·B_w + 2M·B_t) ≥ R

6MI ≥ R·(3I·B_w + 2M·B_t)

M·(6I - 2R·B_t) ≥ 3R·I·B_w
```

求解 per-expert 临界 token 数 M_crit:

```text
┌─────────────────────────────────────────────┐
│                 3R·I·B_w                     │
│  M_crit = ─────────────────                  │
│             6I - 2R·B_t                      │
└─────────────────────────────────────────────┘
```

> **可行性条件**: 6I > 2R·B_t, 即 I > R·B_t/3. 对 BF16: I > 313×2/3 = 209. 实际模型 I ≥ 1024, 始终满足.

#### 3.9.6 近似公式

分母中 token I/O 项 `2R·B_t` 相对于权重项 `6I` 很小 (BF16 时 1252 vs 12288, 占 ~10%), 因此:

```text
┌─────────────────────────────────────────────┐
│  M_crit ≈ R·B_w / 2                         │
└─────────────────────────────────────────────┘
```

**含义**: 每个专家至少需要 **(Ridge Point × 权重字节数 / 2)** 个 token 才能进入 Compute Bound.

近似公式的修正系数:

```text
M_crit_exact / M_crit_approx = 1 / (1 - R·B_t/(3I))
```

对 BF16, I=2048: 修正系数 = 1/(1 - 626/6144) = 1.11, 即近似值偏低 ~11%.

> **关键性质**:
> - **与 H (hidden_size) 无关**: H 在 FLOPs 和 HBM bytes 中同比例出现, 完全约掉.
> - **与 I (intermediate_size) 弱相关**: 近似公式中 I 约掉, 仅在修正项中保留 (~11% 影响).
> - **与 E_L (本地专家数) 无关**: M_crit 是 per-expert 阈值, 不依赖专家数.

#### 3.9.7 全局 Token 数转换

均匀路由假设下, 每专家平均 token 数:

```text
M_avg = T_L × K / E_L = T × K / E
```

由 M_avg ≥ M_crit 得全局临界 token 数:

```text
┌─────────────────────────────────────────────┐
│  T_crit = M_crit × E / K                    │
│                                              │
│  近似: T_crit ≈ R·B_w·E / (2K)              │
└─────────────────────────────────────────────┘
```

> T_crit 与 EP 并行度 D 无关 — 仅取决于全局专家数 E、激活专家数 K 和权重精度 B_w.

#### 3.9.8 数值计算 (DeepSeek-V3-like: E=256, K=8, H=8192, I=2048)

**精确计算 (BF16, B_w=B_t=2)**:

```text
M_crit = 3 × 313 × 2048 × 2 / (6 × 2048 - 2 × 313 × 2)
       = 3,846,144 / 11,036
       = 349 tokens/expert

T_crit = 349 × 256 / 8 = 11,168 tokens (全局)
```

**不同权重精度对比**:

| 权重精度 | B_w | M_crit (精确) | M_crit (近似 R·B_w/2) | T_L_crit (D=4) | **T_crit (全局)** |
|---------|-----|--------------|----------------------|----------------|------------------|
| BF16 | 2 | **349** | 313 | 2,792 | **11,168** |
| INT8 | 1 | **175** | 157 | 1,400 | **5,600** |
| INT4 | 0.5 | **87** | 78 | 698 | **2,792** |

> INT8/INT4 权重量化时, 计算仍以 BF16 执行 (dequantization 为 VPU 操作, 与 MXU 重叠). 量化仅减少 HBM 搬运量, MXU 峰值不变.

#### 3.9.9 与实际工作负载对比

| 场景 | T | D | M_avg | vs M_crit (BF16) | AI (FLOPs/byte) | MXU 利用率 | 瓶颈 |
|------|---|---|-------|-----------------|-----------------|-----------|------|
| Decode 小 batch | 256 | 4 | 8 | **44× 不足** | 8.0 | 2.6% | 严重 HBM BW bound |
| Decode 大 batch | 2,048 | 4 | 64 | **5.5× 不足** | 64.0 | 20.4% | HBM BW bound |
| Prefill 中等 | 8,192 | 4 | 256 | **1.4× 不足** | 213 | 68.1% | HBM BW bound (接近临界) |
| **临界点** | **11,168** | 4 | **349** | **= 1.0×** | **313** | **100%** | **平衡点** |
| Prefill 大 batch | 16,384 | 4 | 512 | 1.5× 超过 | 410 | 76.3%† | Compute bound |

> † Compute bound 时 MXU 利用率 = R / AI = 313 / 410 = 76.3%, 表示 HBM 有余量, MXU 成为瓶颈.

#### 3.9.10 关键结论

1. **简洁规则**: 每专家需 **~R·B_w/2 ≈ 313 (BF16) / 157 (INT8) / 78 (INT4)** 个 token 才能 Compute Bound.

2. **Decode 永远 HBM bound**: 典型 Decode (T=64~2048) 距离 Compute Bound 差 5~44×, MXU 利用率 < 21%.

3. **权重量化是最有效手段**: INT8 将 M_crit 减半, INT4 减为 1/4 — 直接线性降低进入 Compute Bound 的 token 门槛.

4. **vs §3.6 实现公式的关系**: 实现公式 `AI ≈ 2·bt·K / (num_bt·E_L·B_w)` 包含 `num_bt` 权重重读约束. 本节给出的算法下限假设 `num_bt=1` (权重仅读一次), 是**理论最优** — 即使完美实现也至少需要 M_crit 个 token/expert.

5. **与模型维度的依赖关系**: M_crit 与 H 无关, 与 I 弱相关 (~11% 修正). 主要由硬件 Ridge Point (R) 和权重精度 (B_w) 决定.

### 3.10 达到 Memory Bound 的详细设计: FFN1/FFN2 联合计算

> Stage 3 是 kernel 耗时最大的阶段 (典型配置下占端到端 >80%). 当前实现对每个专家按 FFN1 → FFN2 顺序执行. 本节分析顺序执行的 DMA pipeline 效率, 推导 pipeline bubble 来源, 设计 FFN1/FFN2 联合计算方案, 目标是使 Stage 3 达到 HBM 带宽理论上限 (η_mem → 100%).

#### 3.10.1 Memory Bound 效率模型

**理论下限** — 不可避免的 HBM 搬运量 (每个权重恰好读一次, token 读写各一次):

```text
Bytes_stage3_min = E_L × 3 × H × I × B_w            # 全部权重
                 + 2 × bt × K × H × B_t              # token 读 + 写 (所有专家汇总)

T_stage3_ideal = Bytes_stage3_min / HBM_BW
```

> 此为 **num_bt = 1** (权重不重读) 的理论下限. num_bt > 1 时权重被重读 num_bt 次, 理论下限相应增大.

**Memory Bound 效率定义**:

```text
η_mem = T_stage3_ideal / T_stage3_actual

η_mem = 1.0 表示 DMA 引擎 100% 利用, 达到 HBM 峰值带宽.
η_mem < 1.0 的差距 = pipeline bubble 开销.
```

> Memory-bound 下 MXU 计算时间远小于 DMA 传输时间 (典型 Decode: MXU/DMA < 3%). 因此 **DMA 引擎持续满载是达到 Memory Bound 的充要条件**. 所有 bubble 分析围绕 DMA 连续性展开.

#### 3.10.2 当前实现: FFN1 → FFN2 顺序执行 Pipeline 分析

**Per-expert DMA-MXU 时间线** (per bf slice, memory-bound 稳态):

```text
DMA:  ┃W1₀ W3₀┃tok₀┃W1₁ W3₁┃tok₁┃...┃W1ₙ W3ₙ┃tokₙ┃ W2₀ ┃W2₁ ┃...┃W2ₘ ┃
MXU:  ┃       ┃    ┃mm1₀mm3₀┃    ┃...┃       ┃    ┃mm1ₙ ┃FFN2₀┃...┃FFN2ₘ┃
VPU:  ┃       ┃    ┃        ┃    ┃...┃       ┃    ┃     ┃act₀ ┃...┃     ┃
       ↑ startup                              ↑ FFN1→FFN2 transition
```

> 双缓冲确保 DMA tile N+1 的加载与 MXU tile N 的计算重叠. Memory-bound 下 MXU 总是先于 DMA 完成, DMA 引擎是唯一瓶颈.

**Pipeline Bubble 分类与量化**:

| Bubble 类型 | 描述 | 持续时间估计 | 频率 (per bt tile) |
|------------|------|------------|-------------------|
| **Expert startup** | 首个 DMA tile 无 MXU 可重叠 | T_first_tile ≈ bd1c×bfc×B_w×2 / HBM_BW | E_L 次 |
| **Expert drain** | 末尾 MXU 无后续 DMA | T_last_mxu ≈ 0 (MXU << DMA) | E_L 次, 可忽略 |
| **FFN1→FFN2 transition** | Phase 切换 (W1/W3 → W2 加载) | ~0 (W2 prefetch 覆盖) | E_L × num_bf 次 |
| **Expert boundary** | 专家间条件判断 + token 初始化 | T_cond + T_tok_init | E_L - 1 次 |
| **Token staging contention** | token DMA 占用引擎, 阻塞权重 DMA | T_tok_per_tile | 每个 weight tile |

> 当前代码在 FFN1 尾部预取 W2 首个 tile (源码注释: "prefetch W2 for FFN2"). 但 memory-bound 下 MXU 窗口极短 (<0.01 μs), 预取实际依赖 DMA pipeline 的连续性而非 MXU 空闲窗口.

**Per-tile DMA 时间** (DeepSeek-V3-like: H=8192, I=2048, B_w=2, bf=256, bd1=1024, bd2=1024):

```text
FFN1 weight tile:   2 × bd1c × bfc × B_w = 2 × 1024 × 256 × 2 = 1.0 MB  → 0.271 μs
FFN2 weight tile:   bfc × bd2c × B_w     = 256 × 1024 × 2     = 0.5 MB  → 0.135 μs
Token staging tile: btc × bd1c × B_t     = 8 × 1024 × 2       = 16 KB   → 0.004 μs

FFN1 MXU per tile:  4 × btc × bd1c × bfc / MXU_peak = 4×8×1024×256 / 1154e12 = 0.007 μs
FFN2 MXU per tile:  2 × btc × bfc × bd2c / MXU_peak = 2×8×256×1024 / 1154e12 = 0.004 μs
```

> **DMA / MXU 比**: FFN1 tile = 0.271 / 0.007 = **39×**, FFN2 tile = 0.135 / 0.004 = **34×**. MXU 在每个 tile 中仅活跃 ~3%, 剩余 97% 时间等待 DMA.

**Per-expert Bubble 量化** (M=8, Decode):

```text
B_startup         = T_first_tile                    = 0.271 μs
B_expert_boundary = T_cond + T_tok_staging_init     ≈ 0.1 + 0.034 = 0.134 μs  # 估计值
B_tok_contention  = num_bf × num_bd1 × T_tok_tile   = 8 × 8 × 0.004 = 0.256 μs
B_ffn1_ffn2_trans = 0                                                          # W2 prefetch 覆盖

B_per_expert     ≈ 0.271 + 0.134 + 0.256            = 0.661 μs
```

**全 Stage 3 效率** (E_L=64, D=4):

```text
T_ideal  = 64 × 100.7 MB / 3690 GB/s               = 1745 μs
B_total  = 64 × 0.661                               = 42.3 μs
T_actual = T_ideal + B_total                         = 1787 μs

η_mem = 1745 / 1787 = 97.6%
```

**Bubble 分项占比**:

```text
┌──────────────────────────────────────────────────────────┐
│  Expert startup       17.3 μs  ██████████████  41%       │
│  Token staging        16.4 μs  █████████████   39%       │
│  Expert boundary       8.4 μs  ██████          20%       │
│  FFN1→FFN2 trans       0.0 μs                   0%       │
│  ─────────────────────────────────────────────           │
│  Total bubble         42.3 μs  (2.4% of T_actual)       │
└──────────────────────────────────────────────────────────┘
```

> **关键发现**: 顺序执行已达 ~97.6% HBM BW 效率. 但此为理想估算 — 实际实现中编译器调度、DMA 对齐约束、VMEM bank conflict 可能进一步降低效率. 消融实验 (§3.10.7) 将标定实际 η_mem.

#### 3.10.3 FFN1/FFN2 联合计算方案设计

**核心思路**: 消除顺序执行中 DMA 流断裂的三个位置 — expert startup、expert boundary、token staging contention.

##### Strategy A: Cross-bf Pipeline (bf 级 FFN1/FFN2 交错)

当前 bf 循环内, FFN1 和 FFN2 按 phase 顺序执行. Cross-bf pipeline 将 FFN2(bf=i) 与 FFN1(bf=i+1) 的权重加载交错在同一 DMA stream 中:

```text
当前 (顺序, per expert, per bf):
  bf=0: [FFN1₀ ─ all bd1 ─][FFN2₀ ─ all bd2 ─]
  bf=1: [FFN1₁ ─ all bd1 ─][FFN2₁ ─ all bd2 ─]
  ...

Cross-bf pipeline:
  bf=0:     [FFN1₀ ─ all bd1 ─]──────────────────────────┐
  bf=0→1:   ← FFN2₀ DMA ∥ FFN1₁ DMA 交错 →              │
            [W2₀₀ W1₁₀ W3₁₀ W2₀₁ W1₁₁ W3₁₁ ...]       │
            MXU: [FFN2₀ tile₀][FFN1₁ tile₀][FFN2₀ tile₁]│
  bf=1→2:   ← FFN2₁ DMA ∥ FFN1₂ DMA 交错 →              │
  ...                                                     │
  bf=N-1:   [FFN2_{N-1} ─ all bd2 ─]─────────────────────┘
```

**约束**: FFN2(bf=i) 依赖 FFN1(bf=i) 的全部 bd1 累积完成. 不同 bf slice 的 FFN1 和 FFN2 **互相独立**.

**VMEM 额外需求**:

```text
Δ_VMEM = bfc × bd2c × B_w              # W2 双缓冲 (与 W1/W3 同时驻留)
       + btc × bfc × 4 × 2             # 两个 bf slice 的中间激活 (gate_acc + up_acc, F32)
       = 256 × 1024 × 2 + 8 × 256 × 4 × 2 × 2
       = 512 KB + 32 KB ≈ 545 KB       # 对 64 MiB VMEM 可忽略 (<1%)
```

**收益**: 消除 FFN1→FFN2 transition bubble. 但如 §3.10.2 分析, 该 bubble 已被 W2 prefetch 基本消除 (~0 μs), 故 Strategy A **净收益可忽略**.

##### Strategy B: Cross-expert Pipeline (专家级 FFN2/FFN1 交错)

在 expert e 的 FFN2 尾部, 开始加载 expert e+1 的首个权重 tile 和 token 数据:

```text
当前 (顺序):
  Expert e:    [...FFN1(e)...][...FFN2(e)...]
                              ↕ expert boundary bubble
  Expert e+1:                                [...FFN1(e+1)...][...FFN2(e+1)...]

Cross-expert pipeline:
  Expert e:    [...FFN1(e)...][...FFN2(e)...]
                                            ↘ DMA 连续
  Expert e+1:                           [tok(e+1)][W1(e+1)₀ W3(e+1)₀]...
               MXU:                     [FFN2(e) last tiles] [FFN1(e+1)₀]...
                                         ↑ expert e+1 startup 被 expert e FFN2 隐藏
```

**实现要点**:
- `expert_ffn()` 接受 next expert 的元数据 (token 位置, expert size)
- 在 `run_down_slices()` 最后 1-2 个 bd2 tile 时, 启动 next expert 的:
  - Token staging DMA (从 a2a_s_hbm 预加载到 VMEM)
  - W1 首个 tile DMA (双缓冲的第二个 buffer)

**收益**: 消除 expert startup bubble (17.3 μs) 和 expert boundary bubble (8.4 μs), 合计 **~25.7 μs**.

##### Strategy C: Token 全预加载 (消除 token staging contention)

在 Stage 3 循环开始前, 一次性将所有专家的 token 从 `a2a_s_hbm` batch DMA 到 VMEM:

```text
当前:
  权重 DMA stream: [W₀][tok₀][W₁][tok₁][W₂][tok₂]...
                         ↑      ↑      ↑ token DMA 碎片穿插, 阻塞权重 DMA

Token 全预加载:
  预加载阶段: [tok_all ──────────── batch DMA ──]
  权重 DMA:   [W₀][W₁][W₂][W₃][W₄]...  ← 连续, 无 token 碎片中断
```

**VMEM 需求** (H=8192, K=8, B_t=2):

| D | E_L | M_avg (Decode) | VMEM_tok | VMEM 占比 |
|---|-----|----------------|----------|----------|
| 4 | 64 | 8 | 8 MB | 12.5% |
| 16 | 16 | 8 | 2 MB | 3.1% |
| 64 | 4 | 8 | 0.5 MB | 0.8% |
| 128 | 2 | 8 | 0.25 MB | 0.4% |

> Decode 场景 VMEM 需求适中 (D=4 时 12.5%), D≥16 时 < 4%. Prefill 场景 (M_avg=16) 需求翻倍, 可用分批预加载 (方案 C2).

**实现方案**:

```text
方案 C1 — 全量预加载 (Decode):
  Stage 3 开始前: batch DMA 所有 expert token → VMEM
  Stage 3 内部: 跳过 per-bd1 token staging, 直接从 VMEM 读取

方案 C2 — 分批预加载 (Prefill):
  每次预加载 N_batch 个专家的 token (N_batch = VMEM_avail / (M_avg × H × B_t))
  处理完一批后, 预加载下一批

方案 C3 — 与 §2.8 VMEM 驻留联动 (最激进):
  Stage 2 A2A Dispatch 直接将 token 写入 VMEM (不经 HBM)
  Stage 3 直接消费 VMEM 中的 token, 完全消除 token HBM round-trip
```

**收益**: 消除 token staging contention (16.4 μs), 且将碎片化小 DMA 改为单次 batch DMA.

##### Strategy B+C: 联合方案 (Cross-expert Pipeline + Token 预加载)

组合 Strategy B 和 C, **同时消除**所有三类主要 bubble:

```text
T_fused = T_ideal + B_residual

B_residual ≈ B_first_expert_startup + B_last_expert_drain
           = 0.271 + ~0 = 0.271 μs                    # 仅第一个/最后一个专家

η_mem_fused = T_ideal / (T_ideal + 0.271) ≈ 99.98%
```

#### 3.10.4 VMEM 预算分析

> 64 MiB VMEM 总量. 以下分析 Strategy B+C 的 VMEM 布局.

**当前顺序执行 VMEM 布局** (per expert 活跃):

| 缓冲区 | 大小 (DeepSeek-V3) | 说明 |
|--------|-------------------|------|
| W1/W3 双缓冲 | 2 × 2 × bd1c × bfc × B_w = 2.0 MB | 2 buffer × (W1+W3) |
| W2 双缓冲 | 2 × bfc × bd2c × B_w = 1.0 MB | FFN2 权重 |
| gate_acc + up_acc | 2 × btc × bfc × 4 = 16 KB | FFN1 中间激活 (F32) |
| act buffer | btc × bfc × B_t = 4 KB | activation 输出 |
| result acc | btc × bd2c × 4 = 32 KB | FFN2 累加器 (F32) |
| token staging | 2 × btc × bd1c × B_t = 32 KB | 双缓冲 token tile |
| **合计** | **~3.1 MB** | **VMEM 占比 4.8%** |

**Strategy B+C 额外 VMEM** (Decode, D=4):

| 额外缓冲区 | 大小 | 说明 |
|-----------|------|------|
| Token 预加载 (C1) | 8 MB | E_L × M_avg × H × B_t |
| Cross-expert prefetch (B) | ~1 MB | next expert W1/W3 首 tile |
| **额外合计** | **~9 MB** | **VMEM 占比 14.1%** |

```text
Strategy B+C 总 VMEM = 3.1 + 9.0 = 12.1 MB  (18.9%)  ✓ 可行

剩余 VMEM (51.9 MB) 供 SE 计算、A2A buffer、编译器临时使用.
```

#### 3.10.5 各方案收益对比

##### Decode 场景 (T=2048, D=4, E_L=64, M_avg=8, BF16)

| 指标 | 顺序 FFN1→FFN2 | Strategy B (跨专家) | Strategy C (token 预加载) | **B+C (联合)** |
|------|---------------|-------------------|------------------------|---------------|
| Expert startup bubble | 17.3 μs | **~0 μs** | 17.3 μs | **~0 μs** |
| Expert boundary bubble | 8.4 μs | **~2 μs** | 8.4 μs | **~2 μs** |
| Token staging contention | 16.4 μs | 16.4 μs | **~0 μs** | **~0 μs** |
| FFN1→FFN2 transition | ~0 μs | ~0 μs | ~0 μs | ~0 μs |
| **Total bubble** | **42.3 μs** | **18.4 μs** | **25.7 μs** | **~2 μs** |
| **η_mem** | **97.6%** | **99.0%** | **98.5%** | **~99.9%** |
| VMEM 额外需求 | 0 | 1 MB | 8 MB | 9 MB |
| 实现复杂度 | 基线 | 中 | 低 | 中 |

##### 各 EP 规模 Bubble 对比 (T=2048, BF16)

| D | E_L | T_ideal (μs) | B_seq (μs) | η_seq | B_fused (μs) | η_fused | **Δ (μs)** |
|---|-----|-------------|-----------|-------|-------------|---------|-----------|
| 4 | 64 | 1745 | 42.3 | 97.6% | ~2 | 99.9% | **40.3** |
| 16 | 16 | 436 | 10.6 | 97.6% | ~1 | 99.8% | **9.6** |
| 64 | 4 | 109 | 2.6 | 97.7% | ~0.5 | 99.5% | **2.1** |
| 128 | 2 | 54.5 | 1.3 | 97.7% | ~0.3 | 99.5% | **1.0** |

> **关键发现**:
>
> 1. **顺序执行的 η_mem 已达 ~97.6%** — 理论分析显示优化空间有限 (~42 μs / 1745 μs for D=4). 但此为理想分析, 实际编译器/硬件开销可能使 η_mem 显著低于理论值.
> 2. **Expert startup 是最大 bubble 来源 (41%)** — Cross-expert pipeline (Strategy B) 可有效消除.
> 3. **Token staging contention 是第二大来源 (39%)** — Token 预加载 (Strategy C) 以少量 VMEM (<13%) 换取消除.
> 4. **D 越大, 绝对收益越小** — D=64 时总 bubble 仅 2.6 μs, 优化必要性降低.
> 5. **消融实验的核心价值**: 标定实际 η_mem 与理论值的差距, 判断优化优先级.

##### Prefill 场景 (T=8192, bt=128, D=4, num_bt=4, BF16)

| 指标 | 顺序 FFN1→FFN2 | **B+C (联合)** |
|------|---------------|---------------|
| 权重重读次数 | num_bt = 4 | num_bt = 4 (不变) |
| Per bt-tile bubble | 42.3 μs | ~2 μs |
| **Total bubble** | **169 μs** | **~8 μs** |
| T_ideal | 6990 μs | 6990 μs |
| **η_mem** | **97.6%** | **~99.9%** |
| **绝对收益** | — | **~161 μs** |

> Prefill 因 num_bt > 1 (权重重读), bubble 被放大 num_bt 倍. 联合计算在 Prefill 的**绝对收益更大** (161 μs vs 40 μs), 但 **权重重读本身不被联合计算解决** — 降低 num_bt 需要增大 bt (需要更多 VMEM) 或减少 E_L (增大 EP degree).

#### 3.10.6 达到 Memory Bound 的瓶颈层次

```text
优先级排序 (对 η_mem 的影响从大到小):

 ┌─────────────────────────────────────────────────────────────┐
 │ P0: 验证实际 η_mem (消融实验 §3.10.7 Exp S3-A)              │
 │     → 若实际 η_mem ≪ 97.6%, 存在未建模开销, 需先诊断         │
 │     → 若实际 η_mem ≈ 97.6%, 理论分析成立, 按下列优先级优化    │
 ├─────────────────────────────────────────────────────────────┤
 │ P1: Token 预加载 (Strategy C)                               │
 │     收益: 消除 39% bubble; 实现复杂度: 低; VMEM 开销: 适中     │
 ├─────────────────────────────────────────────────────────────┤
 │ P2: Cross-expert pipeline (Strategy B)                      │
 │     收益: 消除 61% bubble; 实现复杂度: 中; VMEM 开销: 低       │
 ├─────────────────────────────────────────────────────────────┤
 │ P3: 降低 num_bt (Prefill 权重重读优化)                       │
 │     正交于联合计算; 需增大 bt 或 VMEM 容量                    │
 └─────────────────────────────────────────────────────────────┘
```

#### 3.10.7 消融实验设计

##### Exp S3-A: Pipeline 效率基线

```python
# 完整 Stage 3, 测量实际 HBM 带宽利用率
result = fused_ep_moe(
    tokens, w1, w2, w3, topk_weights, topk_ids, top_k=8,
    ep_size=4,
    # 隔离 Stage 3: 禁用 A2A + SE + metadata
    disable_a2a=True, disable_shared_expert=True,
    disable_all_reduce_metadata=True, disable_sync_barrier=True,
)

# 计算 η_mem
T_actual = measure_latency(result)
Bytes_total = E_L * 3 * H * I * B_w + 2 * bt * K * H * B_t
η_mem = Bytes_total / (T_actual * HBM_BW)
```

验证目标: η_mem 是否接近理论值 97.6%. 若显著低于 (如 <90%), 存在未建模开销.

##### Exp S3-B: 纯 DMA Pipeline 延迟

```python
# 禁用全部 MXU 计算, 保留 DMA pipeline
result = fused_ep_moe(
    ...,
    disable_dynamic_ffn1=True, disable_dynamic_ffn2=True,
    disable_a2a=True, disable_shared_expert=True,
    disable_all_reduce_metadata=True, disable_sync_barrier=True,
)

T_dma_only = measure_latency(result)
# T_dma_only ≈ T_ideal + B_total (纯 DMA 时间 + 所有 bubble)
```

验证目标: T_dma_only 是否接近 Bytes_total / HBM_BW. 差异 = DMA pipeline bubble 总量.

##### Exp S3-C: Expert Boundary 开销标定

```python
# 对比: 单专家 (所有 token 集中) vs 多专家
for e_l in [1, 4, 16, 64]:
    # 固定总权重量: 调整 E 使 E_L = e_l, M = bt × K / e_l
    result = fused_ep_moe(
        ..., num_experts=e_l * D,
        disable_a2a=True, disable_shared_expert=True,
        disable_all_reduce_metadata=True, disable_sync_barrier=True,
    )

T_per_expert_boundary = (T_multi - T_single_equiv) / (E_L - 1)
```

验证目标: B_expert_boundary 估计值 (0.134 μs) 的准确性.

##### Exp S3-D: Token Staging Contention 标定

```python
# 对比: 有 token staging vs 禁用 token staging
result_baseline = fused_ep_moe(
    ...,
    disable_a2a=True, disable_shared_expert=True,
    disable_all_reduce_metadata=True, disable_sync_barrier=True,
)
result_no_tok = fused_ep_moe(
    ...,
    disable_a2a_s_tile_read=True,   # 禁用 token staging DMA
    disable_a2a=True, disable_shared_expert=True,
    disable_all_reduce_metadata=True, disable_sync_barrier=True,
)

Δ_tok = T_baseline - T_no_tok
# Δ_tok ≈ B_tok_contention (token staging 对权重 DMA 的干扰)
```

验证目标: B_tok_contention 估计值 (16.4 μs for D=4, E_L=64) 的准确性.

##### Exp S3-E: FFN1/FFN2 联合计算分解实验 (核心实验)

```python
for ep_size in [4, 16, 64]:
    for scenario in ["decode", "prefill"]:
        T = 2048 if scenario == "decode" else 8192

        # E1: Baseline — 当前顺序执行
        T_seq = run(disable_a2a=True, disable_shared_expert=True,
                     disable_all_reduce_metadata=True, disable_sync_barrier=True)

        # E2: 纯 DMA (无 MXU) — 量化 DMA pipeline 效率
        T_dma = run(disable_dynamic_ffn1=True, disable_dynamic_ffn2=True,
                     disable_a2a=True, disable_shared_expert=True,
                     disable_all_reduce_metadata=True, disable_sync_barrier=True)

        # E3: 禁 token staging — 消除 token contention
        T_no_tok = run(disable_a2a_s_tile_read=True,
                       disable_a2a=True, disable_shared_expert=True,
                       disable_all_reduce_metadata=True, disable_sync_barrier=True)

        # E4: 禁 token staging + 禁 MXU — 纯权重 DMA (无 token 干扰)
        T_pure_w = run(disable_dynamic_ffn1=True, disable_dynamic_ffn2=True,
                       disable_a2a_s_tile_read=True,
                       disable_a2a=True, disable_shared_expert=True,
                       disable_all_reduce_metadata=True, disable_sync_barrier=True)
```

**分解验证框架**:

```text
T_seq = T_ideal + B_expert + B_tok + B_compute

其中:
  B_expert    = T_pure_w - T_ideal          # expert startup + boundary 开销
  B_tok       = T_dma - T_pure_w            # token staging contention
  B_compute   = T_seq - T_dma              # MXU 干扰 (memory-bound 下 ≈ 0)

完备性验证: B_expert + B_tok + B_compute ≈ T_seq - T_ideal
  → 若成立: bubble 分析无遗漏
  → 若失败: 存在未建模开销 (DMA 对齐, VMEM conflict, 编译器 overhead 等)
```

**扫描矩阵与预期值** (Decode, T=2048, BF16):

| D | E_L | T_ideal (μs) | B_expert 预期 (μs) | B_tok 预期 (μs) | η_seq 预期 | 验证重点 |
|---|-----|-------------|-------------------|----------------|----------|---------|
| 4 | 64 | 1745 | 25.7 | 16.4 | 97.6% | 最多 expert → B_expert 最大 |
| 16 | 16 | 436 | 6.3 | 4.1 | 97.6% | 中等规模 |
| 64 | 4 | 109 | 1.5 | 1.0 | 97.7% | 最少 expert → B_expert 最小 |

**扫描矩阵与预期值** (Prefill, T=8192, bt=128, BF16):

| D | E_L | num_bt | T_ideal (μs) | B_total 预期 (μs) | η_seq 预期 |
|---|-----|--------|-------------|------------------|----------|
| 4 | 64 | 4 | 6990 | 169 | 97.6% |
| 16 | 16 | 1 | 436 | 10.6 | 97.6% |
| 64 | 4 | 1 | 109 | 2.6 | 97.7% |

##### Exp S3-F: 联合计算实现验证 (需 kernel 修改)

```python
# 需要 kernel 新增参数 ffn_pipeline_mode
for mode in ["sequential", "cross_expert", "preload_tokens", "fused"]:
    result = fused_ep_moe(
        ..., ep_size=4,
        ffn_pipeline_mode=mode,    # 新参数
        disable_a2a=True, disable_shared_expert=True,
        disable_all_reduce_metadata=True, disable_sync_barrier=True,
    )
```

| 模式 | 消除的 bubble | 预期收益 (D=4 Decode) | kernel 改动 |
|------|-------------|--------------------|-----------:|
| `sequential` (baseline) | — | 0 μs | 无 |
| `cross_expert` (Strategy B) | expert startup + boundary | ~25.7 μs | 中 |
| `preload_tokens` (Strategy C) | token staging contention | ~16.4 μs | 低 |
| `fused` (Strategy B+C) | 全部 | ~42 μs | 中 |

> **实验优先级**: Exp S3-A (基线 η_mem) 必须先执行. 若实际 η_mem 显著低于理论值 (~97.6%), 优先诊断根因 (Exp S3-B/C/D), 再考虑联合计算实现 (Exp S3-F).

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

### 4.8 Gather 调度策略消融: Per-Expert Pipeline vs Deferred Batch

#### 4.8.1 实验动机

当前 kernel 在每个 expert FFN 完成后立即启动 `start_a2a_gather`, 利用 MXU (FFN) 与 DMA/ICI (Gather) 的资源独立性形成 compute-communication pipeline. 替代方案: 等所有 expert 计算完毕后, 将结果按目标设备合并为一次批量 ICI 传输 (与 §2.9 per-device batch DMA 分析对称).

**核心权衡**: pipeline 重叠收益 vs `E_L × (D-1)` 次小 DMA 的 launch overhead 累积.

#### 4.8.2 策略定义

**Strategy A: Per-Expert Eager Gather (当前实现)**

```text
for local_e_id in 0..E_L-1:
    expert_ffn(local_e_id)                    # MXU compute (T_ffn_e)
    start_a2a_gather(local_e_id):             # 异步 DMA/ICI (T_gather_e)
        for target_device in 0..D-1:
            if has_tokens: async_remote_copy(per-device batch)
    # gather(e) 与 ffn(e+1) 并行 (MXU vs DMA/ICI 独立资源)
wait_a2a_gather_recv_all()
```

- Per-expert ICI ops: `min(D-1, M_avg)` (仅有 token 的远程设备)
- Per-op 传输量: `max(1, M_avg/D) × H × B_t` bytes
- 总 ICI ops: `E_L × min(D-1, M_avg)` (上界)

**Strategy B: Deferred Per-Device Batched Gather**

```text
for local_e_id in 0..E_L-1:
    expert_ffn(local_e_id)                    # 结果留在 a2a_s_acc_x2_hbm

# 所有 expert 完成后, 按目标设备合并发送
for target_device in 0..D-1:
    batch = pack_results_for(target_device)   # 合并 E_L 个 expert 的数据
    async_remote_copy(batch → target.a2a_g)
wait_all()
```

- 总 ICI ops: `D-1` (每目标设备一次)
- Per-op 传输量: `(bt × K / D) × H × B_t` bytes (E_L 个 expert 合并)

> **Op 次数对比**: A = `E_L × min(D-1, M_avg)`, B = `min(D-1, bt×K)`. 每次传输量: B 是 A 的 E_L 倍. 总 ICI 数据量相同 = `bt × K × H × B_t × (D-1)/D`.

#### 4.8.3 延迟模型

**Per-expert FFN 延迟** (HBM BW bound, 对给定模型恒定):

```text
T_ffn_e = 3 × H × I × B_w / HBM_BW
        = 96 MB / 3690 GB/s = 26.0 μs          # DeepSeek-V3-like
```

**Strategy A — Pipeline Model**:

MXU (FFN chain) 与 DMA/ICI (Gather chain) 是独立资源, 可真正并行. Gather(e) 在 FFN(e) 完成后启动, 与 FFN(e+1) 重叠:

```text
Active_remotes_e = min(D-1, M_avg)              # 有数据的远程设备数
per_op_data_e   = max(1, M_avg / D) × H × B_t   # 每次 ICI 传输量

T_gather_e = Active_remotes_e × (T_ici_launch + per_op_data_e / ICI_BW)

# 两条串行链 + 依赖关系 (gather(e) 须等 ffn(e) 完成):
T_total_A = E_L × T_ffn_e + T_gather_e              if T_gather_e ≤ T_ffn_e   (pipeline 有效)
          = T_ffn_e + E_L × T_gather_e               if T_gather_e > T_ffn_e   (pipeline 失效)

T_exposed_A = T_gather_e                             if T_gather_e ≤ T_ffn_e   (仅 last-expert drain)
            = E_L × (T_gather_e - T_ffn_e) + T_ffn_e if T_gather_e > T_ffn_e
```

**Strategy B — Sequential Model**:

```text
batch_data     = (bt × K / D) × H × B_t             # 每目标设备合并后数据量
Active_targets = min(D-1, bt × K)                    # 有数据的目标设备数

T_gather_B = Active_targets × (T_ici_launch + batch_data / ICI_BW)

T_total_B  = E_L × T_ffn_e + T_gather_B              # 无重叠, 全部暴露
T_exposed_B = T_gather_B
```

#### 4.8.4 Pipeline 失效临界条件

Pipeline 有效要求 `T_gather_e ≤ T_ffn_e`. 对 DeepSeek-V3 (T_ffn_e = 26 μs):

当 `M_avg ≤ D` (每设备 ≤ 1 token/expert, 典型 Decode):

```text
M_avg × (T_ici_launch + H × B_t / ICI_BW) ≤ T_ffn_e
M_avg × (500 + 82) ns ≤ 26,000 ns
M_avg ≤ 45                                  # Pipeline 有效条件
```

当 `M_avg > D` (所有 D-1 远程设备均活跃, 典型 Prefill):

```text
(D-1) × (T_ici_launch + M_avg/D × 82 ns) ≤ 26,000 ns
```

| D | 临界 M_avg | 对应 bt 临界 | 含义 |
|---|-----------|------------|------|
| 4 | 399 | bt ≈ 3192 | 所有实际 bt 均 pipeline 有效 |
| 16 | 241 | bt ≈ 482 | 所有实际 bt 均 pipeline 有效 |
| 64 | 45 (M<D 区间) | **bt ≈ 22** | Prefill (bt≥32) 时 pipeline 失效 |
| 128 | 45 (M<D 区间) | **bt ≈ 11** | Prefill (bt≥16) 时 pipeline 失效 |

> **根本原因**: D ≥ 64 时, D-1 次串行 ICI launch 的累积 `(D-1) × T_ici_launch ≥ 63 × 500 ns = 31.5 μs` 已超过 T_ffn_e (26 μs), 即使每次传输数据量极小, 纯 launch overhead 就足以打破 pipeline.

#### 4.8.5 数值分析

**配置**: DeepSeek-V3-like (E=256, K=8, H=8192, I=2048, B_w=B_t=2, T_ici_launch=500 ns)

##### Decode 场景

| D | bt | E_L | M_avg | T_gather_e (μs) | T_g/T_f | T_exposed_A (μs) | T_exposed_B (μs) | **Winner** |
|---|-----|-----|-------|-----------------|---------|-----------------|-----------------|------------|
| 4 | 64 | 64 | 8 | 2.0 | 0.08 | 2.0 | 31.5 | **A (15.8×)** |
| 16 | 16 | 16 | 8 | 8.1 | 0.31 | 8.1 | 17.1 | **A (2.1×)** |
| 64 | 4 | 4 | 8 | 4.6 | 0.18 | 4.6 | 16.8 | **A (3.7×)** |
| 128 | 4 | 2 | 16 | 9.3 | 0.36 | 9.3 | 17.3 | **A (1.9×)** |
| 256 | 2 | 1 | 16 | 9.3 | 0.36 | 9.3 | 9.3 | Tie |

> **计算说明** (D=4 为例):
>
> - Active remotes = min(3, 8) = 3; per-op = 2 tokens × 16 KB = 32 KB
> - T_gather_e = 3 × (500 + 32768/200e9) = 3 × 664 ns = 2.0 μs
> - Pipeline 有效 (2.0 < 26), T_exposed_A = 2.0 μs (仅 drain)
> - Strategy B: batch = 64×8/4 × 16384 = 2 MB, T_B = 3 × 11.0 μs = 33.0 μs
>
> Decode 下 **Strategy A 在所有配置中胜出或持平**. T_g/T_f < 0.4, pipeline 完全隐藏 gather 通信.

##### Prefill 场景 (bt=128)

| D | bt | E_L | M_avg | T_gather_e (μs) | T_g/T_f | T_exposed_A (μs) | T_exposed_B (μs) | **Winner** |
|---|-----|-----|-------|-----------------|---------|-----------------|-----------------|------------|
| 4 | 128 | 64 | 16 | 3.5 | 0.13 | 3.5 | 62.9 | **A (18.0×)** |
| 16 | 128 | 16 | 64 | 12.4 | 0.48 | 12.4 | 82.5 | **A (6.7×)** |
| 64 | 128 | 4 | 256 | 52.2 | **2.01** | **130.8** | **114.1** | **B (1.15×)** |
| 128 | 128 | 2 | 512 | 105.2 | **4.05** | **184.4** | **144.8** | **B (1.27×)** |

> **计算说明** (D=64, pipeline 失效):
>
> - M_avg=256, Active=63 (所有远程设备), per-op = 4 tokens × 16 KB = 64 KB
> - T_gather_e = 63 × (500 + 65536/200e9) = 63 × 828 ns = 52.2 μs > T_ffn_e (26 μs)
> - Pipeline 失效: T_total_A = 26 + 4 × 52.2 = 234.8 μs, T_exposed_A = 130.8 μs
> - Strategy B: batch = 128×8/64 × 16384 = 256 KB, T_B = 63 × 1811 ns = 114.1 μs

##### 决策矩阵

```text
                    D ≤ 16                    D ≥ 64
              ┌──────────────────┬──────────────────┐
  Decode      │   A 胜 (2-18×)    │   A 胜 (2-4×)    │
  (bt 小)     │   pipeline 有效    │   M_avg 小 → ok  │
              ├──────────────────┼──────────────────┤
  Prefill     │   A 胜 (7-18×)    │   B 胜 (15-27%)  │
  (bt=128)    │   E_L ≥ 16 深     │   pipeline 失效   │
              └──────────────────┴──────────────────┘
```

#### 4.8.6 扩展分析: Local Pre-Combine (Stage 4+6 融合)

> 评估将 Stage 6 加权求和提前到 expert 设备执行, 减少 gather 数据量的可行性.

思路: 每个 expert 完成 FFN 后, 在本地计算 `topk_weight × result`, 按 token 累加为 partial sum. Gather 发送 partial sum (每 token 一个 H 向量, F32) 而非 K 个 expert 输出 (每 token K 个 H 向量, BF16).

**ICI 数据量对比** (per bt tile):

| D | Current: `K × H × B_t` | Pre-Combine (F32): `D_active × H × 4` | **变化** |
|---|------------------------|---------------------------------------|---------|
| 4 | 16H bytes | 14.4H bytes | 0.9× (微减) |
| 16 | 16H bytes | 25.6H bytes | **1.6× (增大)** |
| 64 | 16H bytes | 30.8H bytes | **1.9× (增大)** |

> `D_active = D × (1 - (1 - 1/D)^K)`: 每 token 平均有 partial sum 要发送的设备数. D→∞ 时 D_active→K, 此时 pre-combine 的 F32 数据量 = K×H×4, 是 current BF16 的 **2×**. F32 精度要求完全抵消 K→1 的合并收益. **不推荐**, 除非接受 BF16 精度损失且 D ≤ 4.

#### 4.8.7 实验方案

```python
# Exp G1: Baseline — per-expert eager gather (当前实现)
# Exp G2: Deferred batched gather (需 kernel 修改)
#   修改: 移除 expert 循环内的 start_a2a_gather,
#         在所有 expert 完成后执行 batched_a2a_gather_all()
for D in [4, 16, 64, 128]:
    for scenario, T in [("decode", 512), ("prefill", 8192)]:
        for mode in ["eager", "deferred_batch"]:
            result = fused_ep_moe(
                mesh, tokens, w1, w2, w3,
                topk_weights, topk_ids, top_k=8,
                ep_size=D,
                gather_mode=mode,
            )
            profile(result, tag=f"{mode}_D{D}_{scenario}")

# Exp G3: Hybrid — 运行时自动选择策略
# 切换条件: min(D-1, M_avg) × T_ici_launch > T_ffn_e 时使用 deferred batch
fused_ep_moe(..., gather_mode="auto")
```

| 实验 | 扫描变量 | 核心指标 | 预期结果 |
|------|---------|---------|---------|
| G1 vs G2 Decode | D ∈ {4,16,64,128} | gather 阶段延迟 | G1 (eager) 胜, 所有 D |
| G1 vs G2 Prefill | D ∈ {4,16,64,128} | gather 阶段延迟 | D≤16: G1 胜; D≥64: G2 胜 15-27% |
| G3 Hybrid | D × bt | 端到端延迟 | ≤ min(G1, G2) |
| ICI 效率 | D | 有效 ICI 带宽 (GB/s) | Batched 利用率显著高于 per-expert |

> **Implementation Note**: Strategy B 需要一个 reorg 步骤 — 将 expert-major 布局的 `a2a_s_acc` 按 target-device 重排为连续 buffer. 该 reorg 为本地 HBM shuffle: `bt × K × H × B_t / HBM_BW` ≈ 4.6 μs (D=64, bt=128), 需计入 Strategy B 总延迟.

#### 4.8.8 关键结论

1. **Decode (所有 D)**: Strategy A (当前 pipeline) 最优. T_gather_e / T_ffn_e < 0.4, gather 被 FFN compute 完全隐藏, 仅暴露 last-expert drain (2-9 μs).

2. **Prefill + D ≤ 16**: Strategy A 仍最优. E_L ≥ 16 提供充足 pipeline 深度, 且每次 ICI 传输量适中 (32-64 KB, 有效带宽 55-110 GB/s).

3. **Prefill + D ≥ 64**: Strategy B (deferred batch) 可降低 gather 开销 15-27%. 根本原因: D-1 ≥ 63 次串行 ICI launch 的累积开销 (≥ 31.5 μs) 超过 T_ffn_e (26 μs), 导致 pipeline 失效; cross-expert 合并将 ICI ops 从 E_L×(D-1) 降至 D-1, 每次传输量增大 E_L 倍, ICI 带宽利用率显著提升.

4. **Local Pre-Combine 不可行**: F32 精度需求 (2× per element) 完全抵消 K→1 的合并收益, D ≥ 16 时反而增加 ICI 数据量.

5. **Hybrid 推荐**: 运行时根据 `min(D-1, M_avg) × T_ici_launch` vs `T_ffn_e` 自动选择 gather 模式 — Decode 走 pipeline, Prefill + 大 EP 走 deferred batch.


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

> **瓶颈特征**: 严重 **HBM bandwidth bound** (AI ≈ 1, 远低于 ridge point 313). 但该阶段数据量较小 (相比 Stage 3), 绝对延迟不高. 主要延迟来自 **小 DMA 串行化** — 逐 token 逐 expert 的 gather 是 `fori_loop` 串行发起.

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
| 1. Metadata | `2T_L·H·E` (Gate) + VPU | `H·E·4 + T_L·H·B_t·(1+K)` | `T_L/2` (Gate) | HBM BW (Gate) + ICI 延迟 (AR) | MXU+VPU+DMA |
| 2. Scatter | 0 (纯 DMA) | `btH·B_t` (read) | 0 | ICI BW / 串行 DMA | DMA+ICI |
| 3. Expert FFN | `6btKHI` | `E_L·3HI·B_w` + `2btKH·B_t` | `2btK/(E_L·B_w)` | **HBM BW** | MXU |
| 4. Gather | 0 (纯 DMA) | `btKH·B_t` (transfer) | 0 | ICI BW / 串行 DMA | DMA+ICI |
| 5. Shared Expert | `6btH·SE_I` | `3H·SE_I·B_w + ...` | 中等 | HBM BW | MXU |
| 6. Output Acc | `2btKH` | `btH·B_t·(K+1)` | `K/(K+1)` | HBM BW (小 DMA) | VPU |

### 7.3 关键观察

1. **Stage 3 (Expert FFN) 主导**: 占总计算量 >95%, 占总 HBM 搬运 >80%. 权重读取是主要瓶颈. 顺序 FFN1→FFN2 的 DMA pipeline 理论效率 ~97.6%, 联合计算 (§3.10) 可将 η_mem 提升至 ~99.9%.

2. **AI 随 `bt × K / E_L` 增长**: 增大 `bt` 或 `K` 提高计算密度; 减少 `E_L` (通过增大 `D`) 同样有效, 但会增加 ICI 通信.

3. **Pipeline 重叠是关键**: SE 与 A2A 的重叠效率直接影响端到端延迟. `bse` 的选择决定了 SE slice 粒度, 过大导致交错不充分, 过小导致 DMA 启动开销增大.

4. **Decode vs Prefill 差异显著**:

   - Decode: `bt` 小 → AI 低 → 严格 HBM BW bound, 但 A2A 数据量小
   - Prefill: `bt` 大 → AI 提高但仍 BW bound, A2A 数据量大可能成为 ICI 瓶颈

5. **Stage 1 在大 EP 下成为瓶颈**: Stage 3 延迟随 `1/D` 下降 (更少的本地专家), 但 Stage 1 的 AllReduce 延迟随 `log₂(D)` 增长. 在 EP ≥ 128 时, Stage 1 占 MoE 端到端延迟可达 30%–76% (详见 §1.7). Gate GEMM (~2–7 μs, W_gate 读取主导) + Permute 重排 (~0.1–41 μs, 与 T_L 成正比) + 单次 AllReduce 通信 (~6–16 μs) 三项累计, 使 Stage 1 从可忽略的开销变为首要优化目标.

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

T_compute = 51.5e12 / 1154e12 = 44.6 μs
T_hbm     = 6.44e9 / 3690e9 = 1745 μs
T_stage3  ≈ 1745 μs (HBM BW bound, MXU 利用率 ≈ 2.6%)
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

T_compute_total = 412e12 / 1154e12 = 357 μs
T_hbm_total     = 25.8e9 / 3690e9 = 6990 μs
T_stage3 ≈ 6990 μs (HBM BW bound, MXU 利用率 ≈ 5.1%)
```

> **结论**: Decode 和 Prefill 均严格 HBM BW bound. MXU 利用率极低 (< 6%), 主要瓶颈是逐专家权重读取. 优化方向: 增大 `bt×K/E_L` (通过 EP degree, batch size, 或权重量化减少 B_w).

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
