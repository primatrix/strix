---
title: "RFC-0026: Fused EP-MoE Kernel 分阶段性能分析"
status: draft
author: xl
date: 2026-05-05
reviewers: []
---

# RFC-0026: Fused EP-MoE Kernel 分阶段性能分析

> 基于 `kernels/_fused_moe_impl.py`, 对 Fused EP-MoE kernel 在 TPU v7x 上的 6 个阶段进行 Roofline 理论分析与消融实验设计.

---

## 0. 硬件规格与符号

**TPU v7x 单 TensorCore (Chiplet):**

| 参数 | 值 |
|------|-----|
| HBM | 96 GB, 3690 GB/s |
| VMEM | 64 MiB |
| MXU | 1154 TFLOPS (BF16, Dual MXU) |
| ICI | 200 GB/s/axis/dir, 3D Torus |
| Ridge Point | ~313 FLOPs/byte |

**符号:**

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `T` / `T_L` | 全局 / 本地 token 数 | T_L = T/D |
| `E` / `E_L` | 全局 / 本地专家数 | E_L = E/D |
| `K` | top_k | 2-8 |
| `H` / `I` | hidden / intermediate size | 4096-8192 / 1024-4096 |
| `D` | ep_size | 1-256 |
| `B_w` / `B_t` | 权重 / token 字节数 | 2 (BF16) |
| `M_avg` | 每专家平均 token 数 | T_L × K / E_L |

---

## 1. Stage 1: Metadata (Gate, TopK, Permute, AllReduce)

**功能**: Gate 线性投影 → TopK 路由 → token 重排 → 跨设备 AllReduce 交换路由元数据.

- **Phase A** (kernel 外, XLA): `hidden_states @ W_gate` → sigmoid/softmax → grouped top-k → topk_weights, topk_ids
- **Phase A'** (理论): token-major → expert-major 重排 (当前融合在 Stage 2 fori_loop 中)
- **Phase B** (kernel 内): 加载 topk 结果, 计算 routing mask, 单次 AllReduce 交换 expert_sizes

### Roofline

```text
# Gate GEMM (HBM BW bound, W_gate 读取主导):
AI_gate ≈ T_L / 2        # 远低于 ridge point 313

# Permute (纯 DMA, AI ≈ 0):
T_permute = T_L × H × B_t × (1+K) / HBM_BW

# AllReduce (ICI 延迟 bound, payload 极小 ~1KB):
T_AR = log₂(D) × T_ici_step

# Stage 1 总延迟:
T_stage1 ≈ (H×E×4)/HBM_BW + T_permute + log₂(D) × T_ici_step
```

**瓶颈特征**: Phase A HBM BW bound; Phase B ICI 延迟 bound (AllReduce payload 极小, 启动开销主导).

**EP 缩放**: Stage 3 延迟 O(1/D) 下降, Stage 1 AllReduce O(log₂D) 增长. EP≥128 时 Stage 1 占端到端 30-76%.

### 消融实验

```python
# 扫描 EP 规模, 固定 DeepSeek-V3-like 配置 (E=256, K=8, H=8192, I=2048)
for ep_size in [8, 32, 64, 128, 256]:
    for T in [512, 8192]:  # decode / prefill
        topk_w, topk_ids = gate_and_topk(hidden_states, W_gate, ...)
        result = fused_ep_moe(mesh, tokens, w1, w2, w3, topk_w, topk_ids, ...)
        profile(result)
```

| 实验 | 变量 | 验证目标 |
|------|------|---------|
| EP 缩放 (Decode T=512) | D ∈ {8,32,64,128,256} | AllReduce 轮次 vs 延迟的缩放关系 |
| EP 缩放 (Prefill T=8192) | D ∈ {8,32,64,128,256} | Permute 主导 (大 T_L) vs AR 主导 (大 D) 的交叉点 |
| Pure JAX 对照 | JAX psum vs Pallas 手写 allreduce | XLA 自动融合 vs 手写 kernel 的延迟对比 |

---

## 2. Stage 2: All2All Dispatch (Scatter)

**功能**: 根据路由信息, 将本地 token 通过 DMA (本地) / ICI remote copy (远程) 分发到各专家设备.

- **Batch Scatter**: buffer 充足时一次遍历完成

### Roofline

```text
# 总数据量:
Total_bytes = bt × K × H × B_t
  本地: bt × K × H × B_t / D
  远程: bt × K × H × B_t × (D-1)/D

# 延迟:
T_stage2 = max(bt×H×B_t/HBM_BW, Bytes_ICI/ICI_BW) + bt×K×T_per_token_routing
```

**瓶颈特征**: Decode 以串行路由循环延迟为主; Prefill 以 ICI 带宽为主.

**DMA 启动开销**: 当前逐 expert 小 DMA 模式下, 85-92% 时间浪费在 ICI launch overhead. Per-device 合并 DMA 可在 D=4~16 时带来 4-16x A2A 加速.

### 消融实验

| 实验 | 消融 Flag / 方法 | 验证目标 |
|------|----------------|---------|
| 禁用 A2A | `disable_a2a=True` | A2A 占 kernel 总时间比例 |
| 变化 D | 改变 ep_size | D 对 scatter 延迟的缩放 |
| 逐 Expert vs 逐 Device DMA | kernel 修改: per_device batch | 合并 DMA 加速验证 (预期 4-16x) |
| DMA launch overhead 标定 | 1 token × N experts 线性拟合 | 提取 T_dma_launch, T_ici_launch |

---

## 3. Stage 3: Expert Compute (FFN1 + Activation + FFN2)

**功能**: 核心计算阶段. 对每个本地专家执行 SwiGLU FFN:
`Y = (SiLU(X @ W1) * (X @ W3)) @ W2`

### Roofline

```text
# Per-expert FLOPs:
FLOPs = 6 × M × H × I          # W1 + W3 + W2 三个 GEMM

# 总 FLOPs (与 E_L 无关):
Total_FLOPs = 6 × bt × K × H × I

# HBM 搬运 (权重主导):
Bytes_HBM = num_bt × E_L × 3 × H × I × B_w    # 权重
          + 2 × bt × K × H × B_t               # tokens

# Arithmetic Intensity:
AI ≈ 2 × bt × K / (num_bt × E_L × B_w)

# 延迟 (几乎总是 HBM bound):
T_stage3 ≈ Bytes_HBM / HBM_BW
```

**Compute Bound 临界条件**: 每专家需 M_crit ≈ R·B_w/2 个 token 才能 Compute Bound.

| 权重精度 | B_w | M_crit | T_crit (全局, E=256, K=8) |
|---------|-----|--------|--------------------------|
| BF16 | 2 | ~349 | ~11,168 |
| INT8 | 1 | ~175 | ~5,600 |
| INT4 | 0.5 | ~87 | ~2,792 |

> Decode (M_avg=8~64) 永远 HBM bound, MXU 利用率 < 21%. M_crit 与 H 无关, 主要由 Ridge Point 和 B_w 决定.

### 消融实验

| 实验 | 消融 Flag | 验证目标 |
|------|----------|---------|
| 禁用 FFN1 | `disable_dynamic_ffn1=True` | FFN1 compute vs DMA 平衡 |
| 禁用 FFN2 | `disable_dynamic_ffn2=True` | FFN2 compute vs DMA 平衡 |
| 禁用权重加载 | `disable_weight_load=True` | DMA 占比 (纯计算时间) |
| 禁用 token 读 | `disable_a2a_s_tile_read=True` | token staging 开销 |
| 禁用结果写 | `disable_a2a_s_acc_tile_write=True` | output staging 开销 |
| FFN1+FFN2 同禁 | 两个 flag 同开 | 纯 DMA pipeline 延迟 |
| 变化 bf/bd1/bd2/btc | 调整 block_config | DMA/compute 平衡点, MXU 利用率 |
| Pipeline 效率基线 | 隔离 Stage 3 (禁 A2A+SE) | 实际 η_mem vs 理论值 ~97.6% |

---

## 4. Stage 4: All2All Combine (Gather)

**功能**: Stage 2 的反向操作. 将专家计算结果通过 DMA/ICI 收集回原始设备.

### Roofline

```text
Total_bytes = bt × K × H × B_t        # 与 Stage 2 对称
T_stage4 = max(Bytes_ICI/ICI_BW, Bytes_local/HBM_BW) + T_serial
```

**关键**: Stage 4 与 Stage 3 流水线重叠 -- 每个专家完成 FFN 后立即启动 gather, 延迟大部分被 Stage 3 隐藏.

**Pipeline 失效条件**: D≥64 时, (D-1) × T_ici_launch ≥ 31.5 μs 超过 T_ffn_e (26 μs), pipeline 失效. 此时 deferred per-device batched gather 可降低 15-27%.

### 消融实验

| 实验 | 消融 Flag / 方法 | 验证目标 |
|------|----------------|---------|
| 禁用 A2A | `disable_a2a=True` | Stage 2+4 合计占比 |
| 单设备 | `ep_size=1` | 纯本地 DMA 的 gather 开销 |
| Eager vs Deferred Gather | kernel 修改: gather_mode | Pipeline vs batch 策略对比 (预期 D≤16 pipeline 胜, D≥64 batch 胜) |

---

## 5. Stage 5: Shared Expert Compute

**功能**: 与路由专家并行的共享专家 FFN. 所有 token 通过共享专家, 输出在 Stage 6 与路由专家输出相加. 计算被切分为多个 slice, 交错在 Stage 3 专家循环中以最大化重叠.

### Roofline

```text
FLOPs_SE = 6 × bt × H × SE_I
Bytes_SE = 3 × H × SE_I × B_w + se_blocks × bt × H × B_t

T_stage5 = max(FLOPs_SE/MXU_peak, Bytes_SE/HBM_BW)
T_stage5_effective = max(0, T_stage5 - T_overlap_with_A2A)
```

**关键**: 理想情况下 SE 与 A2A 等待时间完全重叠, T_stage5_effective ≈ 0.

### 消融实验

| 实验 | 消融 Flag | 验证目标 |
|------|----------|---------|
| 禁用 SE | `disable_shared_expert=True` | SE 的净延迟增量 (扣除重叠) |
| 变化 bse | 调整 block_config.bse | bse 对交错效率的影响 |
| SE alone | 禁 A2A + 禁路由专家 | SE 裸延迟 |

---

## 6. Stage 6: Top-K 加权求和 + 共享专家输出

**功能**: 从 gather buffer 收集每个 token 的 K 个专家输出, 按 topk_weights 加权求和, 加上 SE 输出, 写入最终 output.

### Roofline

```text
FLOPs = 2 × bt × K × H                         # VPU only
Bytes = bt × H × B_t × (K + 1)                  # 读 K 个 expert 输出 + 写 1 个 output
AI = K/(K+1) ≈ 1 FLOPs/byte                     # 严重 HBM BW bound

T_stage6 = max(Bytes/HBM_BW, bt×K×T_dma_launch) + FLOPs/VPU_peak
```

**瓶颈特征**: AI ≈ 1, 远低于 ridge point. 绝对延迟不高, 主要受小 DMA 串行化影响.

### 消融实验

| 实验 | 方法 | 验证目标 |
|------|------|---------|
| 变化 K | 调整 top_k | K 对 Stage 6 的线性缩放 |
| 禁 SE | `disable_shared_expert=True` | SE add 的边际开销 |

---

## 7. 端到端汇总

### 总延迟模型

```text
T_total ≈ T_stage1 + T_stage3 + T_stage6
         + max(0, T_stage2 - T_overlap)     # scatter 未被隐藏部分
         + max(0, T_stage5 - T_overlap)     # SE 未被隐藏部分
```

### 各阶段特征

| Stage | FLOPs | HBM Bytes | AI | 瓶颈 |
|-------|-------|-----------|----|------|
| 1. Metadata | 2T_L·H·E (Gate) | H·E·4 + T_L·H·B_t·(1+K) | T_L/2 | HBM BW + ICI 延迟 |
| 2. Scatter | 0 | bt·H·B_t | 0 | ICI BW / 串行 DMA |
| 3. Expert FFN | 6bt·K·H·I | E_L·3HI·B_w | 2btK/(E_L·B_w) | **HBM BW** |
| 4. Gather | 0 | bt·K·H·B_t | 0 | ICI BW (被 S3 隐藏) |
| 5. Shared Expert | 6bt·H·SE_I | 3H·SE_I·B_w | 中等 | HBM BW (被 A2A 隐藏) |
| 6. Output Acc | 2bt·K·H | bt·H·B_t·(K+1) | K/(K+1) | HBM BW |

### 关键观察

1. **Stage 3 主导**: 占计算量 >95%, HBM 搬运 >80%. 权重读取是主要瓶颈.
2. **Decode vs Prefill**: Decode AI 低 (严格 HBM BW bound); Prefill AI 稍高但仍 BW bound.
3. **Stage 1 在大 EP 下显著**: EP≥128 时 Stage 1 占 30-76%, 成为首要优化目标.
4. **Pipeline 重叠**: SE 与 A2A 的重叠效率直接影响端到端延迟.

---

## 8. 消融实验矩阵

### Flag 总表

| Flag | 影响阶段 | 作用 |
|------|---------|------|
| `disable_a2a` | 2, 4 | 禁用全部 All2All |
| `disable_dynamic_ffn1` | 3 | 禁用 FFN1 matmul |
| `disable_dynamic_ffn2` | 3 | 禁用 FFN2 matmul + activation |
| `disable_weight_load` | 3, 5 | 禁用全部权重 DMA |
| `disable_a2a_s_tile_read` | 3 | 禁用 token staging read |
| `disable_a2a_s_acc_tile_write` | 3 | 禁用 FFN output write |
| `disable_shared_expert` | 5 | 禁用共享专家 |
| `disable_all_reduce_metadata` | 1 | 禁用 AllReduce (需 disable_a2a 或 ep=1) |
| `disable_sync_barrier` | 1 | 禁用 barrier (需 disable_a2a 或 ep=1) |

### 实验序列

**Level 0 — Baseline**: 完整 kernel, 无消融.

**Level 1 — 粗粒度阶段隔离**:

| ID | Flags | 隔离阶段 |
|----|-------|---------|
| 1.1 | disable_a2a + metadata + barrier | Stage 3+5+6 |
| 1.2 | disable_shared_expert | Stage 1+2+3+4+6 |
| 1.3 | disable_a2a + SE + metadata + barrier | 纯 Expert FFN + Output |

**Level 2 — Stage 3 内部隔离**:

| ID | Flags (在 L1.3 基础上) | 验证目标 |
|----|----------------------|---------|
| 2.1 | + disable_ffn1 + disable_ffn2 | 纯 DMA pipeline |
| 2.2 | + disable_weight_load + tile_read + tile_write | 纯 compute |
| 2.3 | + disable_ffn2 | 仅 FFN1 |
| 2.4 | + disable_ffn1 | 仅 FFN2 |

**Level 3 — Pipeline 效率**:

```python
for bt in [8, 16, 32, 64, 128]:     # token tile 大小
for bse in [128, 256, 512, 1024]:    # SE 交错粒度
for bf in [256, 512, 1024, 2048]:    # intermediate tile
    fused_ep_moe(..., block_config=FusedMoEBlockConfig(...))
```

### Profile 指标

| 指标 | 来源 |
|------|------|
| 端到端延迟 | jax.block_until_ready + timer |
| HBM 读写量 | XLA profiler / LLO IR |
| MXU / ICI 利用率 | XLA profiler |
| VMEM 峰值 | LLO IR |

### Flag 组合安全表

| 组合 | 安全? | 约束 |
|------|-------|------|
| `disable_a2a` alone | ep=1 only | ep>1 需同时禁 metadata + barrier |
| `disable_all_reduce_metadata` alone | **No** | 需 disable_a2a 或 ep=1 |
| `disable_sync_barrier` alone | **No** | 需 disable_a2a 或 ep=1 |
| `disable_ffn1 + disable_ffn2` | Yes | 输出全零 |
| `disable_weight_load` alone | Yes | matmul 用未初始化权重 |
| `disable_shared_expert` alone | Yes | 无副作用 |

---

*v2.0 | 2026-05-05 | 基于 `kernels/_fused_moe_impl.py` 源码分析*
