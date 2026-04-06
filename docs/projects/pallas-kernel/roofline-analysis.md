# Roofline 分析深度指南

> 本文基于 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 系列整理，聚焦于 Pallas kernel 开发中最实用的 Roofline 分析方法。

---

## 1. 核心模型

### 1.1 时间分解

任何算子的执行时间受三个硬件约束限制：

$$T_\text{math} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Bytes}}{\text{Bandwidth Bytes/s}}$$

当计算与通信可以重叠时：

$$T_\text{lower} = \max(T_\text{math}, T_\text{comms})$$

$$T_\text{upper} = T_\text{math} + T_\text{comms}$$

上下界最多相差 2×。

- $T_\text{math} > T_\text{comms}$：**compute-bound**，硬件利用率高
- $T_\text{comms} > T_\text{math}$：**memory/communication-bound**，算力被浪费

### 1.2 Arithmetic Intensity（算术强度）

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$$

当 AI > 硬件临界强度时，算子为 compute-bound：

$$\text{AI}(\text{算子}) > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} = \text{AI}(\text{硬件})$$

### 1.3 各硬件临界强度

| 硬件 | bf16 FLOPs/s | HBM 带宽 | 临界 AI | 备注 |
|------|-------------|---------|--------|------|
| TPU v5e | 1.97×10¹⁴ | 8.1×10¹¹ | **240** | 推理优化芯片 |
| TPU v5p | 4.59×10¹⁴ | 2.8×10¹² | **164** | 训练芯片 |
| TPU v6e | 9.20×10¹⁴ | 1.6×10¹² | **575** | 当前主力 |
| H100 SXM | ~1.0×10¹⁵ | 3.35×10¹² | **298** | 去除 sparsity 2× |

> **从 VMEM 读取时**：VMEM 带宽约为 HBM 的 22×，临界 AI 降至 ~10–20，几乎所有 matmul 都是 compute-bound。这是 Pallas kernel 中 tiling 到 VMEM 的核心优势。

---

## 2. 矩阵乘法 Roofline

### 2.1 基础公式

对于 $X[\text{B}, \text{D}] \times Y[\text{D}, \text{F}] \rightarrow Z[\text{B}, \text{F}]$（bf16）：

| 指标 | 公式 |
|------|------|
| FLOPs | $2 \times B \times D \times F$ |
| 读取字节 | $2(BD + DF)$ |
| 写入字节 | $2BF$ |
| 总字节 | $2(BD + DF + BF)$ |
| AI | $\frac{BDF}{BD + DF + BF}$ |

### 2.2 简化情况：$B \ll D, F$

分母中 $DF$ 项主导：

$$\text{AI} \approx \frac{BDF}{DF} = B$$

**关键结论**：bf16 matmul 在 **B > 临界 AI** 时 compute-bound。

| 硬件 | 临界 batch size |
|------|----------------|
| TPU v5e | ~240 tokens |
| TPU v6e | ~575 tokens |
| H100 | ~298 tokens |

这里 B 是 **per-replica** 的 token 数（非 sequence 数），且与 sharding 无关（sharding 等比缩放计算和带宽）。

### 2.3 不同精度组合

| 权重精度 | 激活精度 | 计算精度 | 临界 batch |
|---------|---------|---------|-----------|
| bf16 | bf16 | bf16 | ~240 (v5e) |
| int8 | int8 | int8 | ~243 (v5e) |
| int8 | bf16 | bf16 | **~120** (v5e) |

int8 权重 + bf16 激活的组合最有利：权重字节减半，但计算速率不变，使得临界 batch 减半。

### 2.4 Tiling 强度

大 matmul 被分块为 tile 放入 VMEM。对于 tile 尺寸 $(bm, bk, bn)$：

$$\text{AI}(\text{tile}) \approx \frac{bm \times bn}{bm + bn}$$

例如 $bm = bn = 128$：$\text{AI} = 64$。$bm = bn = 256$：$\text{AI} = 128$。

> 在 Pallas kernel 中选择 tile 尺寸时，应确保 tile AI 大于 VMEM↔MXU 临界强度（~10–20），通常 128×128 已足够。

---

## 3. 逐元素运算 Roofline

### 3.1 Dot Product

对于 $x \cdot y$（bf16[N], bf16[N] → bf16[1]）：

$$\text{AI} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2} \text{ as } N \to \infty$$

**结论**：dot product 永远是 communication-bound（AI = 0.5 远低于任何硬件的临界值）。

### 3.2 通用逐元素运算

所有 elementwise 运算（ReLU、add、multiply 等）AI ≈ 1，**始终 memory-bound**。

**优化策略**：将逐元素运算与相邻的 matmul 融合（fuse），避免额外的 HBM 往返。

---

## 4. 注意力机制 Roofline

### 4.1 单头注意力算术强度

对于 Flash Attention 融合后的注意力（查询 T tokens，KV 缓存 S tokens）：

$$\text{AI} = \frac{ST}{S + T}$$

| 场景 | S | T | AI | 结论 |
|------|---|---|-----|------|
| Prefill/训练 | = T | T | T/2 | T > ~480 时 compute-bound |
| 解码生成 | ≫ 1 | 1 | ≈ 1 | **永远 memory-bound** |

### 4.2 GQA 对推理的影响

使用 Grouped Query Attention（G = N/K 个 Q 头共享一组 KV）：

- 生成时 AI → G（当 S 很大）
- 增大 G（更多 Q 头共享 KV）可提升算术强度，但仍难以达到 compute-bound

### 4.3 注意力 vs MLP FLOPs 占比

$$\frac{\text{attention FLOPs}}{\text{MLP FLOPs}} = \frac{T}{8D}$$

当 $T > 8D$ 时注意力 FLOPs 占主导。对于 D=8192，阈值约 64K tokens。

---

## 5. 多芯片通信 Roofline

### 5.1 ICI 通信临界强度

对于跨 ICI 的 sharded matmul：

| 硬件 | bf16 FLOPs/s | ICI 双向带宽 | 临界 AI |
|------|-------------|-------------|--------|
| TPU v5e | 1.97×10¹⁴ | 9.0×10¹⁰ | ~2,200 |
| TPU v5p | 4.59×10¹⁴ | 1.8×10¹¹ | ~2,550 |
| TPU v6e | 9.20×10¹⁴ | 1.8×10¹¹ | ~5,100 |

### 5.2 DP 临界 batch

纯数据并行（AllReduce 梯度）compute-bound 条件：

$$\frac{B}{X} > \frac{C}{W_\text{ici}} \div M_X$$

其中 $M_X$ 是 ICI 轴数量。TPU v5p 三轴：每芯片最低 ~850 tokens。

### 5.3 TP 临界条件（与 batch 无关）

$$Y < \frac{F \times M_Y}{C / W_\text{ici}}$$

TPU v5p：$Y < F / 2550$。对于 F=28672（LLaMA-70B）：最多 ~11 路 TP。

### 5.4 DCN 跨 pod 临界强度

$$\frac{C}{W_\text{dcn}} \approx \frac{4.59 \times 10^{14}}{6.25 \times 10^{9}} \approx 73,440$$

跨 pod 通信极其昂贵，要求每 slice 至少 ~73K tokens。

---

## 6. 实用 Roofline 分析流程

### 6.1 Kernel 开发前分析

1. **计算 FLOPs**：$2 \times \prod(\text{contracting dims}) \times \prod(\text{batch/output dims})$
2. **计算访存字节**：所有输入和输出的 $\text{size} \times \text{bytes\_per\_element}$
3. **算 AI**：FLOPs / Bytes
4. **对比硬件临界 AI**：确定 compute-bound 还是 memory-bound
5. **选择优化方向**：
   - Compute-bound → 提升 MXU 利用率（增大 tile、减少 pipeline bubble）
   - Memory-bound → 减少访存（fuse ops、减少精度、增大 batch）

### 6.2 Roofline 绘图

```python
import matplotlib.pyplot as plt
import numpy as np

def roofline_plot(B_range, D, F, flops_s, hbm_bw, label):
    total_flops = 2 * B_range * D * F
    flops_time = total_flops / flops_s
    comms_time = (2 * B_range * D + D * F + 2 * B_range * F) / hbm_bw
    total_time = np.maximum(flops_time, comms_time)
    return total_flops / total_time

bs = np.arange(1, 1024)

# TPU v6e
perf_v6e = roofline_plot(bs, 4096, 4096, 9.2e14, 1.6e12, 'v6e')

plt.figure(figsize=(10, 5))
plt.plot(bs, perf_v6e / 1e12, label='D=F=4096, TPU v6e')
plt.axhline(y=920, color='r', linestyle='--', label='Peak (920 TFLOPS)')
plt.xlabel('Batch Size (tokens)')
plt.ylabel('Achievable TFLOPS')
plt.legend()
plt.grid(True)
plt.title('Matmul Roofline')
```

---

## 7. 参考链接

- [How to Scale Your Model — Ch1: Rooflines](https://jax-ml.github.io/scaling-book/roofline)
- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
