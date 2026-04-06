# TPU 硬件规格参考

> 本文基于 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 系列整理，为 kernel 开发者提供跨代次的 TPU 硬件规格快速参考。

---

## 1. 计算与内存规格

### 1.1 芯片级规格

| 型号 | HBM 容量/芯片 | HBM 带宽/芯片 (B/s) | bf16 FLOPs/s/芯片 | int8 OPs/s/芯片 | MXU 尺寸 | 核数/芯片 |
|------|-------------|---------------------|-------------------|-----------------|---------|----------|
| TPU v3 | 32 GB | 9.0×10¹¹ | 1.4×10¹⁴ | 1.4×10¹⁴ | 128×128 | 2 |
| TPU v4p | 32 GB | 1.2×10¹² | 2.75×10¹⁴ | 2.75×10¹⁴ | 128×128 | 2 |
| TPU v5p | 96 GB | 2.8×10¹² | 4.59×10¹⁴ | 9.18×10¹⁴ | 128×128 | 2 |
| TPU v5e | 16 GB | 8.1×10¹¹ | 1.97×10¹⁴ | 3.94×10¹⁴ | 128×128 | 1 |
| TPU v6e | 32 GB | 1.6×10¹² | 9.20×10¹⁴ | 1.84×10¹⁵ | **256×256** | — |

### 1.2 临界算术强度

| 型号 | HBM 临界 AI | VMEM 临界 AI (≈HBM/22) | ICI 临界 AI |
|------|-----------|----------------------|-----------|
| TPU v5e | 240 | ~11 | ~2,200 |
| TPU v5p | 164 | ~7 | ~2,550 |
| TPU v6e | 575 | ~26 | ~5,100 |

> VMEM 临界 AI 极低（~10–20），意味着从 VMEM 读取的操作几乎总是 compute-bound。这是 Pallas tiling 策略的理论基础。

---

## 2. 互联规格

### 2.1 ICI（芯片间直连）

| 型号 | 单向带宽/链路 (B/s) | 双向带宽/链路 (B/s) | 拓扑 | Pod 尺寸 |
|------|-------------------|-------------------|------|---------|
| TPU v3 | 1.0×10¹¹ | 2.0×10¹¹ | 2D torus | 32×32 |
| TPU v4p | 4.5×10¹⁰ | 9.0×10¹⁰ | **3D torus** | 16×16×16 |
| TPU v5p | 9.0×10¹⁰ | 1.8×10¹¹ | **3D torus** | 16×20×28 |
| TPU v5e | 4.5×10¹⁰ | 9.0×10¹⁰ | 2D torus | 16×16 |
| TPU v6e | 9.0×10¹⁰ | 1.8×10¹¹ | 2D torus | 16×16 |

### 2.2 Wraparound 规则

- **v5e/v6e**：仅轴长度为 16 时有 wraparound（如 8×16 的长轴有 wrap）
- **v4p/v5p**：轴长度为 4 的倍数时有 wraparound（通过光交换实现）
- **无 wrap 时**：通信时间大约 **翻倍**（从 N/2 跳变为 N-1 跳）

### 2.3 DCN（数据中心网络）

| 型号 | DCN 出口带宽/芯片 (B/s) |
|------|----------------------|
| TPU v5e | 3.125×10⁹ |
| TPU v5p | 6.25×10⁹ |
| TPU v6e | 1.25×10¹⁰ |

### 2.4 PCIe

| 型号 | PCIe 带宽/芯片 (B/s) |
|------|-------------------|
| TPU v4p | 1.6×10¹⁰ |
| TPU v6e | 3.2×10¹⁰ |

> PCIe 比 HBM 慢约 100×，从 host 加载数据通常需要 B > 60,000 才能 compute-bound。

---

## 3. 芯片/核/Host 组织

```text
Host (CPU)
  │  PCIe
  ├── Tray: 4 chips (v4p/v5p)  或  8 chips (v5e/v6e)
  │     ├── Chip 0 ── Core 0 ── MXU×4, VPU, VMEM
  │     │            └── Core 1 ── MXU×4, VPU, VMEM  (megacore)
  │     ├── Chip 1
  │     └── ...
  │
  └── ICI direct links between chips (不经过 host)
```

- **v4p/v5p**：每芯片 2 核（megacore），共享内存，作为一个加速器使用
- **v5e**：每芯片 1 核，推理优化
- **v6e**：每芯片核数未公开

---

## 4. MXU 详细规格

### 4.1 Systolic Array 工作原理

MXU 是 128×128（v6e 为 256×256）的脉动阵列，每个 ALU 执行 multiply-and-add：

```text
权重 (RHS, 128×128) 从上方流入
        ↓
  ┌─────────────────────────┐
  │  ALU  ALU  ALU  ...     │ ← 输入 (LHS, 8×128) 从左侧流入
  │  ALU  ALU  ALU  ...     │
  │  ...                    │
  └─────────────────────────┘
        ↓
  输出从底部流出
```

- 单次操作：`bf16[8,128] × bf16[128,128] → f32[8,128]`，耗时 8 cycles
- 初始 pipeline bubble：权重和激活对角线式加载
- 后续输入无额外 bubble

### 4.2 MXU 对维度的要求

| 规则 | 说明 |
|------|------|
| 最小有效维度 | ≥ 128（v6e ≥ 256） |
| 多 MXU 扩展 | 需 ≥ 128 × MXU 数量（v4p/v5p 有 4 MXU → ≥512） |
| 不足时 | 硬件自动 pad 到 128 的倍数，浪费计算和带宽 |

---

## 5. VPU 详细规格

### 5.1 架构

VPU 是 **(8, 128)** 的 2D SIMD 向量单元：8 sublanes × 128 lanes。

| 属性 | TPU v5p |
|------|--------|
| ALU 数/指令 | 4（每 lane×sublane 对有 4 独立 ALU） |
| bf16 FLOPs/s | ~1.4×10¹³（约为 MXU 的 1/10） |
| 指令延迟 | 2 cycles |
| 吞吐量 | 1 cycle（pipeline） |

### 5.2 VREG（Vector Register）

| 属性 | TPU v4 | TPU v5p |
|------|--------|--------|
| VREG shape | (8, 128) | (8, 128) |
| VREG 大小 | 4 KiB (32-bit) | 4 KiB |
| VREG 数量/核 | 32 | 64 |
| 总容量/核 | ~128 KiB | ~256 KiB |
| VMEM 读带宽 | — | 3 VREGs/cycle |
| VMEM 写带宽 | — | 1 VREG/cycle |

### 5.3 归约操作

- **sublane 内归约**（8 个元素）：shuffle 操作 ~1 cycle，3 步即可完成（shift 4, 2, 1）
- **跨 lane 归约**（128 lanes）：需要 XLU（Cross Lane Unit），**极慢**，应尽量避免

---

## 6. 内存带宽层次

```text
速度从快到慢：
VMEM ↔ MXU    ：~22× HBM 带宽
HBM ↔ VMEM   ：~1-2 TB/s
PCIe (CPU↔HBM)：~100× 慢于 HBM
DCN           ：最慢 (~6.25 GB/s/芯片)
```

### 6.1 VMEM 关键特性

- 容量：TPU v5e 为 128 MiB
- 可编程控制（非自动 cache）
- 数据必须先从 HBM 拷贝到 VMEM 才能被 MXU 使用
- 支持 prefetch：可在 attention 计算时预加载 FFW 权重

---

## 7. 与 GPU 的关键差异

| 维度 | TPU | GPU (H100) |
|------|-----|-----------|
| 核心数量 | 1-2 大核 | 132 SM |
| 编程模型 | SIMD | SIMT（每线程独立指令指针） |
| 快速缓存 | 128 MB VMEM（可编程） | 32 MB SMEM + 50 MB L2（L2 不可编程） |
| VMEM 带宽 | ~40 TB/s | SMEM ~5.5 TB/s（L2） |
| 寄存器 | 256 KB VREG | 32 MB 寄存器 |
| 延迟隐藏 | 编译器流水线 | 硬件 warp 调度（64 warps/SM） |
| 互联 | 2D/3D torus（最近邻） | NVLink + 交换机（fat-tree） |
| 矩阵 vs 向量 FLOPs | ~10:1 | ~30:1 |
| 达到峰值难度 | 较容易（编译器优化） | 较难（需协调 SM 共享 L2） |

**对 Pallas kernel 开发者的启示**：

- TPU 的 VMEM 比 GPU 的 SMEM 大 4×，延迟更低 → 寄存器溢出（spill）到 VMEM 代价较小
- TPU 更依赖编译器进行指令调度，而非硬件线程切换 → kernel 设计应注重编译器友好的访问模式
- 矩阵 vs 向量 FLOPs 比率更低（10:1 vs 30:1）→ VPU 运算在 TPU 上相对不那么"便宜"

---

## 8. 参考链接

- [How to Scale Your Model — Ch2: All About TPUs](https://jax-ml.github.io/scaling-book/tpus)
- [How to Scale Your Model — Ch12: GPUs](https://jax-ml.github.io/scaling-book/gpus)
- [TPU Deep Dive](https://henryhmko.github.io/posts/tpu/tpu.html)
- [Writing TPU Kernels with Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)
- [Rafi Witten's High Performance LLMs 2024](https://github.com/rwitten/HighPerfLLMs2024)
