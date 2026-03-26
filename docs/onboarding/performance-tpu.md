# 当我们谈论 TPU/DSA 性能优化时，我们在说什么？（GPU Kernel 开发者避坑指南）

对于习惯了在 CUDA 编程模型和 GPU 架构下进行性能调优的开发者来说，首次接触 TPU（Tensor Processing Unit）或类似的特定领域加速器（DSA，如 DTU）可能会遇到不少反直觉的性能瓶颈。本文旨在帮助有 GPU 背景的 Kernel 开发者快速建立直觉，并掌握从宏观架构到微观 LLO（底层优化）的调优技法。

---

## 1. 宏观哲学：收敛 Search Space，降维打击

从 Search Space（搜索空间）的角度考虑，TPU 的性能友好区间（Sweet Spot）非常狭窄，远比 GPU 小得多。在 TPU 上做性能优化的核心本质是：**要把喂给硬件的复杂性和不确定性降到最低**。

GPU 的设计哲学是“动态掩盖”（通过庞大的 Warp Scheduler 处理分支和内存延迟），而 TPU 是纯粹的“静态规划”。为了让代码逻辑落入 TPU 的性能友好区间，我们需要从上层框架到计算库，一层一层地剥离复杂度（例如在框架层就做好 Padding，消除 Varlen 的动态性）。只有最终喂给底层 TPU 的指令足够“单纯”，才能彻底拉满硬件利用率。

---

## 2. 核心概念对照表：CUDA vs. TPU 罗塞塔石碑

理解 TPU 最快的方式是将其底层组件与 CUDA 概念进行一一映射：

| CUDA (GPU) 概念 | TPU (DSA) 对应物 | 核心差异点 |
| :--- | :--- | :--- |
| **Streaming Multiprocessor (SM)** | **TPU Core** | TPU Core 拥有更强的独立性，通常是单核单指令流。 |
| **Tensor Core** | **MXU (Matrix Multiply Unit)** | TPU MXU 是庞大的 128x128（或更高）阵列，追求大尺寸吞吐，而非 GPU 的碎块调用。 |
| **Shared Memory** | **VMEM / SPMA (SRAM)** | 都是显式管理的局部内存。但 TPU 上由编译器静态编排，没有 Bank Conflict 困扰。 |
| **Registers** | **VR (Vector) / VACC (Accumulator)** | TPU 寄存器分工极细：输入向量在 VR，矩阵在 MXU，累加结果在 VACC。 |
| **Warp Scheduler (动态掩盖)** | **VLIW + Software Pipelining (静态掩盖)** | **关键区别：** TPU 没有硬件级线程上下文切换，全靠编译器静态排布指令掩盖延迟。 |
| **NCCL / NVLink** | **ICI (Inter-Core Interconnect)** | TPU 芯片间通过专用 ICI 网络直连（2D/3D Torus），延迟极低且无需主机参与。 |
| **Triton / CUDA C++** | **JAX Pallas / LLO** | Pallas 是 TPU 界的 Triton，要求开发者显式定义数据 Tiling 与 DMA 调度。 |

---

## 3. 算子优化铁律：向脉动阵列对齐

TPU/DSA 的绝对算子核心是 **脉动阵列（Systolic Array）**。为了发挥这部分算力，必须在数据规模和排布上做出妥协。

### 3.1 空间对齐与隐性 Padding 惩罚

在 GPU 上，127x127 的小矩阵能依靠 Warp 调度容忍。但在 TPU 的刚性网格下，硬件会强行 **补零（Padding）** 到 128x128。这意味着你的计算资源和内存带宽正在被大量的零所浪费！

- **铁律**：务必让张量维度（Batch, Hidden, SeqLen）是 **128、32 或 8 的整数倍**。

### 3.2 大尺寸至上与 CMAR 提升

GPU 在很小的 Shape 下能跑得不错，但 TPU 只有在大吞吐下才能打满利用率。

- **实测推荐**：对于 MatMul `[M, K] x [K, N]`，推荐 **M >= 512, K >= 1024, N >= 512**。
- **维度优先级**：**N >= K > M**。数据量越大，Engine（MXU 和 1D Vector）的利用率越高。

---

## 4. 掩盖延迟机制：从 Occupancy 到软件流水线

这是 GPU 开发者最需要转变的直觉。在 CUDA 中，我们通过提高 **Occupancy（占用率）** 来掩盖延迟——当一个 Warp 停顿时，硬件自动切到另一个。
**但在 TPU 上，完全没有硬件级线程切换（No Warp Scheduler）。**

### 4.1 软件流水线 (Software Pipelining)

TPU 掩盖延迟的唯一武器是：**手动/编译器的指令排布**。

- **异步 DMA**：当 MXU 在计算第 `i` 块数据时，DMA 引擎必须已经在异步加载第 `i+1` 块数据到 VMEM 中。
- **VLIW 打包**：在同一个 VLIW 指令包中，开发者（或优化器）需要同时塞入 `Vector Load`、`Scalar Add` 和 `Matrix Multiply` 指令。
- **展开 (Unroll) 的真意**：在 TPU 上，循环展开不仅仅是为了减少开销，更是为了腾出足够的“指令槽位”来排布异步加载指令，使得流水线不被打断。

---

## 5. 访存黑盒：“最低两维”铁律

TPU 内存操作极其死板，开发者必须主动迎合。

### 5.1 最低两维度保护法则

尽量让张量的 **最低两个维度**（内层连续维度）避开 `transpose`、`gather`、`scatter` 等复杂操作。

- **推荐**：`Linear Copy` 效率最高。
- **高维降维打击**：高维度的复杂变换，尽量在 Scalar Engine 中通过 **计算逻辑地址偏移** 来模拟，而不是触发真实物理内存搬迁。
- 倒数第一维做复杂操作的成本高于倒数第二维的成本。

### 5.2 算力换带宽 (Rematerialization)

在训练大模型时，**激活重计算（jax.remat）** 是 TPU 的神技。TPU 的 MXU 算力极其过剩，但 HBM 带宽是瓶颈。宁可重算，也不要频繁读写 HBM。

---

## 6. 分布式优势：降维打击的 SPMD

相比于在 GPU 上手写复杂的 Megatron 分布式算子和 NCCL 通信，TPU 提供了 **SPMD (Single Program Multiple Data)**。

- **ICI 直连**：TPU Pod 内部走专用 ICI 网络，无需通过 PCIe 或 CPU。
- **自动切分**：开发者只需定义全局大张量（Global Tensor）和物理网格（Mesh），XLA 编译器会自动在 MatMul 内部插入高效的 `AllGather` 或 `ReduceScatter` 算子。

---

## 7. 终极验证：微观 Profiling 与 Busy Engine

所有的调优最终都要在 **ns 级 Trace Viewer** 中验证。

### 7.1 寻找“Busy Engine”

TPU 内部包含：Scalar (0D)、Vector (1D)、MXU (2D/Matrix)、XLU (Transpose)。
**优化目标**：在 Trace 时间轴中，至少有一家 Engine 处于持续的 Busy 状态。如果所有 Engine 都有大段空白，说明逻辑阻塞严重。

### 7.2 核心循环定位法（循环计数法）

在底层 LLO 指令中定位源码位置：

1. **算力估算**：根据 IO 规模估算大小循环次数。
2. **Trace 计数**：在 Trace Viewer 中“数格子”，先定位大循环块，再定位内层核心循环。
3. **模式识别**：通过观察 VR/VACC 寄存器的 `load/store/spills`（寄存器溢出）频率，确认当前指令片段是否为预想的 Kernel 逻辑。

---

## 8. 总结：TPU 老将的三板斧

1. **静态化与空间对齐**：固定 Shape，拉大尺寸，把逻辑交给谓词（Select/Where）执行。
2. **掩盖延迟靠展开**：利用 Software Pipelining，在计算的同时预加载下一批次数据。
3. **守住最低两维**：确保内存线性访问，利用 Scalar 引擎做高维逻辑变换。

TPU 是一台极其刚性但威力巨大的“压路机”。只要你主动收敛 Search Space，铺平对齐与线性的赛道，它的算力滚轮就能榨干哪怕最极端的巅峰性能。
