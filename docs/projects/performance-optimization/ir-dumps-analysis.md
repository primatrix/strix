# IR Dumps 性能分析指南

本文档说明 `ir_dumps/` 目录下各文件的内容，以及如何利用这些中间表示（IR）文件进行 TPU kernel 性能分析。

## 1. IR Dumps 目录与文件概览

### 1.1 目录结构

```text
ir_dumps/
├── hlo/          # XLA StableHLO 输入（最高层 IR）
├── mosaic/       # Mosaic 编译器各阶段 IR（MSA → LLO lowering）
└── llo/          # LLO 后端流水线（指令选择 → VLIW bundle 打包 → 寄存器分配）
```

### 1.2 HLO 目录（StableHLO）

每个子目录对应一个 JIT 编译函数，包含：

| 文件 | 说明 |
|------|------|
| `module.mlir` | StableHLO MLIR 模块，包含 `stablehlo.constant`、`stablehlo.broadcast_in_dim` 等高级算子 |
| `compile_options.pb` | XLA 编译选项 protobuf |
| `topology.pb` | TPU 拓扑配置（mesh 划分信息） |

**用途**：理解 XLA 输入的计算图——算子类型、张量形状、sharding 策略（`sdy.sharding`）。这是最接近用户代码的 IR，适合做 roofline 分析时确认算子清单。

### 1.3 Mosaic 目录（编译器全流程）

文件命名规则：

```text
{timestamp}-{pass_序号}-mosaic-dump-{kernel_name}-{pass_name}.txt
```

例如：`1777010575361678845-0013-...-post-lower-to-llo.txt`

每个 kernel 的 dump 文件分两组：**deserialization**（pass 0000，独立 timestamp）和**主优化流水线**（pass 0001-0015，同一 timestamp）。此外还有独立 timestamp 的 `.mlirbc` 二进制输出。同一 kernel（同一 timestamp）的多个 `.txt` 文件按 pass 序号（0001 到 0015）排列，记录了编译的全流程。下面是各 pass 的内容说明：

| # | Pass 名称 | IR 层级 | 内容说明 |
|---|-----------|---------|---------|
| 0000 | `post-deserialization` | MSA (TPU Dialect) | **独立 timestamp**。序列化 IR 的初始加载，memref 仅有 `#tpu.memory_space<>` 标注，无 tiling layout。包含 `tpu.device_id`、`tpu.sem_signal`/`sem_wait`、`tpu.enqueue_dma` 等 |
| 0001 | `original` | MSA (TPU Dialect) | 用户代码直接映射的 TPU 方言 IR。与 0000 逻辑内容相同，但所有 memref 带完整的 `#tpu.tiled<>` layout 标注。包含 `tpu.device_id`、`tpu.sem_signal`、`tpu.enqueue_dma`、`scf.for`/`scf.if` 控制流，memref 类型标注 `#tpu.memory_space<hbm/vmem/smem/semaphore_mem>` |
| 0002 | `post-infer-memref-layout` | MSA | 推断每个 memref 的 tiling layout，通过 `tpu.erase_memref_layout` 和 `tpu.reinterpret_cast` 调整布局 |
| 0003 | `post-infer-memref-space` | MSA | 明确每个 buffer 的物理存储空间（HBM / VMEM / SMEM / semaphore_mem） |
| 0004 | `post-infer-memref-layout-simplify` | MSA | 简化 layout 表达式，消除冗余的 tiling 嵌套（IR 大小约缩减 50%） |
| 0005 | `post-pre-canon-optimization` | MSA | 预规范化优化——公共子表达式消除、死代码消除、常量折叠 |
| 0006 | `post-canonicalize-mosaic` | MSA | 将 TPU 自定义方言规范化为基础操作（如 `tpu.sem_signal` → 标准同步形式） |
| 0007 | `post-canonicalize-mosaic-simplify` | MSA | 规范化后的清理优化 |
| 0008 | `post-infer-vector-layout` | MSA→LLO 过渡 | 推断向量寄存器的数据布局，为 lowering 做准备 |
| 0009 | `post-relayout-insertion` | MSA→LLO 过渡 | 在需要数据格式转换的位置插入 relayout 指令（如 HBM 连续布局 ↔ 向量 tiled 布局） |
| 0010 | `post-apply-vector-layout` | MSA→LLO 过渡 | 应用向量寄存器布局，将 memref 操作转换为向量操作 |
| 0011 | `post-apply-vector-layout-simplify` | MSA→LLO 过渡 | 清理向量布局应用后的冗余指令 |
| 0012 | `post-logical-to-physical-device-id` | MSA | device_id 从逻辑映射到物理 |
| **0013** | **`post-lower-to-llo`** | **LLO** | **核心文件**。完整的 Low-Level Operations IR，包含所有 VLIW 向量/标量指令（`llo.vector_load`、`llo.vadd`、`llo.vmatmul` 等），DMA 操作使用 `llo.dma_done` 同步 |
| 0014 | `post-eliminate-llo-extensions` | LLO | 消除 LLO 扩展指令。**实测与 0013 内容完全相同**（`llo.dma_done` 仍然保留，共 69 处），此 pass 当前为空操作 |
| **0015** | **`post-finalize-llo`** | **LLO (终)** | **最终 LLO**。DMA 操作已 finalize（`llo.enqueue_dma` 包含完整的 src/dst strides 和 priority），`llo.dma_done` 仍保留用于同步等待，是最接近实际硬件指令的文本 IR |
| — | `post-finalize-llo-post-emitter` | Binary | **独立 timestamp**（不属于 0001-0015 序列）。`.mlirbc` 二进制 MLIR，供 LLO 后端流水线使用 |

### 1.4 LLO 目录（后端流水线）

Mosaic 的 `post-finalize-llo.txt`（0015）输出后，进入 LLO 后端流水线。LLO 目录包含 ~24,000 个文件，来自 **423 次编译运行**，涵盖 **47 个不同的编译单元**（`TLP`、`broadcast_in_dim.0`、`fused-moe-*`、`copy.*`、`fusion.*`、`<late-initialization>`、`<late-finalization>` 等）。每个编译单元经历相同的线性流水线（约 79 个 pass）。

文件命名规则：

```text
{timestamp}-{compilation_unit}-{pass_number}-{pass_name}.txt
```

例如：`1777010474964125622-TLP-30-DGO-vliw-packed-bundles.txt`

其中 `compilation_unit` 是编译单元名称（如 `TLP`、`broadcast_in_dim.0`、`fused-moe-k_8-renorm_k-bt_32_32_32-...`），`pass_number` 是两位数序号（00-78）。

#### 完整 LLO 流水线（以 TLP 为例，79 个 pass）

下表列出单个编译单元的完整 pass 序列。不同编译单元可能只运行其中一部分 pass。

#### 元数据与预处理（Pass 00-02）

| Pass | 名称 | 说明 |
|------|------|------|
| — | `hlo` | HLO 输入模块 |
| 00 | `fingerprints` | Kernel 指纹信息 |
| — | `memory-space-assignment-*` | 内存空间分配元数据（buffer info / alloc info / schedule info，非编译 pass） |
| 01 | `pre-dedup` | 预去重 |
| 02 | `pre-auto-mxu-assigner` | MXU 自动预分配 |

#### 前端优化（Pass 03-22）

| Pass | 名称 | 说明 |
|------|------|------|
| 03 | `original` | 初始 LLO IR |
| 04 | `post-invert-loops` | 循环反转优化 |
| 05 | `post-CP` | 常量传播（Copy Propagation） |
| 06 | `post-rematerialize-allocations` | 重新物化内存分配 |
| 07 | `post-simplifier-1` | 第 1 轮代码简化 |
| 08 | `post-bf16-coalescing` | bf16 合并优化 |
| 09 | `post-mem-to-reg` | 内存提升到寄存器 |
| 10 | `post-GVN` | 第 1 轮全局值编号（消除冗余） |
| 11 | `post-DCE` | 第 1 轮死代码消除 |
| 12 | `post-simplifier-2` | 第 2 轮代码简化 |
| 13 | `post-x8-coalescing` | 8-lane 合并优化 |
| 14 | `post-MXU-assigner` | MXU 矩阵乘法单元分配 |
| 15 | `post-GVN-2` | 第 2 轮全局值编号 |
| 16 | `post-DCE-2` | 第 2 轮死代码消除 |
| 17 | `post-decomposer` | 复杂指令分解为基础指令 |
| 18 | `post-if-conversion` | 分支转 predicated 指令 |
| 19 | `post-cssa` | 第 1 轮 CSSA 构造 |
| 20 | `post-constant-materializer` | 第 1 轮常量化 |
| 21 | `pre-cssa` | Pre-CSSA |
| 22 | `post-cssa` | 第 2 轮 CSSA 构造 |

#### DGO 阶段：依赖图优化 + VLIW 打包（Pass 23-34）

| Pass | 名称 | 说明 |
|------|------|------|
| **23** | **`critical-path`** | **关键路径分析**：输出每个 region 中各条指令到 block 末端的周期距离 |
| 24 | `DGO-post-critical-path-scheduler` | 基于关键路径的指令调度 |
| 25 | `DGO-post-load-cse-and-s2l-forwarding` | Load CSE + Store-to-Load 前向 |
| 26 | `DGO-post-load-store-optimizer` | Load/Store 优化 |
| 27 | `DGO-post-memory-instruction-fusion` | 访存指令融合 |
| 28 | `DGO-post-reassociate-accumulations` | 累加重新关联 |
| 29 | `DGO-post-llo-late-decomposer` | 第 1 轮延迟指令分解 |
| **30** | **`DGO-vliw-packed-bundles`** | **VLIW bundle 打包**：指令打包为 bundle，`;;` 分隔同一 bundle 内的多条指令 |
| 31 | `DGO-post-vliw-bundle-scheduler` | VLIW bundle 级调度 |
| 32 | `DGO-post-llo-late-decomposer-2` | 第 2 轮延迟指令分解 |
| 33 | `DGO-post-sink-address-calculation` | 地址计算下沉 |
| 34 | `DGO-post-rematerialization` | 重物化 |

#### 后端：寄存器分配前优化（Pass 35-43）

| Pass | 名称 | 说明 |
|------|------|------|
| 35 | `post-llo-dependency-graph-optimizations` | 依赖图优化 |
| 36 | `post-independent-region-scheduler` | 独立区域调度 |
| 37 | `post-bounds-check` | 边界检查 |
| 38 | `post-cssa` | 第 3 轮 CSSA |
| 39 | `post-address-relocation` | 地址重定位 |
| 40 | `post-GVN-3` | 第 3 轮全局值编号 |
| 41 | `post-DCE-3` | 第 3 轮死代码消除 |
| 42 | `post-cssa` | 第 4 轮 CSSA |
| 43 | `post-constant-materializer` | 第 2 轮常量化 |

#### 寄存器分配与 Bundle 重打包（Pass 44-64）

| Pass | 名称 | 说明 |
|------|------|------|
| **44** | **`schedule-analysis_packed-bundles-pre-ra`** | RA 前调度分析 |
| **45** | **`packed-bundles-pre-ra`** | **寄存器分配前的 VLIW bundles**（含虚拟寄存器） |
| 46 | `post-bundle-packing-pre-ra` | Bundle 打包清理 |
| 47 | `post-grt` | Global Register Transfer |
| 48 | `pre-finalize-registers-and-allocations` | 寄存器与分配预终结 |
| 49 | `pre-availability-transform` | 可用性变换前 |
| 50 | `post-availability-transform` | 可用性变换后 |
| **51** | **`post-ra`** | **寄存器分配完成** |
| 52 | `post-scalar-select-transform` | 标量 select 变换 |
| **53** | **`schedule-analysis_packed-bundles-no-ra-deps`** | 消除 RA 依赖后调度分析 |
| **54** | **`packed-bundles-no-ra-deps`** | 消除 RA 依赖后的 bundles |
| 55 | `post-bundle-packing-no-ra-deps` | Bundle 打包清理 |
| 56 | `region-expansion-no-ra-deps` | 区域展开 |
| **57** | **`schedule-analysis_packed-bundles-no-spills-fills`** | 消除 spill/fill 后调度分析 |
| **58** | **`packed-bundles-no-spills-fills`** | 消除 spill/fill 后的 bundles（关键：显示实际寄存器压力） |
| 59 | `post-bundle-packing-no-spills-fills` | Bundle 打包清理 |
| 60 | `region-expansion-no-spills-fills` | 区域展开 |
| **61** | **`schedule-analysis_packed-bundles-post-ra`** | RA 后调度分析 |
| **62** | **`packed-bundles-post-ra`** | **寄存器分配后的 VLIW bundles**（物理寄存器，spill/fill 已插入） |
| 63 | `post-bundle-packing-post-ra` | Bundle 打包清理 |
| 64 | `region-expansion` | 最终区域展开 |

#### 最终输出（Pass 65-78）

| Pass | 名称 | 说明 |
|------|------|------|
| 65 | `pre-delay_hlo-static-per-bundle-utilization` | 延迟变换前的利用率矩阵 |
| 66 | `post-delay-converter` | 延迟变换 |
| **67** | **`final_hlo-static-per-bundle-utilization`** | **逐 bundle 硬件利用率矩阵**（每行 = 一个 bundle 在各 FU 上的占用数） |
| **68** | **`schedule-analysis_final_bundles`** | **最终调度统计**（总 bundle 数、空 bundle 数、按 HLO 分类的 bundle 分布） |
| **69** | **`final_bundles`** | **最终 VLIW bundles**：每条指令带物理地址（`0x00`, `0x01`, ...），`;;` 分隔同 bundle 指令，`(%p_)` 标记 predicated 指令 |
| 70 | `hlo-static-bundle-profile` | Bundle profile |
| 71 | `trace-markers` | Trace markers |
| 72 | `final-top-level-llo` | 最终顶层 LLO |
| 73-74 | `bundles-pre-codegen` | Codegen 前 bundles + 调度分析 |
| 75 | `all-fingerprints` | 所有 pass 指纹 |
| 76 | `post-code-generation` | Codegen 完成 |
| 77 | `heap_sizes` | 堆大小 |
| 78 | `overlay-graph` | Overlay 图 |

### 1.5 哪个文件最关键？

**分析 kernel 性能时，按分析需求选择文件：**

| 分析目的 | 推荐文件 |
|---------|---------|
| 理解算法结构、控制流 | `mosaic/original.txt`（0001） |
| 指令级分析（数据依赖、寄存器压力） | `mosaic/post-lower-to-llo.txt`（0013） |
| DMA/compute overlap 分析 | `mosaic/post-finalize-llo.txt`（0015） |
| **VLIW bundle 级分析** | `llo/*-final_bundles.txt` |
| **逐 bundle 硬件利用率** | `llo/*-final_hlo-static-per-bundle-utilization.txt` |
| **关键路径（执行周期下界）** | `llo/*-critical-path.txt` |
| **调度统计（bundle 数、空泡率）** | `llo/*-schedule-analysis_final_bundles.txt` |
| **寄存器分配前后对比（spill 分析）** | `llo/*-packed-bundles-pre-ra.txt` vs `llo/*-packed-bundles-post-ra.txt` |
| **VLIW 第一次打包** | `llo/*-DGO-vliw-packed-bundles.txt` |

### 1.6 LLO 指令速查表

| 功能类别 | 指令 | 硬件单元 | 说明 |
|----------|------|---------|------|
| **矩阵乘法** | `llo.vmatprep.mubr` | MXU | 准备矩阵乘法操作数（左矩阵） |
| | `llo.vmatmul.mubr` | MXU | 执行矩阵乘法（右矩阵） |
| | `llo.vmatres` | MXU | 读取矩阵乘法结果（返回 `vector<8x128xf32>`） |
| **向量运算** | `llo.vadd.f32/s32` | Vector ALU | 向量加法 |
| | `llo.vmul.f32/s32` | Vector ALU | 向量乘法 |
| | `llo.vselect` | Vector ALU | 向量条件选择（predicated move） |
| | `llo.vexp` / `llo.vdiv` / `llo.vsqrt` / `llo.vrsqrt` | Vector ALU | 向量超越函数（通过查表+多项式逼近实现） |
| | `llo.vmax` / `llo.vmin` | Vector ALU | 向量比较 |
| | `llo.vcmp` | Vector ALU | 向量比较（返回 mask） |
| | `llo.vreduce` | Vector ALU | 向量归约 |
| | `llo.vconv` | Vector ALU | 向量类型转换（如 bf16→f32） |
| | `llo.vand` / `llo.vor` / `llo.vnot` / `llo.vxor` | Vector ALU | 向量位运算 |
| | `llo.vneg` / `llo.vabs` | Vector ALU | 向量取负/取绝对值 |
| | `llo.vfloor` / `llo.vceil` | Vector ALU | 向量取整 |
| **向量数据搬运** | `llo.vector_load` | Vector Load | 从 VMEM/SMEM 加载向量到 VPR |
| | `llo.vector_store` | Vector Store | 从 VPR 写入 VMEM/SMEM |
| | `llo.vslreplicate` | Vector ALU | 跨 sublane 复制 |
| | `llo.vbcast_sublane_chunk` | Vector ALU | 子 lane chunk 广播 |
| | `llo.vbroadcast` | Vector ALU | 标量广播为向量 |
| | `llo.vxlaneid` | Vector ALU | 获取当前 lane ID |
| | `llo.vtranspose` | Vector ALU | 向量转置 |
| **标量运算** | `llo.sadd.s32` / `llo.ssub.s32` / `llo.smul.u32` / `llo.sdiv.u32` / `llo.srem.u32` | Scalar ALU | 标量算术 |
| | `llo.constant` | Scalar ALU | 标量常量 |
| | `llo.sld` | Scalar Load | 从 SMEM 加载标量 |
| | `llo.saddr_scaled` | Scalar ALU | 标量地址计算（base + offset × scale） |
| | `llo.assume_multiple` | Scalar ALU | 对齐假设优化 |
| **DMA** | `llo.enqueue_dma` | DMA Engine | 发起异步 DMA 传输（HBM↔VMEM） |
| | `llo.dma_done` | DMA Engine | 等待 DMA 完成（lowers to `enqueue_dma` + signal） |
| | `llo.dma_start` / `llo.dma_wait` | DMA Engine | 显式 DMA 启动/等待 |
| **同步** | `llo.vsync.add` / `llo.vsync.add.remote` | Sync | 信号量 increment（本地/远程） |
| | `llo.vwait.ge` | Sync | 等待信号量 ≥ 阈值 |
| **控制流** | `scf.for` / `scf.if` / `scf.yield` | Scalar ALU | 结构化控制流（循环/分支） |

### 1.7 VLIW 指令集完整参考

本节列出 `final_bundles.txt` 中出现的所有 VLIW 指令助记符（mnemonic）。这些是硬件级指令，不带 `llo.` 前缀。指令按功能单元分类，与 §2.1 的 VLIW 架构对应。

> **标记说明**：标注 `(pre-lowering)` 的指令仅在寄存器分配前（如 `original`、`packed-bundles-pre-ra`）出现，最终 `final_bundles.txt` 中会被替换为等价的 lowered 形式。

#### DMA 引擎（8 条）

独立于三条指令槽，异步执行 HBM ↔ VMEM/SMEM 数据传输。

| 指令 | 说明 |
|------|------|
| `dma.general` | 通用 DMA 操作（编译器内部表示） |
| `dma.hbm_to_vmem` | HBM → VMEM 传输 |
| `dma.vmem_to_hbm` | VMEM → HBM 传输 |
| `dma.hbm_to_smem` | HBM → SMEM 传输 |
| `dma.smem_to_hbm` | SMEM → HBM 传输 |
| `dma.vmem_to_smem` | VMEM → SMEM 传输 |
| `dma.done` | DMA 完成检测 (pre-lowering) |
| `dma.done.wait` | DMA 完成等待（阻塞直到传输完成） |

#### 标量 ALU（Scalar ALU，14 条）

处理地址计算、循环控制、标量算术。每 cycle 最多发射 1 条。

| 指令 | 说明 |
|------|------|
| `sadd.s32` | 有符号 32 位加法 |
| `ssub.s32` | 有符号 32 位减法 |
| `smul.u32` | 无符号 32 位乘法 |
| `smul.addr` | 地址计算专用乘法 (pre-lowering) |
| `sdivrem.u32` | 无符号 32 位除法 + 取余（同时产生商和余数） |
| `srem.u32.pop` | 无符号 32 位取余 (pre-lowering) |
| `smin.u32` | 无符号 32 位取最小值 |
| `sshll.u32` | 逻辑左移 |
| `sshllo.u32` | 逻辑左移（带溢出检测，pre-lowering） |
| `sshrl.u32` | 逻辑右移 |
| `sshra.s32` | 算术右移（保留符号位） |
| `sand.u32` | 按位 AND |
| `sor.u32` | 按位 OR |
| `sxor.u32` | 按位 XOR |

#### 标量比较（Scalar Compare，7 条）

比较结果写入 predicate 寄存器，用于条件分支和 predicated 执行。

| 指令 | 说明 |
|------|------|
| `scmp.eq.s32.totalorder` | 等于（全序比较） |
| `scmp.ne.s32.totalorder` | 不等于 |
| `scmp.lt.s32.totalorder` | 小于（有符号） |
| `scmp.lt.u32.totalorder` | 小于（无符号） |
| `scmp.le.s32.totalorder` | 小于等于 |
| `scmp.gt.s32.totalorder` | 大于 |
| `scmp.ge.s32.totalorder` | 大于等于 |

#### 标量内存与寻址（Scalar Memory，9 条）

| 指令 | 说明 |
|------|------|
| `sld` | 从 SMEM 加载标量值 |
| `sst` | 向 SMEM 存储标量值 |
| `smov` | 标量移动 / 常量物化 |
| `scalar_lea.hbm` | 加载有效地址（HBM 指针） |
| `scalar_lea.vmem` | 加载有效地址（VMEM 指针） |
| `scalar_lea.smem` | 加载有效地址（SMEM 指针） |
| `scalar_lea.sflag` | 加载有效地址（同步标志指针） |
| `scalar_parameter_address` | 加载 kernel 参数地址 |
| `scalar_select` | 标量条件选择（三元运算） |

#### 标量控制流（Scalar Control，8 条）

| 指令 | 说明 |
|------|------|
| `sbr.rel` | 相对分支（条件跳转） |
| `sfence` | 标量屏障（确保之前所有标量操作完成） |
| `shalt.err` | 错误停机（断言失败时终止） |
| `setrngseed` | 设置随机数种子 |
| `sphi` | 标量 phi 节点（SSA 合流点） |
| `spop` | 标量弹栈 |
| `spop.drf` | 标量弹栈（DRF 变体） |
| `scheckge` / `scheckne` | 标量断言检查（注释中出现） |

#### 标量类型转换（4 条）

| 指令 | 说明 |
|------|------|
| `int_to_ptr.hbm` | 整数 → HBM 指针 |
| `int_to_ptr.vmem` | 整数 → VMEM 指针 |
| `int_to_ptr.sflag` | 整数 → 同步标志指针 |
| `int_to_ptr.sparse_core_sequencer_sflag` | 整数 → 稀疏核心排序器标志指针 |

#### 向量 ALU（Vector ALU，12 条）

处理向量算术和位运算。每 cycle 最多发射 4 条（VALU 容量 = 4）。

| 指令 | 说明 |
|------|------|
| `vadd.f32` | 向量加法（f32） |
| `vadd.low.f32.bf16` | 向量加法（bf16 输入提升到 f32 后加） |
| `vadd.s32` | 向量加法（有符号 32 位整数） |
| `vsub.s32` | 向量减法（有符号 32 位整数） |
| `vmul.f32` | 向量乘法（f32） |
| `vmul.u32` | 向量乘法（无符号 32 位整数） |
| `vmax.f32` | 向量取最大值（f32） |
| `vand.u32` | 向量按位 AND |
| `vand.u32.u16` | 向量按位 AND（u16 输入） |
| `vand.u32.u8` | 向量按位 AND（u8 输入） |
| `vor.u32` | 向量按位 OR |
| `vxor.u32` | 向量按位 XOR |

#### 向量移位（Vector Shift，3 条）

| 指令 | 说明 |
|------|------|
| `vshll.u32` | 向量逻辑左移 |
| `vshrl.u32` | 向量逻辑右移 |
| `vshra.s32` | 向量算术右移 (pre-lowering) |

#### 向量比较（Vector Compare，8 条）

比较结果写入向量 mask 寄存器。

| 指令 | 说明 |
|------|------|
| `vcmp.eq.f32.partialorder` | 向量等于（f32，偏序——NaN 不比较） |
| `vcmp.eq.s32.totalorder` | 向量等于（s32，全序） |
| `vcmp.lt.f32.partialorder` | 向量小于（f32，偏序） |
| `vcmp.lt.s32.totalorder` | 向量小于（s32，全序） |
| `vcmp.lt.u32.totalorder` | 向量小于（u32，全序） |
| `vcmp.gt.s32.totalorder` | 向量大于（s32，全序） |
| `vcmp.ge.s32.totalorder` | 向量大于等于 (pre-lowering) |
| `vcmp.ne.s32.totalorder` | 向量不等于（s32，全序） |

#### 向量超越函数（Vector Math，8 条）

通过查表 + 多项式逼近实现，延迟较高。

| 指令 | 说明 |
|------|------|
| `vlog2.f32` | 向量 log₂（f32） |
| `vlog2.pop` | 向量 log₂ (pre-lowering) |
| `vpow2.f32` | 向量 2ˣ（f32） |
| `vpow.pop` | 向量幂 (pre-lowering) |
| `vrcp.f32` | 向量倒数（f32） |
| `vrcp.pop` | 向量倒数 (pre-lowering) |
| `vrsqrt.f32` | 向量平方根倒数（f32） |
| `vrsqrt.pop` | 向量平方根倒数 (pre-lowering) |

#### 向量选择与掩码（Vector Select/Mask，3 条）

| 指令 | 说明 |
|------|------|
| `vcmask` | 从比较结果生成 mask |
| `vsel` | 向量条件选择（基于 mask 选择两个源操作数） |
| `vsmask.f32` | 带 mask 的向量存储 |

#### 向量内存操作（Vector Memory，16 条）

VLOAD 容量 = 3（每 cycle 最多 3 条 load），VSTORE 容量 = 2。

| 指令 | 说明 |
|------|------|
| `vld` | 从 VMEM 加载向量到 VPR |
| `vld.sshfl` | 加载 + sublane shuffle |
| `vst` | 从 VPR 存储向量到 VMEM |
| `vst.msk` | 带 mask 的向量存储 |
| `vst_source` | 向量存储源标注（调试用） |
| `vstv` | 标量值广播为向量并存储 |
| `vpack.c.b16` | 向量打包（压缩到 b16） |
| `vpack.c.bf16` | 向量打包（压缩到 bf16） |
| `vunpack.c.0.s8` | 向量解包（s8，元素 0） |
| `vunpack.c.h.bf16` | 向量解包（bf16，高位半） |
| `vunpack.c.l.b16` | 向量解包（b16，低位半） |
| `vunpack.c.l.bf16` | 向量解包（bf16，低位半） |
| `vunpack.c.l.s4` | 向量解包（s4，低位） |
| `vunpack.i.l.bf16` | 向量解包（bf16，交错低位） |
| `vtos` | 向量 → 标量（提取单个元素） |
| `vdelay` | 向量延迟（插入等待周期） |

#### 向量广播与置换（Vector Broadcast/Permute，6 条）

XLU0/XLU1 是两条独立的跨 lane 数据搬运通道。

| 指令 | 说明 |
|------|------|
| `vbcast.lane.b32.xlu0` | 跨 lane 广播（通过 XLU0） |
| `vbcast.lane.b32.xlu1` | 跨 lane 广播（通过 XLU1） |
| `vperm.xlu0` | 跨 lane 置换（通过 XLU0） |
| `vperm.xlu1` | 跨 lane 置换（通过 XLU1） |
| `vrot.lane.b32.xlu0` | lane 内旋转（通过 XLU0） |
| `vrot.slane` | sublane 内旋转 |

#### 向量转置与交换（Vector Transpose/Exchange，8 条）

多步转置协议：start → cont → end，或 start.end 合并。

| 指令 | 说明 |
|------|------|
| `vxpose.xlu0.b32.start` | 转置开始（XLU0） |
| `vxpose.xlu0.b32.cont` | 转置继续（XLU0） |
| `vxpose.xlu0.b32.end` | 转置结束（XLU0） |
| `vxpose.xlu0.b32.start.end` | 转置开始+结束合并（XLU0，小矩阵） |
| `vxpose.xlu1.b32.start` | 转置开始（XLU1） |
| `vxpose.xlu1.b32.cont` | 转置继续（XLU1） |
| `vxpose.xlu1.b32.end` | 转置结束（XLU1） |
| `vcombine.high` / `vcombine.low` | 向量高/低半合并 |

#### MXU 矩阵运算（14 条）

MXU 是 TPU 的核心算力单元。指令助记符编码了数据类型、broadcast 模式和目标 MXU。

| 指令 | 说明 |
|------|------|
| `vmatprep.mubr.bf16.mxu0` | 矩阵乘法准备——左矩阵加载到 MXU0（bf16，MU broadcast row） |
| `vmatprep.mubr.bf16.mxu1` | 同上，目标 MXU1 |
| `vmatprep.mubr.f32.mxu0` | 同上，f32 精度，MXU0 |
| `vmatprep.mubr.f32.mxu1` | 同上，f32 精度，MXU1 |
| `vmatprep.subr.bf16.mxu0` | 矩阵准备（SU broadcast row 模式，bf16，MXU0） |
| `vmatprep.subr.bf16.mxu1` | 同上，MXU1 |
| `vmatpush1.bf16.msra.mxu0` | 矩阵数据推入 MXU0（bf16，MSRA 模式） |
| `vmatpush1.bf16.msra.mxu1` | 同上，MXU1 |
| `vmatmul.mubr.bf16.gmra.mrb` | 矩阵乘法执行（bf16，GMRA+MRB 模式） |
| `vmatmul.mubr.bf16.vlgmr.msra.gmra.mrb` | 矩阵乘法执行（bf16，完整 pipeline 模式） |
| `vmatmul.mubr.f32.gmra.mrb` | 矩阵乘法执行（f32，GMRA+MRB 模式） |
| `vmatmul.mubr.f32.vlgmr.msra.gmra.mrb` | 矩阵乘法执行（f32，完整 pipeline 模式） |
| `vsetiar.raw.iar0` | 设置索引地址寄存器 IAR0 |
| `vsetiar.raw.iar1` | 设置索引地址寄存器 IAR1 |

**MXU 指令后缀含义**：

| 后缀 | 全称 | 说明 |
|------|------|------|
| `mubr` | MU Broadcast Row | 左矩阵按行广播 |
| `subr` | SU Broadcast Row | 右矩阵按行广播 |
| `gmra` | GM Read A | 从通用矩阵寄存器读取 A 矩阵 |
| `mrb` | Matrix Register Buffer | 矩阵寄存器缓冲区 |
| `msra` | MS Read A | 从矩阵存储读取 A 矩阵 |
| `vlgmr` | VLGM Read | 从向量/逻辑通用矩阵读取 |
| `mxu0` / `mxu1` | MXU 0 / MXU 1 | 目标矩阵乘法单元编号 |

#### 向量同步（Vector Synchronization，10 条）

用于多设备/多核间的 semaphore 同步。

| 指令 | 说明 |
|------|------|
| `vsyncadd` | 信号量加值（本地） |
| `vsyncadd.remote` | 信号量加值（远程设备，通过 ICI） |
| `vsyncset` | 信号量设值（本地） |
| `vsyncset.remote` | 信号量设值（远程设备） |
| `vsyncmov` | 信号量移动 |
| `vsyncpa` | 信号量程序地址同步 |
| `vwait.ge` | 等待信号量 ≥ 阈值（硬同步点） |
| `vdwg.mxu0` | MXU0 数据等待门控 (pre-lowering) |
| `vlaneseq` | 生成 lane 序列号（当前 core 的 lane ID） |
| `vsettm` | 设置线程掩码 |

#### 向量控制与杂项（Vector Control/Misc，17 条）

| 指令 | 说明 |
|------|------|
| `vmov` | 向量移动 |
| `vmmov` | 向量 mask 移动 |
| `vpush` | 向量压栈 |
| `vphi` | 向量 phi 节点 (pre-lowering) |
| `vmphi` | 向量 mask phi 节点 (pre-lowering) |
| `vmand` | 向量 mask AND |
| `vmor` | 向量 mask OR |
| `vtrace` | 向量 trace（调试/性能采样） |
| `vrng` | 向量随机数生成 |
| `vpop.eup` | 从 EUP（执行单元池）弹出结果 |
| `vpop.f32.mrb` | 从 MRB（矩阵寄存器缓冲）弹出 f32 结果 |
| `vpop.sfrf` | 从 SFRF（特殊功能寄存器）弹出结果 |
| `vpop.permute.xlu0` | 弹出 + XLU0 置换 |
| `vpop.permute.xlu1` | 弹出 + XLU1 置换 |
| `vpop.trf.xlu0` | 从 TRF（转置寄存器）弹出，XLU0 |
| `vpop.trf.xlu1` | 从 TRF（转置寄存器）弹出，XLU1 |
| `vsmask.f32` | 带 mask 的向量存储（f32） |

#### 谓词操作（Predicate Operations，5 条）

谓词寄存器用于条件执行（predicated instruction），标记为 `(%p_)`。

| 指令 | 说明 |
|------|------|
| `pmov` | 谓词移动 |
| `pneg` | 谓词取反 |
| `por` | 谓词 OR |
| `pnand` | 谓词 NAND |
| `pphi` | 谓词 phi 节点 (pre-lowering) |

#### 指令命名约定

VLIW 指令助记符遵循以下命名规则：

| 前缀 | 功能单元 | 示例 |
|------|---------|------|
| `s` | Scalar ALU | `sadd`, `smul`, `sld` |
| `v` | Vector ALU / MXU | `vadd`, `vmatmul`, `vld` |
| `dma` | DMA 引擎 | `dma.hbm_to_vmem` |
| `p` | 谓词寄存器 | `pmov`, `por` |
| `scalar_` | Scalar 寻址 | `scalar_lea.hbm` |
| `int_to_ptr` | 类型转换 | `int_to_ptr.vmem` |

数据类型后缀：`.f32`（单精度浮点）、`.bf16`（brain float 16）、`.s32`（有符号 32 位）、`.u32`（无符号 32 位）、`.b16`（16 位）、`.s8`（8 位有符号）、`.s4`（4 位有符号）。

内存空间后缀：`.hbm`（片外显存）、`.vmem`（向量 SRAM）、`.smem`（标量 SRAM）、`.sflag`（同步标志）。

## 2. VLIW 指令执行周期分析

### 2.1 TPU v7x VLIW 架构概述

TPU v7x 每个 TensorCore 在每个时钟周期可以同时发射 **三条指令** 到不同功能单元：

```text
┌──────────────────────────────────────────────────┐
│                  VLIW Bundle (每 cycle)            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ Scalar  │  │ Vector  │  │   MXU   │  + DMA   │
│  │  ALU    │  │  ALU    │  │ (矩阵)  │  (异步)   │
│  └─────────┘  └─────────┘  └─────────┘          │
│  地址计算      向量运算      矩阵乘法    独立调度  │
│  控制流        数据搬运      卷积        │
│  标量算术      格式转换      ...        │
└──────────────────────────────────────────────────┘
```

关键约束：

- **每周期每单元最多 1 条指令**（Scalar/Vector/MXU 各 1 条）
- MXU 指令有 **多周期延迟**（pipeline 深度），结果不能立刻使用
- DMA 与计算通过 **semaphore** 同步

上述功能单元的具体分工：

- **Scalar ALU**：处理地址计算、循环控制、标量算术。瓶颈出现在循环迭代变量、条件判断密集时。
- **Vector ALU**：处理向量加减乘除、激活函数、数据格式转换、寄存器间搬运。包含 Vector Load/Store 单元。
- **MXU**：处理矩阵乘法和卷积。是 TPU 算力的主要来源，但需要足够的 tile 填充率才能喂饱。
- **DMA Engine**：异步执行 HBM → VMEM 和 VMEM → HBM 的数据传输，独立于三条指令槽。

### 2.2 三种周期分析方法（按精度递进）

#### 方法 A：直接读 `final_bundles.txt`（最精确）

`llo/*-final_bundles.txt` 是经过 VLIW bundle 打包和寄存器分配的最终输出。每条指令带有 **物理地址**（`0x00`, `0x01`, ...），同一 bundle 内的指令共享一个地址并用 `;;` 分隔：

```text
0x2b   :  { %v329_v24 = vxor.u32 %v328_v23, %v327_v22 }
0x2c   :  { %v330_v25 = vxor.u32 1519409121, %v329_v24  ;;  %v343_v34 = vxor.u32 2925155241, %v329_v24 }
0x2d   :  { %v331_v26 = vmul.u32 2449846741, %v330_v25  ;;  %v344_v37 = vmul.u32 2223506493, %v343_v34 }
0x2e   :  { %v332_v27 = vshrl.u32 %v331_v26, 16  ;;  %v345_v40 = vshrl.u32 %v344_v37, 16 }
```

**分析**：

- 第 0x2b 行：仅 1 条 Vector 指令（bubble: Scalar + MXU 空闲）
- 第 0x2c 行：2 条 Vector 指令（`;;` 分隔），**无 Scalar/MXU 指令** → Vector 并行，但 MXU 空闲
- 第 0x2d-0x2e 行：2 条 Vector 指令
- **总 bundle 数** = 最后地址 + 1

#### 方法 B：读 `critical-path.txt`（下界估算）

`llo/*-critical-path.txt` 给出每个 basic block 中各条指令到 block 结束的 **最小周期距离**：

```text
New basic block in region: region106
  Length to end: 112, %s300 = sld [smem:[#allocation26]]
  Length to end: 106, %s477 = sshll.u32 %s300, 1
  Length to end: 105, %s303 = sadd.s32 %s477, %s301
  Length to end: 104, %v308 = vstv %s303
  ...
```

`Length to end` = 从当前指令到该 block 末尾的 **关键路径长度**（cycle 数）。block 第一条指令的 `Length to end` 就是该 block 的总执行周期下界。实际运行时可能因资源冲突而更长。

#### 方法 C：从 Mosaic LLO 估算（最粗略）

当没有 llo/ 后端文件时（如只有 Mosaic IR dump），可以从 `mosaic/post-lower-to-llo.txt` 按功能单元分组估算：

```python
# 按功能单元分类统计
scalar_ops = count(sadd, ssub, smul, sdiv, srem, constant, sld, saddr_scaled, assume_multiple)
vector_ops = count(vadd, vmul, vexp, vmax, vreduce, vselect, vconv, vslreplicate,
                   vbcast_sublane_chunk, vbroadcast, vxlaneid, vtranspose,
                   vector_load, vector_store)
mxu_ops = count(vmatmul, vmatprep, vmatres)

# 下界 ≈ max(scalar_ops, vector_ops, mxu_ops)
# 上界 ≈ scalar_ops + vector_ops + mxu_ops（完全无并行）
```

**实际精度**：由于 Mosaic LLO 未 bundle，只能得到非常粗糙的估计。推荐始终使用 `final_bundles.txt` 做精确周期分析。

### 2.3 从 `final_bundles.txt` 读取精确的 VLIW bundle

`final_bundles.txt` 中的 bundle 格式：

```text
0x0b   :  { %v419_v0 = vlaneseq }
0x0c   :  { %v828_v1 = vshrl.u32 %v419_v0, 7 }
0x0d   :  { %v421_v2 = vshrl.u32 %v828_v1, 1  ;;  %v422_v3 = vand.u32 1, %v828_v1 }
0x0e   :  { %v423_v4 = vshll.u32 %v422_v3, 2  ;;  %v430_v6 = vsub.s32 %v421_v2, %v828_v1 }
```

**读取规则**：

- `0x0b`, `0x0c` ... 是 **物理指令地址**（= 周期序号）
- 同一 bundle 内的多条指令用 `;;` 分隔
- 每条指令有后缀 `_vN` 表示虚拟寄存器号（`%v419_v0` → vreg #419, slot 0）
- `(%p_)` 前缀表示 predicated（条件执行）指令
- `vstv %s303` 是 scalar-to-vector 搬运
- `[smem:[#allocation26]]` 是标量内存引用

**从 bundle 序列直接计算执行时间**：

1. 总 bundle 数 = 最后地址 + 1 = 总 cycle 数
2. 空 bundle（`{}`）= bubble cycle（所有 FU 空闲）
3. 单指令 bundle = 2 个 FU 空闲
4. 双指令 bundle（1 个 `;;`）= 1 个 FU 空闲的 VLIW 并行
5. 三指令 bundle（2 个 `;;`）= 满利用率

实际例子（从 `final_bundles.txt` 第 10-19 行）：

```text
0x0b: { vlaneseq }            ← 1 Vector, Scalar+MXU idle
0x0c: { vshrl }               ← 1 Vector, Scalar+MXU idle
0x0d: { vshrl ;; vand }      ← 2 Vector (VLIW并行), MXU idle
0x0e: { vshll ;; vsub }      ← 2 Vector, MXU idle
```

这 4 个 cycle 都是 Vector-dominant，MXU 完全空闲 → 属于地址计算 prologue。

### 2.4 使用 `schedule-analysis_*.txt` 做宏观统计

`llo/*-schedule-analysis_final_bundles.txt` 提供 kernel 级别的调度统计：

```text
Schedule analysis:
    total scheduled bundles:    298
    empty scheduled bundles:    14
    non empty scheduled bundles: 284
     281.50 scheduled bundles (99.12%): <no hlo>
       2.50 scheduled bundles ( 0.88%): %broadcast_in_dim.0 = ...
    [opcode]       4 scheduled bundles ( 1.41%): broadcast
```

**关键指标**：

| 指标 | 含义 | 健康值 |
|------|------|--------|
| `total scheduled bundles` | 总 VLIW bundle 数 = 总执行 cycle 数 | — |
| `empty scheduled bundles` | 所有 FU 均空闲的 bundle | 应 < 5% |
| `<no hlo>` bundles | 编译器生成的辅助代码（地址计算、同步等） | overhead，越低越好 |
| 按 HLO/opcode 的 bundle 分布 | 实际计算占总 cycle 的比例 | 应 > 80% |

**计算 bubble rate**：`empty / total = 14 / 298 = 4.7%` → 相对健康。

### 2.5 `critical-path.txt` 与 VLIW 调度的关系

`critical-path.txt` 在 DGO 阶段生成（VLIW 打包之前），给出的是 **无资源约束** 下的最小执行周期：

```text
New basic block in region: region106 {members=109}
  Length to end: 112, %s300 = sld [smem:[#allocation26]]
  Length to end: 106, %s477 = sshll.u32 %s300, 1
  Length to end:  65, %v336 = vsub.s32 %v334, %v335
  ...
  Length to end:   1, 最后一条指令
```

Block 第一条指令的 `Length to end` 就是该 block 的理想执行周期。对比 `final_bundles.txt` 中该 block 的实际 bundle 数，可知 **调度损失**：

```text
调度损失 = actual_bundles - critical_path_length
```

损失来源：

- VLIW bundle slot 冲突（多条 Vector 指令不能同时发射 >2 条）
- MXU pipeline 阻塞
- 强制同步点（semaphore 等待）

## 3. 数据依赖与指令分析

### 3.1 SSA 依赖图：Def-Use 链

LLO IR 使用 SSA（Static Single Assignment），每个 `%N` 有唯一的一次定义（def），可以被多次使用（use）。分析数据依赖就是追踪 def-use 链：

```mlir
%279 = llo.vector_load %271 : i32 -> vector<8x128xi32>     // def %279
%296 = llo.vadd.s32 %295, %72 : vector<8x128xi32>           // def %296
%298 = llo.vslreplicate %279, %297 : ... -> ...             // use %279, def %298
%300 = "llo.vbcast_sublane_chunk"(%298, %299) : ...         // use %298, def %300
...
%681 = llo.vadd.s32 %491, %503 : vector<8x128xi32>          // use %491, %503 (前面 def 的)
%683 = llo.vadd.s32 %681, %515 : vector<8x128xi32>          // RAW: %681 def → use
%685 = llo.vadd.s32 %683, %527 : vector<8x128xi32>          // RAW: %683 def → use
```

**RAW（Read After Write）是关键**：后面的指令必须等待前面的指令产生结果。

### 3.2 循环携带依赖与 Pipeline Stall

在 `scf.for` 中，`iter_args` 表示循环间携带的依赖：

```mlir
%46 = scf.for %arg35 = %c0 to %c4 step %c1
      iter_args(%arg36 = %c0) -> (i32) {
    // iter_args 的初始值 = %c0
    // 每次迭代 yield 新值给下一轮
    ...
    scf.if %65 {
        %1002 = arith.andi %61, %c1  // use %61 来自循环体内的 def
        ...
    }
    scf.yield %61  // yield 新值，依赖所有之前的操作
}
```

循环携带依赖意味着：**下一轮迭代必须等待上一轮迭代的 `yield` 值**。如果 yield 值的计算链较长，会直接限制 VLIW 的指令级并行度。

### 3.3 DMA 异步依赖链

DMA 是异步的，依赖通过 semaphore 传递：

```mlir
// Step 1: 发起 DMA（将数据从 HBM 搬到 VMEM）
tpu.enqueue_dma source(%35 : ...hbm...) target(%37 : ...vmem...)
                target_semaphore(%39 : ...semaphore...)

// Step 2: 等待 DMA 完成（通过 semaphore 同步）
"llo.dma_done"(%263, %262)   // 等待 semaphore %262 >= %263

// Step 3: 从 VMEM 加载到向量寄存器
%279 = llo.vector_load %271 : i32 -> vector<8x128xi32>  // 消费 DMA 写入的数据
```

**分析要点**：

- DMA 发起后、数据可用前，可以执行不依赖该数据的计算（**overlap**）
- `dma_done` 是硬同步点：之后的指令必须等待 DMA 完成
- 如果 `enqueue_dma` 和 `dma_done` 之间没有足够的计算指令填充，说明 **DMA 延迟未充分掩盖**

### 3.4 跨设备远程 DMA 同步依赖（EP=8）

本 kernel 是一个 EP=8 的 MoE kernel——8 个设备各持有一组 experts。路由阶段之后，每个设备需要将 tokens 通过远程 DMA 发送到其他 7 个设备，同时从其他 7 个设备接收 tokens。

#### Semaphore 全同步协议

在数据交换开始前，8 个设备通过 semaphore 完成一次全同步（all-gather barrier）：

```mlir
// Original IR（0001-original.txt）中，8 个 tpu.sem_signal 各指向不同 device_id
// device_id = row * 8 + col，col 从 0 到 7
tpu.sem_signal %arg34, %c1 device_id %10   // signal device col=0
tpu.sem_signal %arg34, %c1 device_id %13   // signal device col=1
tpu.sem_signal %arg34, %c1 device_id %16   // signal device col=2
// ... 共 8 个 signal，覆盖全部 EP shard ...
tpu.sem_signal %arg34, %c1 device_id %31   // signal device col=7

// 等待 8 个远程 signal（来自全部 8 个 device）
tpu.sem_wait %arg34, %c8
```

Lower 到 LLO 后，`tpu.sem_signal ... device_id` 变为 `llo.vsync.add.remote`，`tpu.sem_wait` 变为 `llo.vwait.ge`：

```mlir
// post-eliminate-llo-extensions.txt 中对应的 lowered 形式
llo.vsync.add.remote %arg34, %133, %135, %76   // 给 (row, col) 设备发 signal
// ... 共 7 个 remote signal（不含本地）...
llo.vsync.add %arg34, %179                      // 本地 signal
llo.vwait.ge %arg34, %178                       // 等待计数 >= 8
```

#### 远程 DMA 数据交换

每个设备选出路由目标后，通过 `tpu.enqueue_dma ... device_id` 将 token 数据发往远程设备，同时通过 `tpu.wait_dma2` 接收来自远程设备的数据：

```mlir
// 发送：远程 DMA，指定 target device_id
tpu.enqueue_dma source(%src_vmem) target(%dst_vmem)
    source_semaphore(%src_sem) target_semaphore(%dst_sem)
    device_id(%device)

// 接收：等待远程 DMA 完成，数据从 src（远程 VMEM）写入 dst（本地 VMEM）
tpu.wait_dma2 semaphore(%sem) src(%remote_vmem) dst(%local_vmem)
```

**关键性能分析点**：

- **显式同步**：`tpu.sem_wait` / `llo.vwait.ge` 是一个硬同步点——必须等待所有 8 个设备的 signal 到达才能继续，如果某些设备还未完成前序计算，此处产生 stall
- **远程 DMA 延迟**：跨设备 DMA 延迟远高于本地 DMA（需经过 ICI 互联），`wait_dma2` 返回前数据不可用
- **DMA/Compute Overlap**：理想情况下，远程 DMA 应该与本地计算 overlap。可以通过检查 `sem_wait` / `vwait` 被下发的时间点来判断：如果它在远程 DMA 发起后立即被等待，说明没有 overlap；如果中间插入了本地计算指令，说明存在 overlap 机会
- **本 kernel 中**：路由阶段后 8 个设备各自完成 all-to-all 数据交换，semaphore 等待和远程 DMA 等待是两个关键同步点

### 3.5 数据依赖分析工具方法

在实际分析中，你可以：

1. **提取依赖图**：将 `%N` 的 def 和所有 use 构建有向图
2. **计算关键路径长度**：在依赖图上做拓扑排序，路径权重 = 指令延迟
3. **识别同步点**：标记所有 `dma_done` / `vwait` 作为必须等待的硬同步
4. **分析循环携带依赖**：检查 `scf.yield` 的值是否在下一轮迭代早期就被需要

伪代码方法：

```text
def critical_path(instructions):
    ready_time = {}  # %value → 最早可用 cycle

    for inst in topo_order:
        issue_cycle = max(ready_time[operand] for operand in inst.inputs)
        # 考虑功能单元冲突（VLIW 约束）
        issue_cycle = max(issue_cycle, fu_available[inst.functional_unit])

        for result in inst.outputs:
            ready_time[result] = issue_cycle + inst.latency
        fu_available[inst.functional_unit] = issue_cycle + 1

    return max(ready_time.values())
```

## 4. 寄存器占用与压力分析

### 4.1 TPU v7x 寄存器层次

```text
寄存器层次（从快到慢）：

1. VPR (Vector Processor Register) — 向量寄存器文件
   - 容量：每个 TensorCore 有固定数量的 vector 寄存器
   - 物理大小：每个寄存器 **4096 bytes**（固定，与数据类型无关）
   - 实际加载类型（来自 `post-eliminate-llo-extensions.txt` 中的 `llo.vector_load`）：
     - `vector<8×128×i32>` = 8 × 128 × 4 = 4096 bytes
     - `vector<8×128×f32>` = 8 × 128 × 4 = 4096 bytes
     - `vector<8×128×2×bf16>` = 8 × 128 × 2 × 2 = 4096 bytes（bf16 通过维度 2 补齐到与 f32/i32 等宽）
   - 用途：向量 ALU 和 MXU 的操作数/结果

2. VMEM (Vector Memory) — 向量暂存器
   - 物理上是在芯片上的 SRAM（~几十 MB 量级）
   - 用途：tile 的本地存储，算法中反复使用的数据块

3. SMEM (Scalar Memory) — 标量寄存器文件
   - 容量远小于 VMEM
   - 用途：循环计数、地址、标志位

4. HBM (High Bandwidth Memory) — 片外显存
   - 96 GiB / chip
   - 所有 VMEM/HBM 间传输通过 DMA
```

### 4.2 三种方法分析寄存器压力

#### 方法 A：读 `packed-bundles-pre-ra.txt` 和 `packed-bundles-post-ra.txt`（最精确）

`llo/*-packed-bundles-pre-ra.txt` 的头部列出了所有内存分配（allocation），包括大小、地址和用途：

```text
$region0:
  #allocation0 [shape = 's32[]', space=sflag, size = 0x4, offset = 0xc8,
                tag = 'sflag constant byte address 0xc8 - global barrier sync flag']
  #allocation1 [shape = 'u8[4096]{0}', space=hbm, size = 0x1000,
                offset = 0x17afbff000, tag = 'hbm constant byte address ...']
  #allocation6 [shape = 'u32[]', space=smem, size = 0x4,
                offset = 0x4, tag = 'smem constant byte address 0x4 - core index']
  #allocation27 [shape = 'u32[]', space=smem, size = 0x4,
                 offset = 0xffff4, tag = 'smem constant byte address 0xffff4 - program id location']
  ...
```

**对比 pre-ra 和 post-ra**：

| 文件 | 寄存器状态 | 分析用途 |
|------|-----------|---------|
| `packed-bundles-pre-ra` | 虚拟寄存器（%v419_v0） | 看理想 VPR 分配，无 spill |
| `packed-bundles-post-ra` | 物理寄存器 + spill/fill 指令 | **看寄存器压力是否导致 spill** |
| `packed-bundles-no-spills-fills` | 消除 spill/fill 后的 IR | 看是否存在 spill 瓶颈 |

关键分析：

- Post-RA 文件中出现 `VLOAD:FILL` / `VSTORE:SPILL` = 寄存器不足，触发了 spill
- 对比 `pre-ra` 和 `post-ra` 的指令数：增量 = spill/fill 开销
- 在 `final_hlo-static-per-bundle-utilization.txt` 中查看 VLOAD:FILL 和 VSTORE:SPILL 列确认 spill 频率

#### 方法 B：从 `llo/*-memory-space-assignment-allocinfo.txt` 读精确分配

LLO 后端的内存分配文件包含详细的 buffer 分配、偏移量和 size。可以直接读取所有 VMEM allocation 的总大小。

#### 方法 C：从 Mosaic LLO 函数签名读 VMEM 分配

`post-lower-to-llo.txt` 的函数参数直接告诉你 VMEM 的分配情况：

```mlir
func.func @main(
    // === HBM 参数（passthrough_operand，不占 VMEM）===
    %arg0: llo.type = memref<128x2x1024xbf16, ..., #tpu.memory_space<hbm>>  // 输入 tokens
    %arg1: llo.type = memref<32x2048x512xbf16, ..., #tpu.memory_space<hbm>> // 权重 wi_0
    %arg2: llo.type = memref<32x512x2048xbf16, ..., #tpu.memory_space<hbm>> // 权重 wi_1
    ...

    // === SMEM 参数 ===
    %arg10: llo.type = memref<2x32x128xi32, ..., #tpu.memory_space<smem>>   // 路由表
    %arg11: llo.type = memref<2x8x1x256xi32, ..., #tpu.memory_space<smem>>  // expert 索引
    ...

    // === VMEM 参数（关键！）===
    %arg16: llo.type = memref<2x8x16x2x1024xbf16, ..., #tpu.memory_space<vmem>> // 输入 tile buffer
                                                                  // = 2×8×16×2×1024×2 bytes = 1,048,576 bytes ≈ 1 MB
    %arg17: llo.type = memref<2x32x128xf32, ..., #tpu.memory_space<vmem>>      // 评分 buffer
                                                                  // = 2×32×128×4 = 32,768 bytes
    %arg18: llo.type = memref<2x32x128xi32, ..., #tpu.memory_space<vmem>>      // 路由表 buffer
    %arg19: llo.type = memref<2x32x2048xbf16, ..., #tpu.memory_space<vmem>>    // 激活 buffer
    %arg20-22: llo.type = memref<2x2x512x512xbf16, ..., #tpu.memory_space<vmem>> // 权重 tile buffers
    %arg23: llo.type = memref<2x256x1x512xf32, ..., #tpu.memory_space<vmem>>   // 累加 buffer
    %arg24-25: llo.type = memref<...x32x2x512xbf16, ..., #tpu.memory_space<vmem>> // 输出 tile buffers
```

**VMEM 总量估算**：把所有 `vmem` 类型的参数字节数加起来，对比 VMEM 总容量（TPU v7x per-core VMEM ~几十 MB）。如果 VMEM 分配接近或超过容量，编译器会插入 spill（溢出到 HBM），这是性能下降的重要信号。

### 4.3 VPR（向量寄存器）压力分析

每个 `llo.vector_load` 产生一个 VPR 值，后续的向量操作会消费和产生新的 VPR 值。IR 中实际出现的加载类型：

- `vector<8×128×i32>`（路由索引）→ 4096 bytes / 寄存器
- `vector<8×128×f32>`（浮点权重/累加器）→ 4096 bytes / 寄存器
- `vector<8×128×2×bf16>`（MXU 输入）→ 4096 bytes / 寄存器

三种类型均占 **1 个 VPR 寄存器**（4096 bytes）。

**估算 VPR 占用**：

1. 在任何程序点，统计**同时存活**的 `%N` 变量数（live variables）
2. 每个 `llo.vector_load` 结果占 **1 个 VPR 寄存器**
3. 如果 live 变量数超过 VPR 文件大小，编译器必须 spill（写入 VMEM 再读回）

**从 LLO 代码中识别 VPR spill**：

- 如果某段代码中 `vector_load` 的数量异常多于计算所需，可能是在 reload spill
- 如果 `vector_store` 后面紧跟着 load 同一个地址，可能是 spill/fill 对

### 4.4 寄存器压力分析实例

在 `post-lower-to-llo.txt` 第 279 行附近的 index 计算区域：

```mlir
%279 = llo.vector_load %271 : i32 -> vector<8x128xi32>       // VPR[0]: 路由表 chunk 0
%284 = llo.vector_load %271 + %75 : i32 -> vector<8x128xi32>  // VPR[1]: 路由表 chunk 1
%289 = llo.vector_load %271 + %74 : i32 -> vector<8x128xi32>  // VPR[2]: 路由表 chunk 2
%294 = llo.vector_load %271 + %73 : i32 -> vector<8x128xi32>  // VPR[3]: 路由表 chunk 3

// 然后对每个加载的值做 replicate + broadcast，产生大量临时 VPR
%298 = llo.vslreplicate %279, %297  // VPR[4]
%300 = "llo.vbcast_sublane_chunk"(%298, %299)  // VPR[5]
%302 = llo.vslreplicate %279, %301  // VPR[6]
%304 = "llo.vbcast_sublane_chunk"(%302, %303)  // VPR[7]
// ... 持续增加 live 寄存器 ...
```

这段代码中，4 个 load 产生了 4 个基础 VPR，然后每个通过 replicate + broadcast 产生 8× 子 lane 值（共 32 个临时 VPR），加上后续的 compare/select 操作，短期内可能有 40-50 个 VPR 同时存活。如果 VPR 文件只有 32 或 64 个寄存器，就可能触发 spill。

### 4.5 Scalar 寄存器压力

标量操作（`llo.constant`、`llo.sadd` 等）使用标量寄存器。这类压力通常较低，但在以下情况可能成为瓶颈：

- 大量常量（文件开头有 128 个 `llo.constant` 定义，全部维持到函数结尾）
- 深层嵌套循环的迭代变量

**优化线索**：如果看到大量 `llo.constant` 且在后续代码中很少使用，说明编译器未能将常量 fold 到立即数（可能是编译器限制）。

## 5. 硬件利用率与 Fusion 分析

### 5.1 理解 TPU v7x 的 Fusion

Fusion（算子融合）是 XLA/Mosaic 编译器最关键的优化之一。它将多个逻辑算子合并为一个 kernel，消除中间的 HBM 读写。

**Fusion 的效果可以在 IR dump 中追踪**：

| IR 阶段 | 算子形态 | Fusion 程度 |
|---------|---------|-------------|
| HLO `module.mlir` | 独立的 `stablehlo` 算子 | 完全未融合 |
| Mosaic `original` | 粗粒度 `scf.for` + `tpu.enqueue_dma` | Mosaic 前端融合完成 |
| Mosaic `post-lower-to-llo` | 细粒度 `llo.v*` 指令 | 已完全展开为 VLIW |
| LLO `final_bundles` | 物理地址 VLIW bundles | 最终硬件指令 |

通过对比 **HLO 模块** 和 **Mosaic original**，可以看到哪些算子被融合进了同一个 kernel：

```mlir
// HLO（多个独立 kernel）
func @jit_matmul_1() -> ...       // matmul #1
func @jit_activation_fn() -> ...  // 激活函数
func @jit_add_bias() -> ...       // 加 bias
func @jit_matmul_2() -> ...       // matmul #2

// Mosaic original（融合为 1 个 fused_moe kernel，包含所有上述操作）
module @"fused-moe-k_8-renorm_k-bt_32_32_32-..." {
    func.func @main(...)
}
```

### 5.2 直接读 `final_hlo-static-per-bundle-utilization.txt`（推荐）

这是最权威的硬件利用率来源——编译器在代码生成后对每个 VLIW bundle 做静态分析的结果：

```text
== CAPACITY:
MXU, XLU, VALU, EUP, VLOAD, VLOAD:FILL, VSTORE, VSTORE:SPILL, SALU
    2     2     4     1     3     3     2     2     2
== UTILIZATION:
0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 1
0 0 0 0 1 0 0 0 0        ← bundle 3: 1× VLOAD
0 0 0 0 0 0 0 0 1        ← bundle 4: 1× SALU
0 0 1 0 1 0 0 0 1        ← bundle 14: 1× VALU + 1× VLOAD + 1× SALU (3 FU 并行)
0 0 2 0 0 0 0 0 0        ← bundle 17: 2× VALU (Vector 满)
0 0 3 0 0 0 0 0 2        ← bundle 25: 3× VALU + 2× SALU
```

**格式说明**：

| 列 | 硬件单元 | 容量 | 说明 |
|----|---------|------|------|
| MXU | 矩阵乘法单元 | 2 | 可同时执行 2 条 MXU 指令 |
| XLU | 超越函数单元 | 2 | 处理 exp/div/sqrt 等 |
| VALU | 向量 ALU | 4 | 可同时执行 4 条 Vector ALU 指令 |
| EUP | 执行单元池 | 1 | 通用执行槽位 |
| VLOAD | 向量 load | 3 | VMEM→VPR 加载 |
| VLOAD:FILL | spill fill | 3 | 寄存器 reload |
| VSTORE | 向量 store | 2 | VPR→VMEM 写入 |
| VSTORE:SPILL | spill | 2 | 寄存器 spill |
| SALU | 标量 ALU | 2 | 可同时执行 2 条 Scalar ALU 指令 |

**分析方法**：

1. **计算每列的平均利用率**：按列求平均 / capacity
2. **识别瓶颈**：利用率最高的列就是瓶颈 FU
3. **识别气泡**：全 0 行 = 空 bundle（所有 FU 空闲）
4. **查找 spill**：VLOAD:FILL 和 VSTORE:SPILL 列非零 = 该 cycle 发生寄存器 spill/fill

```python
import numpy as np

# 读取 UTILIZATION 段，每行为一个 bundle 的 9 列值
data = np.loadtxt('utilization.txt', skiprows=4)
capacity = np.array([2, 2, 4, 1, 3, 3, 2, 2, 2])

avg_util = data.mean(axis=0) / capacity  # 每列平均利用率

bubble_rate = (data.sum(axis=1) == 0).mean()  # 空 bundle 比例

spill_rate = (data[:, 5].sum() + data[:, 7].sum()) / data.shape[0]  # spill/fill 比例
```

### 5.3 从 Mosaic LLO 分析 MXU 利用率（无 llo/ 文件时的替代方法）

#### 识别 MXU 指令的密度

```mlir
// 高效模式：MXU pipeline 持续运转
"llo.vmatprep.mubr"(%5326) ...   // prep wave 1
"llo.vmatmul.mubr"(%5325) ...    // matmul wave 1
"llo.vmatprep.mubr"(%5330) ...   // prep wave 2（可与 wave 1 的 result 读取并行）
"llo.vmatmul.mubr"(%5329) ...    // matmul wave 2
%5337 = "llo.vmatres"() ...      // 读 wave 1 结果（同时 wave 2 在计算）
%5345 = "llo.vmatres"() ...      // 读 wave 2 结果
```

- 理想情况：`vmatmul` 每个 cycle 都发射一条（持续流水线填充）
- 次优情况：`vmatmul` 之间有大量 `vmatres` + Vector ALU 操作插入（MXU 空闲等待）

#### 估算 MXU bubble 比例

```text
mxu_utilization = mxu_active_cycles / total_cycles

mxu_active_cycles = 有效 matmul 发射次数（一次发射处理 tile 乘法）
total_cycles = 关键区域的总周期数

bubble_rate = 1 - mxu_utilization
```

#### 结合 tile 大小判断 MXU 填充率

从 `vmatmul` 的操作数类型 `vector<8x128x2xbf16>`（8 × 128 × 2 = 2048 个 bf16 元素，共 4096 bytes）可以推算每次 MXU 操作处理的数据量。对比 MXU 的理论峰值（TPU v7x: 每 cycle 可处理 N 个 bf16 MAC），可以计算硬件单元使用率。

### 5.4 分析 Vector ALU 利用率

Vector ALU 处理所有非矩阵的向量操作。对同一段代码：

```mlir
%5337 = "llo.vmatres"() ...         // MXU output
%5338 = llo.vadd.f32 %124, %5337    // Vector ALU: 累加到初始化零
%5339 = "llo.vmatres"() ...         // MXU output
%5340 = llo.vadd.f32 %124, %5339    // Vector ALU
// ... 重复 4 次（对应 4 路累加）...

%5356 = "llo.vmatres"() ...         // 第二波 MXU output
%5357 = llo.vadd.f32 %5338, %5356   // Vector ALU: 累加到之前的 partial sum
```

这段代码中 Vector ALU 在 MXU `vmatres` 之后交替执行，利用率取决于 MXU pipeline 的延迟。

**分析关键**：

- 如果你看到连续的 `vmatres`（读 MXU 结果）和 `vadd`（累加），却没有新的 `vmatprep/vmatmul`，说明 MXU 在等待数据（可能是 DMA 未完成）
- 如果你看到大量 `vexp`、`vdiv`、`vmax`（激活函数）而 MXU 空闲，说明是 vector-bound 区域

### 5.5 分析 Scalar ALU 利用率

Scalar ALU 承担循环控制、地址计算、条件判断：

```mlir
// 地址计算链（Scalar ALU 密集）
%133 = llo.sdiv.u32 %131, %132      // 1 cycle
%135 = llo.srem.u32 %131, %134      // 1 cycle（可与上一行并行如果无依赖）
llo.vsync.add.remote %arg34, ...    // 同步（Scalar ALU）
%137 = llo.sld %arg35 + %136        // 从 SMEM 加载（Scalar Load, 1 cycle）
%139 = llo.sdiv.u32 %137, %138      // 1 cycle
```

**标记 Scalar 瓶颈**：当前分析结果中，GLA backward kernel 的 Scalar ALU 平均利用率为 34%（MXU 仅 11.6%），说明 Pallas kernel 的控制流（`fori_loop`/`lax.cond`）产生大量标量运算，MXU 处于饥饿状态。

### 5.6 计算/访问比与 Bottleneck 分类

从 IR dump 中的指令分布可以直接估算 **Compute vs Memory** 瓶颈：

| 指令类型 | 对应硬件活动 | 瓶颈信号 |
|---------|-------------|---------|
| `llo.vector_load` | VMEM → VPR | 连续大量 load 无足够计算 → Memory bound |
| `llo.vector_store` | VPR → VMEM | store 峰值接近带宽上限（Vector Store max 97%） → Write BW bound |
| `llo.vmatmul` | MXU 计算 | vmatmul 稀疏穿插在标量/向量指令间 → MXU 饥饿 |
| `llo.enqueue_dma` | HBM ↔ VMEM | DMA 数量多且 `dma_done` 等待时间长 → HBM BW bound |

### 5.7 DMA/计算 Overlap 分析

在 `post-finalize-llo.txt` 中，DMA 操作有显式的优先级和 strides：

```mlir
"llo.enqueue_dma"(%79, %21, %80, %81, %82)
    <{priority = 0 : i32, src_strides = ..., dst_strides = ...}>
```

**分析 overlap 的方法**：

1. 找到 `enqueue_dma` 的位置（数据开始传输）
2. 找到对应的 `dma_done` 位置（等待数据到位）
3. 检查两者之间的指令数：
   - 如果之间有大量计算指令 → DMA 延迟被有效掩盖
   - 如果直接紧邻 `dma_done` → 无 overlap，核在等待数据
4. 估算 DMA 时间：`data_size / HBM_bandwidth`，对比 overlap 区域的计算时间

### 5.8 实战：分析一个 Matmul 序列的利用率和瓶颈

以下来自 `post-lower-to-llo.txt` 第 7958 行的 MoE matmul 核心区域：

```mlir
// ── Wave 1: 加载权重 tile，启动 MXU ──
"llo.vmatprep.mubr"(%5326) ...      // prep 左矩阵 (权重 tile wi_0)
"llo.vmatmul.mubr"(%5325) ...       // matmul 执行 (激活 × 权重)
%5337 = "llo.vmatres"() ...         // 读结果 [0]
%5338 = llo.vadd.f32 %124, %5337    // 累加 = 0 + result[0]
%5339 = "llo.vmatres"() ...         // 读结果 [1]
%5340 = llo.vadd.f32 %124, %5339    // 累加 = 0 + result[1]  (独立，可与上面并行)

// ── Wave 2: 下一个权重 tile，前一个结果累加 ──
"llo.vmatprep.mubr"(%5330) ...      // prep 左矩阵 (权重 tile wi_1)
"llo.vmatmul.mubr"(%5329) ...       // matmul 执行
%5356 = "llo.vmatres"() ...         // 读结果 [0]
%5357 = llo.vadd.f32 %5338, %5356   // 累加 = partial_sum + new_result[0]
```

**分析**：

- **MXU 利用率**：每波 2 次 matmul（wi_0 + wi_1），之间读取 4 个 result。如果 MXU pipeline depth 长，`vmatres` 前的等待是 MXU bubble
- **Vector ALU**：每波 4 个 `vadd` 做累加（2 个清零 + 2 个累加）。如果 Vector ALU 跟不上 MXU 产出，会成为瓶颈
- **寄存器压力**：`%5338` 存活时间跨两波 MXU（作为 partial sum），至少 2 个 VPR 长期占用
- **Fusion 效果**：wi_0 和 wi_1 两个权重 tile 的 matmul 被融合在同一段代码中，partial sum 在 VMEM 中累加，避免了写回 HBM

---

## 6. 数据布局变换开销分析

TPU 的 VMEM 采用 **tiled 布局**——数据按 sublane 分布在向量寄存器的不同 lane 上。MXU 矩阵乘法单元、向量 ALU、DMA 引擎各自期望不同的数据排列方式，导致编译器必须在算子之间插入大量 **布局变换指令**（layout transformation）。这些指令不产生有效计算，却消耗 Vector ALU 的指令槽，是 TPU kernel 性能损耗的重要来源。

### 6.1 布局变换的三级展开

布局变换在编译流水线中经历三级展开，每级指令数量呈 1:N 膨胀：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Mosaic Pass 0009 (post-relayout-insertion)                         │
│   tpu.relayout %x {in_layout=vpad<...>, out_layout=vpad<...>}     │
│   — 高级布局标注，每条 relayout 对应一次逻辑布局转换                    │
│   — 典型数量：~101 条（fused-MoE bt_8_8_8）                         │
├─────────────────────────────────────────────────────────────────────┤
│ Mosaic Pass 0010 (post-apply-vector-layout)                        │
│   tpu.sublane_shuffle %a, %b, [0,1,2,3,8,9,10,11]                 │
│   tpu.gather %x[[0,2,4,6,1,3,5,7]] in 0                           │
│   tpu.roll_vectors %a, %b, shift=N                                 │
│   — 1 条 relayout → 多条 sublane_shuffle + gather                   │
│   — 典型数量：~1152 条 sublane_shuffle + ~100 条 roll_vectors       │
├─────────────────────────────────────────────────────────────────────┤
│ LLO Final Bundles (pass 69)                                        │
│   vrot.slane, vcombine.high/.low, vunpack.*, vpack.*,              │
│   vbcast.lane.xlu0/.xlu1, vpop.permute.xlu0/.xlu1, vstv          │
│   — 硬件级 VLIW 指令，与 MXU/Scalar 指令打包在同一 bundle             │
│   — 典型数量：~2451 条（fused-MoE bt_32_32_32）                     │
└─────────────────────────────────────────────────────────────────────┘
```

**关键数据**（fused-MoE kernel bt_8_8_8 为例）：

| 阶段 | 指令类型 | 数量 | 膨胀倍数 |
|------|---------|------|---------|
| Pass 0009 | `tpu.relayout` | 101 | 1× |
| Pass 0010 | `tpu.sublane_shuffle` | 1152 | 11.4× |
| Pass 0010 | `tpu.roll_vectors` | 460 → 100（simplify 后） | — |
| Pass 69 | VLIW layout 指令 | 792 | 7.8×（相对 relayout） |

### 6.2 五类布局变换指令

#### a) Sublane 旋转（`vrot.slane`）— 最大开销来源

**用途**：在 MXU operand 加载前，将 VMEM 的 tiled 布局旋转到 MXU 期望的 sublane 排列。MXU 的 `vmatprep` 要求操作数按特定 sublane 顺序分布，而 VMEM load 的数据可能分布在不同的 sublane 上。

**开销**：在 fused-MoE bt_32_32_32 中占 **1033 条**（42% 的布局指令）。

```text
// 典型模式：load → rotate → MXU prep
0x17 : >>> { %5951 = vmatprep.subr.bf16.mxu0 %v4661 ;; %6057 = vmatprep.subr.bf16.mxu1 %v4663
             ;; %v4609 = vld [vmem:[%s4603 + $0x14] sm:$0xf]
             ;; %v4850 = vrot.slane %v4606, %v644        ← sublane 旋转
             ;; %v4874 = vrot.slane %v4607, %v644        ← sublane 旋转
             ;; %v4898 = vrot.slane %v4608, %v644 }      ← sublane 旋转
```

`vrot.slane` 的第二个操作数（`%v644`）是旋转偏移量，通常在循环外预计算。注意它与 `vmatprep` 打包在同一 bundle 中——编译器试图将旋转与 MXU 准备并行执行，但旋转仍消耗 Vector ALU 槽。

#### b) Tile 分解（`vcombine.high` / `vcombine.low`）

**用途**：当向量寄存器持有大 tile（如 `vector<8x128xi32>`）而后续操作需要子 tile 时，`vcombine` 提取高半或低半。

**开销**：在 bt_32_32_32 中占 **776 条**（32%）。

```text
0x18 : >>> { %v4665 = vld [vmem:[#allocation12 + $0xe8] sm:$0xff]
             ;; %v4803 = vcombine.high %v4802, %v4802     ← 提取高半
             ;; %v4827 = vcombine.high %v4826, %v4826     ← 提取高半
             ;; %v4922 = vrot.slane %v4609, %v644
             ;; %v4946 = vrot.slane %v4610, %v644 }
```

`vcombine` 通常成对出现（high + low），将一个大寄存器拆成两个子 tile，分别送入 MXU 的两个 pipeline（mxu0/mxu1）。

#### c) 数据类型转换 + 布局变换（`vunpack` / `vpack`）

**用途**：bf16 ↔ f32 类型转换时，数据在 sublane 中的打包方式发生变化。`vunpack` 将 bf16 对展开为 f32（sublane → lane 扩展），`vpack` 将 f32 压缩回 bf16（lane → sublane 压缩）。

**开销**：在 bt_32_32_32 中占 **501 条**（448 unpack + 53 pack，20%）。

```text
// bf16 → f32 解包（为 MXU 累加做准备）
%v_a = vunpack.c.l.b16 %v_bf16_data     // 低位 bf16 → f32
%v_b = vunpack.c.h.bf16 %v_bf16_data     // 高位 bf16 → f32

// f32 → bf16 打包（写回 VMEM 前）
%v_result = vpack.c.bf16 %v_f32_a, %v_f32_b   // 两个 f32 → bf16 对
```

#### d) Lane 广播（`vbcast.lane` + `vpop.permute`）

**用途**：将单个 sublane 的值广播到整个 tile 行。MXU 的 broadcast 模式（`mubr`/`subr`）要求操作数在特定维度上是广播的，需要先用 `vbcast.lane` 扩展数据。

**开销**：在 bt_32_32_32 中占 **134 条**（16 broadcast + 118 pop.permute，5%）。

```text
// XLU0/XLU1 配对广播（编译器均匀分配到两个 XLU）
%v_bc0 = vbcast.lane.b32.xlu0 %v_scalar    // 通过 XLU0 广播
%v_bc1 = vbcast.lane.b32.xlu1 %v_scalar    // 通过 XLU1 广播
%v_perm0 = vpop.permute.xlu0 ...            // XLU0 置换分发
%v_perm1 = vpop.permute.xlu1 ...            // XLU1 置换分发
```

`vbcast.lane` 总是以 **xlu0/xlu1 配对**出现（各 8 条），编译器将广播负载均匀分配到两条 XLU pipeline。

#### e) 标量广播（`vstv`）

**用途**：将标量值（如常量、RNG seed）splat 到向量所有 lane。开销较小，通常出现在初始化阶段。

### 6.3 不同 Tile 大小的开销对比

以 fused-MoE kernel 为例，布局指令随 tile 大小的变化：

| Tile Size (bt) | 布局指令 | 计算指令 | 访存指令 | 布局/计算比 | 总 Bundle 数 |
|---|---|---|---|---|---|
| 2×2×2 | 273 | 72 | 794 | **3.8×** | — |
| 4×4×4 | 450 | 151 | 893 | **3.0×** | — |
| 8×8×8 | 792 | 256 | 1001 | **3.1×** | — |
| 16×16×16 | 1464 | 491 | 1289 | **3.0×** | — |
| 32×32×32 | 2451 | 706 | 1743 | **3.5×** | 7469 |

**关键观察**：

- 布局指令是 **计算指令的 3.0–3.8 倍**——这是最显著的开销
- 开销随 tile 大小近似线性增长（273 → 2451）
- 小 tile（2×2×2）的布局/计算比反而最高（3.8×），因为小 tile 的 MXU 计算量小但布局变换的固定开销不变
- bt_32_32_32 的 7469 个 bundle 中，56 个为空 bundle（0.75% 空泡率），说明调度器有效利用了布局指令与计算指令的并行性

### 6.4 `vpad` 布局标注语义

Mosaic pass 0009 的 `tpu.relayout` 使用 `#tpu.vpad<>` 标注描述源和目标布局：

```text
vpad<"N,{replicate_flags},(tile_rows,tile_cols)[,interleave]">
```

| 字段 | 含义 | 示例 |
|------|------|------|
| `N` | 向量寄存器的 sublane 数 | `32` = 32 sublane |
| `{replicate_flags}` | 哪些维度是广播（`*`）而非实际数据（`0`） | `{*,*}` = 全广播，`{0,0}` = 全实际 |
| `(tile_rows, tile_cols)` | tile 大小（sublane 行 × 列） | `(8,128)` = 8 行 128 列 |
| `,interleave` | 交错模式（负数 = 奇偶分离） | `-2` = 奇偶交错 |

**常见 relayout 模式**（fused-MoE bt_32_32_32）：

| 源布局 | 目标布局 | 含义 | 次数 |
|--------|---------|------|------|
| `vpad<"32,{*,*},(8,128)">` | `vpad<"32,{0,0},(1,128)">` | 广播 → 实际数据，tile 8→1 行 | 25 |
| `vpad<"32,{*,*},(8,128)">` | `vpad<"32,{0,0},(8,128)">` | 广播 → 实际数据，tile 不变 | 12 |
| `vpad<"32,{*,*},(8,128)">` | `vpad<"32,{0,0},(2,128)">` | 广播 → 实际数据，tile 8→2 行 | 10 |
| `vpad<"32,{0,0},(8,128)">` | `vpad<"32,{0,0},(1,128)">` | tile 缩小 8→1 行 | 9 |
| `vpad<"16,{0,0},(2,128),-2">` | `vpad<"16,{0,0},(16,128)">` | 奇偶交错 → 线性化 | 8 |

`{*,*}` → `{0,0}` 的转换最多（25 次），对应 MXU broadcast 模式需要将"逻辑广播"的常量/权重展开为"物理实际"的 sublane 数据。

### 6.5 relayout 的 sublane 展开过程

一条 `tpu.relayout` 在 pass 0010 展开为多条 `tpu.sublane_shuffle` + `tpu.gather`：

```mlir
// Pass 0009: 一条 relayout
%40 = tpu.relayout %38 {in_layout = [#tpu.vpad<"32,{0,*},(8,128)">],
                         out_layout = [#tpu.vpad<"32,{0,0},(8,128)">]}
    : vector<32x8x256xi32> -> vector<32x8x256xi32>

// Pass 0010: 展开为多条 sublane 操作
%675 = tpu.sublane_shuffle %673, %674, [0, 1, 2, 3, 8, 9, 10, 11]
%676 = tpu.sublane_shuffle %673, %674, [4, 5, 6, 7, 12, 13, 14, 15]
%677 = tpu.gather %675[[0, 2, 4, 6, 1, 3, 5, 7]] in 0
%678 = tpu.gather %676[[0, 2, 4, 6, 1, 3, 5, 7]] in 0
%679 = tpu.sublane_shuffle %677, %677, [4, 5, 6, 7, 12, 13, 14, 15]
```

**展开比例**：1 条 relayout → 2 条 sublane_shuffle + 2 条 gather + 1 条 sublane_shuffle = **5 条硬件操作**。这解释了 101 条 relayout → 1152 条 sublane_shuffle 的 11.4× 膨胀。

### 6.6 XLU 双通道分配

TPU v7x 有两条 XLU（Cross-Lane Unit）pipeline：XLU0 和 XLU1，分别操作向量寄存器文件的两半。编译器将布局变换均匀分配到两条 XLU：

```text
// bt_32_32_32 中的 XLU 配对模式
vbcast.lane.b32.xlu0  × 8     // XLU0 广播
vbcast.lane.b32.xlu1  × 8     // XLU1 广播（配对）
vpop.permute.xlu0     × 59    // XLU0 置换
vpop.permute.xlu1     × 59    // XLU1 置换（配对）
```

XLU 分配在 MXU assigner pass（pass 14）自动完成。两条 XLU 可以在同一 bundle 中并行执行不同的布局操作。

### 6.7 优化建议

基于布局开销分析，以下是减少布局变换开销的方向：

1. **Tile 大小选择**：小 tile（2×2×2）的布局/计算比最高（3.8×），大 tile（16×16×16）相对较低（3.0×）。在寄存器压力允许的情况下，优先使用大 tile
2. **减少 relayout 次数**：`{*,*}` → `{0,0}` 的广播展开是最大的 relayout 来源（25 次）。如果能在算法层面避免频繁的 broadcast ↔ actual 切换，可以减少 relayout
3. **利用 XLU 并行**：编译器已自动做 XLU 配对分配，但手动检查 `vbcast.lane` 是否在两条 XLU 上均衡分布，可以发现调度不均的瓶颈
4. **关注 Mosaic pass 0009**：`tpu.relayout` 的数量和模式直接决定了最终的布局开销。在 Mosaic 层面优化 tiling 策略（减少 `vpad` 转换次数）比在 LLO 层面优化更有效
5. **奇偶交错布局**：`vpad<"16,{0,0},(2,128),-2">` → `vpad<"16,{0,0},(16,128)">` 的奇偶交错 → 线性化转换需要额外的 `vcombine` 操作。如果 MXU 和 DMA 使用相同的奇偶布局，可以消除这类转换

---

## 参考资料

- [性能理论分析框架](./analysis-framework) — Roofline Bound + System Bound 两级分析
- [理论分析结果](./analysis-results) — 128 TPU v7x 配置的 MFU 预测
- [性能优化工作拆解](./work-breakdown) — 优化项优先级与 profiling 结果
