# 硬件约束与 Pallas API 限制

本文系统性记录 TPU 硬件特性和 Pallas API 的硬性约束，是所有 pallas-kernel 开发者的必读参考。

通用的 TPU 性能优化背景知识请参阅 [TPU 性能优化指南](/onboarding/performance-tpu)，本文聚焦于 Pallas kernel 开发中的具体约束。

---

## 1. TPU 硬件架构概要

### 1.1 计算单元

TPU Core 内部包含多个专用计算单元，各有不同的能力和约束：

| 单元 | 功能 | 关键特性 |
|------|------|----------|
| **MXU** (Matrix Multiply Unit) | 矩阵乘法 (`jax.lax.dot`) | 128×128 systolic array（TPUv6e 及以后为 256×256），是算力核心 |
| **Vector Unit** | 逐元素运算（加减乘除、比较、mask） | 向量计算主力 |
| **EUP** | 非线性算子（`exp`、`log`） | 集成在 Vector Unit 内部 |
| **XLU** (Transpose Unit) | 跨 lane shuffle、transpose、permutation | 数据重排 |
| **Scalar ALU** | 标量计算、索引计算、控制流 | **顺序发射，极易成为瓶颈** |
| **DMA Engine** | HBM ↔ VMEM/SMEM 数据搬运 | 异步执行 |

**核心约束**：Scalar ALU 是顺序发射的。过多的索引计算和控制流会导致 Scalar ALU 成为瓶颈，即使 MXU 和 Vector Unit 空闲。

### 1.2 MXU 维度对齐

MXU 是 128×128 的脉动阵列（Systolic Array，TPUv6e 及以后为 256×256），对输入数据有严格的对齐要求：

- 张量维度对齐要求：
  - TPUv6e 及以后：建议 **256 的倍数** 以获得最佳 MXU 效率
  - TPUv6e 之前：建议 **128 的倍数**
  - 硬件底层的最小对齐单位为 **8**（sublane）和 **128**（lane）的倍数
- 不对齐的维度会被硬件自动 padding 到最近的对齐边界，浪费计算和带宽
- 对于 MatMul `[M, K] × [K, N]`，推荐 **M ≥ 512, K ≥ 1024, N ≥ 512**

### 1.3 内存层次

```text
HBM (High Bandwidth Memory)
  │  容量大（GB 级），带宽相对低
  │  DMA 异步搬运
  ▼
VMEM (Vector Memory / On-chip SRAM)
  │  容量有限（MB 级），带宽极高
  │  MXU 和 Vector Unit 直接访问
  │  128-byte 对齐
  ▼
SMEM (Scalar Memory)
     容量极小，用于索引查找表等标量数据
     Scalar ALU 直接访问
```

### 1.4 DMA 异步数据搬运

TPU 掩盖内存延迟的唯一武器是**软件流水线**（Software Pipelining）：

- 没有硬件级线程切换（无 Warp Scheduler）
- 全靠编译器或开发者手动安排指令，在 MXU 计算第 `i` 块时，DMA 异步加载第 `i+1` 块
- 循环展开（Unroll）的真正意义：腾出足够的"指令槽位"来排布异步加载指令

### 1.5 寄存器约束

TPU 的寄存器分为两类：向量寄存器（VREG）和标量寄存器（SREG）。完整的数据通路为 **HBM ⇔ VMEM ⇔ VREG ⇔ MXU/VPU**。

#### VREG（Vector Register）

每个 VREG 是一个 **(8, 128)** 的二维寄存器，存储 32-bit 值，单个 VREG 大小为 8 × 128 × 4 = **4 KiB**。

| 属性 | TPU v4 | TPU v5p |
|------|--------|---------|
| **VREG 数量（每核）** | 32 | 64 |
| **VREG 总容量（每核）** | ~128 KiB | ~256 KiB |
| **VMEM 读带宽** | — | 3 VREGs/cycle |
| **VMEM 写带宽** | — | 1 VREG/cycle |

> TPU v6e 的 VREG 数量未公开披露，VREG tile shape 仍为 (8, 128)。

#### SREG（Scalar Register）

SREG 供 Scalar ALU 使用，存储标量值（索引、控制流变量等）。容量极小，由 Scalar Unit 直接访问。

#### VREG 对齐要求

对 VMEM 的读写以 **(8, 128)** tile 为单位进行。为获得最佳性能：

- 最后两维的基址偏移应分别是 **8** 和 **128** 的倍数
- 读写区域的大小也应是 tile 尺寸的整数倍
- 一维张量占据整个 (8, 128) tile——两个 1×1 数组的加法开销等同于两个 8×128 数组

#### 寄存器溢出（Register Spilling）

VREG 容量有限，当 kernel 中同时活跃的变量超出 VREG 容量时，编译器会将多余的寄存器**溢出到 VMEM**，引入额外的 load/store 延迟。

**缓解策略**：

- **减少活跃变量数**：缩小同时持有的中间结果，及时释放不再使用的张量
- **动态切片加载**：对较大的 Ref 使用动态切片（`ref.at[slice]`）按需加载，避免一次性将过多数据读入 VREG 导致寄存器压力过大
- **注意 padding 开销**：向量计算会将不足的维度 pad 到 (8, 128)，浪费的元素同样占用 VREG

---

## 2. Pallas API 硬性约束

### 2.1 `make_async_copy`

```python
# 正确用法
copy = pltpu.make_async_copy(
    src_ref.at[(slice_0, slice_1)],  # 必须用 .at 取引用
    dst_scratch_ref.at[buf],
    sem_ref.at[0, buf],              # 必须提供信号量
)
copy.start()
# ... 其他计算（掩盖 DMA 延迟）...
copy.wait()
```

**硬性约束**：

- **必须**提供一个 semaphore（信号量）
- 参数**必须**是 ref，在 kernel 函数中用 `.at` 获取引用
- async_copy 的 handler **不能**作为 `fori_loop` 的参数或返回值传递
- DMA 的 wait 是按**槽位（slot）**计算的：多次 start 可共享一个 wait

### 2.2 条件执行

```python
# 推荐：pl.when
@pl.when(i + 1 < NT)
def do_prefetch():
    ...

# 不推荐：lax.cond（引入额外 Scalar ALU 开销）
lax.cond(i + 1 < NT, prefetch_fn, lambda _: None, None)
```

**`pl.when` 比 `lax.cond` 快**，用于 prefetch 等操作时应优先使用。

### 2.3 BlockSpec

- `memory_space` 默认不指定时数据放在 VMEM
- 指定为 `pltpu.HBM` 时，ref 持有**全部 shape** 的数据（需手动管理搬运，可能导致 OOM）
- `pipeline_mode`：不支持 lookahead；`buffer_count` 仅支持 1 或 2

### 2.4 Grid 与 dimension_semantics

```python
grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV))
dimension_semantics = ("parallel", "parallel", "arbitrary", "arbitrary")
```

- Grid 参数**必须为编译时常量**，不可动态计算
- `"parallel"`：维度间无数据依赖，允许编译器在核间并行调度
- `"arbitrary"`：维度间可能有依赖（如序列维度的累加），顺序执行
- **Batch 维度放 Grid 比在 kernel 内部用 `lax.fori_loop` 展开要快**

### 2.5 控制流与循环

- `kernel_body` 中**禁止使用 Python 动态控制流**
- 用 Python for 循环让循环变量成为编译时常量：

```python
def kernel_body(...):
    carry = init_carry
    for i in range(NT):
        buf = i % 2           # Python int，编译时常量
        next_buf = (i + 1) % 2
        ...
```

- `fori_loop` 中只用循环变量 `i` 计算索引，不引入其他变量

### 2.6 BlockSpec 自动管理 vs 手动 DMA

**核心结论：规则访问模式下，BlockSpec 自动管理几乎总是优于手动 DMA。**

| 方案 | 典型耗时 | 说明 |
|------|---------|------|
| BlockSpec 自动管理 | ~19μs | 编译器自动 double buffer |
| 手动 DMA（单 grid） | ~26μs | 循环索引管理引入 Scalar ALU 开销 |

**什么时候才值得手动 DMA**：

- 运行时数据依赖的访问模式（如 `cu_seqlens` 变长序列）
- 跨核通信场景
- 不规则/非矩形访问模式

---

## 3. 内存对齐规则

### 3.1 VMEM 对齐

- VMEM 要求 **128-byte 对齐**
- 矩阵维度需为 **128 的倍数**
- 不满足时需要在 host 侧做 padding，kernel 内部处理 padding 后的维度

### 3.2 Padding 策略

```python
def _pad_axis(x: jax.Array, axis: int, multiple: int) -> jax.Array:
    """将指定轴 pad 到 multiple 的倍数。"""
    size = x.shape[axis]
    remainder = size % multiple
    if remainder == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, multiple - remainder)
    return jnp.pad(x, pad_width)
```

### 3.3 SMEM 用途

SMEM 容量极小，主要用于存放**索引查找表**（替代 Scalar ALU 的整数除法/取模运算）：

```python
# BlockSpec 指定 SMEM
in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))

# Kernel 内部直接读表，避免 Scalar ALU 计算
b_i = idx_ref[i, 0]
h_i = idx_ref[i, 1]
```

---

## 4. 性能陷阱

### 4.1 Scalar ALU 瓶颈

Scalar ALU 是最常见的性能瓶颈。以下操作会在 Scalar ALU 上执行，应尽量避免：

- 索引的整数除法和取模运算
- 动态 buffer index 计算（如 `buf = jnp.mod(i_t, 2)`）
- `lax.cond` 引入的控制流开销

**优化手段**：

1. **SMEM 索引查找表**：将索引预计算放入 SMEM

```python
# Host 端预计算
flat_idx = jnp.arange(total + 1, dtype=jnp.int32)
b_tab = flat_idx // h_sum_v
h_tab = (flat_idx % h_sum_v) // NK_sum_v
index_table = jnp.stack([b_tab, h_tab, k_tab, v_tab], axis=1)
```

1. **位打包（Bit Packing）**：将多个小索引打包进一个 int32

```python
# 打包
bh_packed = (b_tab << 16) | h_tab

# 解包
b_i = bh >> 16
h_i = bh & 0xFFFF
```

1. **Unroll 消除动态计算**：让 `i` 成为编译时常量

### 4.2 Prefetch Pipeline

标准 Double Buffer + Prefetch 模式：先发起下一次 prefetch，再 wait 当前数据，最后计算。

```python
# 假设 async_copy 是对 pltpu.make_async_copy 的封装
# 1. 启动下一次迭代的 prefetch
@pl.when(i + 1 < NT)
def do_prefetch():
    copy_next.start()

# 2. Wait 当前迭代的数据
copy_cur.wait()

# 3. 计算
tile = scratch_ref[buf]
result = jax.lax.dot(tile, ...)
```

**注意**：BlockSpec 自带 double buffer 机制，手动实现往往不如编译器自动优化。仅在上述"值得手动 DMA"的场景中使用。

### 4.3 其他注意事项

- **OOM 风险**：`memory_space=pltpu.HBM` 使 ref 持有全部数据，`batch_size × T` 大时可能 OOM
- **VMEM 限制**：可通过 `CompilerParams(vmem_limit_bytes=128 * 1024 * 1024)` 控制上限
- **边界检查**：手动 DMA 时可设置 `disable_bounds_checks=True` 减少开销
- **输出异步**：中间状态写回 HBM 也可以用 DMA async，避免阻塞计算流水线
- **信号量管理**：多种数据类型的 DMA 建议使用二维信号量 `pltpu.SemaphoreType.DMA((N, 2))` 统一管理
