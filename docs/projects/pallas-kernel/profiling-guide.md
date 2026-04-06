# Profiling 深度指南

> 本文基于 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 系列整理，补充现有 [Benchmark 规范](./benchmarking) 中的 profiling 部分，聚焦于如何读懂 profile、分析 HLO、定位瓶颈。

---

## 1. TPU 软件栈

```text
JAX → StableHLO → HLO → LLO → 机器码
      (jax.jit)    (XLA 编译器决定 fusion, layout)   (直接编程 TPU 硬件)
```

- **HLO**：工程师分析性能的主要层级
- **LLO**：调度 VMEM 拷贝、驱动 systolic array，开发者通常不直接操作
- **Pallas**：当需要比 HLO 更底层的控制时使用的 custom kernel 方案

### 1.1 获取 HLO

```python
hlo_text = jax.jit(f).lower(*args, **kwargs).compile().as_text()
```

---

## 2. Profiler 使用

### 2.1 基础 Profiling

```python
import jax

with jax.profiler.trace("/tmp/tensorboard"):
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024, 1024))
    y = x @ x
    y.block_until_ready()

# 查看结果
# tensorboard --logdir=/tmp/tensorboard
```

### 2.2 Kernel Profiling 最佳实践

```python
# 1. 预热（确保 JIT 编译完成）
output = kernel(...)
jax.block_until_ready(output)

# 2. 在顶层 jit 包裹下 profiling
jax.profiler.start_trace("/path/to/profile")
for _ in range(3):
    output = kernel(...)
    jax.block_until_ready(output)
jax.profiler.stop_trace()
```

> **关键**：profiling 必须在顶层 `jax.jit` 包裹下运行，否则产生碎片化 HLO 模块，导致 xprof 分析失效。

### 2.3 使用 `jax.named_scope` 标注

```python
with jax.named_scope("attention"):
    q = x @ w_q
    k = x @ w_k
    attn = softmax(q @ k.T)

with jax.named_scope("ffn"):
    y = attn @ w_v
    y = y @ w_out
```

标注后在 Trace Viewer 中可清晰识别各组件对应的时间段。

---

## 3. 读懂 XLA Op

### 3.1 HLO 操作解析

示例：

```text
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(
    bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32),
    kind=kCustom, calls=%all-reduce-scatter.3
```

| 组件 | 含义 |
|------|------|
| `fusion.3` | Op 名称（fusion 包含至多 1 个 matmul + 相关 VPU 逐元素运算） |
| `bf16[32,32,4096]` | 输出 dtype 和 shape |
| `{2,1,0}` | 内存中的维度排列顺序（右到左读取，映射到逻辑维度） |
| `T(8,128)(2,1)` | Tiling：外层 8×128，内层 2×1 |
| `S(1)` | 内存位置：S(0)=HBM, **S(1)=VMEM**, S(2)/S(3)=其他 |
| `kind=kCustom` | 操作类型 |

### 3.2 Tiling 和 Layout 标记

对于 `f32[3,5]{1,0:T(2,2)}`：

- `{1,0}`：物理内存中的维度顺序
- `T(2,2)`：数组被 2×2 tiling，行主序排列
- Padding：shape [3,5] 被 pad 到 [4,6]，内存扩大 ~1.6×

对于 bf16 数组，双层 tiling `T(8,128)(2,1)`：

- 外层 8×128 tiling（匹配 VREG shape）
- 内层 2×1 确保 load 总是 4 字节对齐

> **性能影响**：Tiling 影响加载效率。XLA 可能插入 retile 拷贝，有时开销不可忽略。可用 `AUTO` layout 让 XLA 自动选择最优 layout。

### 3.3 内存位置标记

| 标记 | 位置 |
|------|------|
| S(0) 或无标记 | HBM |
| S(1) | VMEM |
| S(2) | SMEM |
| S(3) | 其他 |

---

## 4. 关键分析视图

### 4.1 Trace Viewer

最有用的视图，显示每个 TPU 核上所有操作的时间线。

**使用技巧**：

- 通常只看 TPU:0（所有 TPU 执行相同指令）
- 顶行（XLA Ops）显示真实 TPU 操作
- 其他行是近似的 Python 栈追踪
- WASD 控制：A/D 左右平移，W/S 缩放
- 点击 XLA op 可查看源码位置和 Graph Viewer 链接

### 4.2 Graph Viewer

可视化 HLO 计算图，显示操作之间的数据流和 sharding 模式。悬停节点可查看源码行号。

### 4.3 Memory Profile

显示程序内存随时间的变化，用于调试 OOM。

---

## 5. 计算预期耗时

### 5.1 Compute-bound 操作

对于 FFW 上投影，8 TPUv2 核，4 路 DP，2 路 MP：

```text
真实 shape: bf16[32, 1024, 8192] × bf16[8192, 32768]
每分片:     bf16[8, 1024, 8192] × bf16[8192, 16384]
每 DP 分片 batch: 8 × 1024 = 8192 → 远超临界值，compute-bound

预期: 2 × 32 × 1024 × 8192 × 32768 / (23e12 × 8) = 95.6ms
实测: 96ms → MFU ≈ 100%
```

### 5.2 Communication 操作

对于 ReduceScatter，TPUv2 4×2 拓扑：

```text
数组: 2 × 32 × 1024 × 8192 bytes，batch 分 4 片 → 128MB/分片
单跳: 1.2e11 B/s 双向带宽

预期: ~1.1ms
实测: 1.13ms → 接近峰值
```

### 5.3 从 Profile 推断 Sharding

AllReduce 中的 <code v-pre>replica_groups={{0,16,32,48,...}, ...}</code> 可以推断 model parallelism 度。每分片 shape 乘以并行度即可得到真实张量维度。

---

## 6. 诊断问题清单

分析 Transformer profile 时，尝试回答：

| 问题 | 如何回答 |
|------|---------|
| 使用了什么 sharding 策略？ | 查看 AllReduce/AllGather 的 replica_groups |
| batch size, d_model, d_ff 是多少？ | 从 matmul shape 和分片数反推 |
| attention vs MLP 的时间占比？ | 在 Trace Viewer 中对比两段时间 |
| 每个 op 的 roofline 预期耗时？ | 用公式计算，与实测对比 |
| 有无不必要的通信？ | 检查是否有本可避免的 AllGather |
| 有无 retile 拷贝？ | 查看 `copy` op 的耗时占比 |

---

## 7. 常见性能问题

### 7.1 Blocking AllGather

表现：profile 中 AllGather 占 50%+ 时间。

原因：Shardy 编译器未能将 AllGather 与计算重叠。

解决：添加 `jax.lax.with_sharding_constraint` 约束中间张量分片：

```python
def matmul(x, Win, Wout):
    hidden = jnp.einsum('bd,df->bf', x, Win)
    hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
    return jnp.einsum('bf,df->bd', hidden, Wout)
```

### 7.2 Retile 开销

表现：profile 中出现大量 `copy` 操作。

原因：相邻操作需要不同的 tensor layout，XLA 插入格式转换。

解决：考虑用 `jax.jit` 的 `in_shardings` 设置 `AUTO` layout。

### 7.3 MXU 利用率低

表现：matmul 耗时远超 roofline 预期。

可能原因：

- Tile 尺寸太小（未填满 MXU 128×128 / 256×256）
- 非对齐维度导致大量 padding
- DMA pipeline 未充分隐藏延迟

### 7.4 Scalar ALU 瓶颈

表现：profile 中 Scalar ALU 利用率高，但 MXU 空闲。

原因：过多的索引计算和控制流。

解决：SMEM 查找表、unroll、位打包（见 [硬件约束文档](./hardware-constraints#4-性能陷阱)）。

---

## 8. xprof 关键指标速查

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| Scalar ALU 利用率 | 过高 = 标量计算瓶颈 | SMEM 查找表、unroll |
| MXU 利用率 | 过低 = MXU 未充分利用 | 增大 tile、减少 DMA 等待 |
| Vector Spills | 有 spill = VREG 溢出 | 减小 tile、减少活跃变量 |
| Vector Store 利用率 | 接近 100% = 写回瓶颈 | 异步写回 |
| HBM 读/写带宽 | 接近峰值 = memory-bound | 增大 batch 或 fuse ops |
| ICI 带宽利用率 | 过高 = 通信瓶颈 | 调整 sharding 策略 |

---

## 9. 参考链接

- [How to Scale Your Model — Ch9: Profiling](https://jax-ml.github.io/scaling-book/profiling)
- [How to Scale Your Model — Ch10: All About JAX](https://jax-ml.github.io/scaling-book/jax-stuff)
- [JAX TPU 显存分析指南](/onboarding/profiling-hbm-guide)
