# Pallas Kernel 编写经验总结

基于 `chunk_fwd_h_kernel` 实践总结，涵盖 TPU Pallas kernel 的 API 约束、性能优化策略和 profiling 分析方法。

---

## 1. TPU 硬件基础

| 单元 | 功能 |
|------|------|
| **MXU (Matrix Multiply Unit)** | 矩阵乘法专用单元，执行大规模矩阵乘（如 `jax.lax.dot`），是算力核心 |
| **Vector Unit** | 向量计算主力，负责逐元素运算（加减乘除、比较、mask 等） |
| **EUP** | 集成在 Vector Unit 内部，专门负责复杂算术运算（非线性算子如 `exp`） |
| **XLU (Transpose Unit)** | 跨 lane shuffle、transpose、permutation |
| **Scalar ALU** | 标量计算，**顺序发射**，容易成为瓶颈 |
| **DMA Engine** | HBM <-> VMEM/SMEM 数据搬运 |

关键约束：**Scalar ALU 是顺序发射的**

---

## 2. Pallas API 关键约束

### 2.1 `make_async_copy`

- **必须**提供一个信号量（semaphore）
- 参数**必须**是 ref，在 kernel 函数中需要用 `.at` 获取引用
- async_copy的handler **不能**作为 `fori_loop` 的参数传递，也不能作为返回值
- DMA 的 wait 是按照**槽位**来算的，可能存在多个发起但只有一个 wait 的情况

```python
# 正确用法
copy = pltpu.make_async_copy(
    k_ref.at[(b_i, h_i, t_slice, k_slice)],  # 必须用 .at 取引用
    k_scratch_ref.at[buf],
    sems.at[0, buf],  # 必须有信号量
)
copy.start()
# ... 其他计算 ...
copy.wait()
```

### 2.2 条件执行

- **`pl.when` 比 `lax.cond` 快**，运行 prefetch 等操作应优先使用 `pl.when`
- `lax.cond` 会引入额外的 Scalar ALU 开销

```python
# 推荐
@pl.when(i + 1 < NT)
def do_prefetch():
    ...

# 不推荐
lax.cond(i + 1 < NT, prefetch_fn, lambda _: None, None)
```

### 2.3 Shape 与索引

- 逻辑中**不要有动态 shape**，比如 `where` 之后取索引
- `fori_loop` 中最好**只用循环变量 `i`** 来计算所有索引，不要引入其他变量
- `pl.loop` 配合 `unroll=True` 可以让信号量索引成为**编译时常量**（TPU DMA 要求）

```python
# 推荐：unroll=True 使 i 成为 Python int，所有算术在 trace time 求值
@pl.loop(0, NT, unroll=True)
def body(i):
    buf = i % 2           # 编译时常量
    next_buf = (i + 1) % 2  # 编译时常量
    ...
```

### 2.4 BlockSpec

- `memory_space` 默认不指定时数据放在 **VMEM** 上
- 指定为 `pltpu.HBM` 时，ref 是**全部 shape** 的数据（需要手动管理搬运）
- `pipeline_mode` 参数：不可以指定 lookahead；`buffer_count` 只能为 1 或 2，指定为 2 时性能基本无提升

---

## 3. Grid 与并行策略

### 3.1 Batch 维度放 Grid vs 内部展开

**结论：Batch 维度放在 Grid（外部划分）比在 kernel 内部用 `pl.loop` 展开要快。**

- Grid 中的 `"parallel"` 维度允许编译器在核间并行调度
- 内部展开增加 Scalar ALU 负担

```python
# 推荐：Batch 放 grid
grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV))
dimension_semantics = ("parallel", "parallel", "arbitrary", "arbitrary")

# 不推荐：Batch 在 kernel 内部循环
grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV))
@pl.loop(0, B, unroll=True)
def b_body(b_i):
    ...
```

### 3.2 dimension_semantics

- `"parallel"`：维度间无数据依赖，可以并行调度
- `"arbitrary"`：维度间可能有依赖（如序列维度的累加），顺序执行

---

## 4. 数据搬运优化

### 4.1 BlockSpec 自动管理 vs 手动 DMA

**核心结论：规则访问模式下，BlockSpec 自动管理几乎总是优于手动 DMA。**

| 方案 | chunk_size=512 耗时 | 说明 |
|------|-------------------|------|
| BlockSpec 自动管理 | ~19us | 编译器自动 double buffer |
| 完全手动 DMA（单 grid） | ~26us | 循环索引管理开销 |

**什么时候才值得手动 DMA**：

- 运行时**数据依赖**的访问模式（如 `cu_seqlens` 变长序列）
- **跨核通信**
- **不规则/非矩形**访问模式

### 4.2 使用 VMEM Scratch 的注意事项

将循环迭代维度放入 grid（利用 scratch 做跨迭代累加）相比 `fori_loop`，每次循环会多出额外的 load/store 操作，可能更慢。

```python
# scratch 做跨迭代累加的模式
scratch_ref[...] = scratch_ref[...] + jax.lax.dot(k_tile.T, v_tile)
# 每次迭代多 2 次访存（读+写 scratch），如果计算量不大则不划算
```

### 4.3 Double Buffer + Prefetch 模式

标准模式：prefetch 放在 wait 之前，让 DMA 在 wait + compute 期间运行。

```python
# 1. 先启动下一次迭代的 prefetch
@pl.when(i + 1 < NT)
def do_prefetch():
    _async_copy(k_ref.at[...], k_scratch_ref.at[next_buf], sems.at[0, next_buf])
    _async_copy(v_ref.at[...], v_scratch_ref.at[next_buf], sems.at[1, next_buf])

# 2. 再 wait 当前迭代的数据
_async_copy(k_ref.at[...], k_scratch_ref.at[buf], sems.at[0, buf], wait=True)
_async_copy(v_ref.at[...], v_scratch_ref.at[buf], sems.at[1, buf], wait=True)

# 3. 计算
k_tile = k_scratch_ref[buf]
...
```

**但注意**：BlockSpec 本身就自带 double buffer 机制，手动实现往往比不过编译器自动优化。

---

## 5. 减少 Scalar ALU 开销

Scalar ALU 是最常见的瓶颈。以下技术可以减少开销：

### 5.1 SMEM 索引查找表

用预计算的查找表替代 kernel 内部的整数除法/取模运算：

```python
# Host 端预计算索引表，放入 SMEM
flat_idx = jnp.arange(total + 1, dtype=jnp.int32)
b_tab = flat_idx // h_sum_v
h_tab = (flat_idx % h_sum_v) // NK_sum_v
...
index_table = jnp.stack([b_tab, h_tab, k_tab, v_tab, t_tab], axis=1)

# BlockSpec 指定 SMEM
in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))

# Kernel 内部直接读表
b_i = idx_ref[i, 0]
h_i = idx_ref[i, 1]
```

### 5.2 位打包（Bit Packing）

进一步压缩 SMEM 访问次数：

```python
# 将两个 16-bit 索引打包到一个 int32
bh_packed = (b_tab << 16) | h_tab
kv_packed = (k_tab << 16) | v_tab

# Kernel 内解包
bh = idx_ref[i, 0]
b_i = bh >> 16
h_i = bh & 0xFFFF
```

### 5.3 避免动态 buffer index 计算

`buf = jnp.mod(i_t, 2)` 这类动态计算会在 Scalar ALU 上执行。如果可以 unroll，让 `i` 成为编译时常量。

---

## 6. Profiling 分析方法

### 6.1 采集 Profile

```python
output = kernel(...)
jax.block_until_ready(output)  # 预热
jax.profiler.start_trace("/path/to/profile")
for i in range(3):
    output = kernel(...)
    jax.block_until_ready(output)
jax.profiler.stop_trace()
```

### 6.2 关注指标

| 指标 | 含义 |
|------|------|
| **Scalar ALU 利用率** | 过高说明标量计算是瓶颈 |
| **Vector Spills** | 无 spill 说明向量寄存器未打满 |
| **XLU 活动** | 跨 lane 通信，常见于广播和 transpose |
| **Vector Store 利用率** | 未满说明写回不是瓶颈 |
| **DMA wait vs hbm_to_vmem 数量** | 不一致可能是多发起单等待的优化 |

### 6.3 CostEstimate

可以为 `pallas_call` 提供 cost estimate 帮助编译器优化：

```python
cost_estimate = pl.CostEstimate(
    flops=body_cost.flops,
    transcendentals=body_cost.transcendentals,
    bytes_accessed=input_bytes + output_bytes,
)
```

---

## 7. 其他注意事项

- **OOM**：当 `batch_size * T` 很大时，BlockSpec 指定 `memory_space=pltpu.HBM` 让 ref 持有全部数据，可能导致 OOM
- **vmem_limit_bytes**：可以通过 `CompilerParams(vmem_limit_bytes=128 * 1024 * 1024)` 控制 VMEM 使用上限
- **disable_bounds_checks**：手动管理 DMA 时可以设置 `disable_bounds_checks=True` 减少边界检查开销
- **输出也可以 async**：中间状态写回 HBM 也可以用 DMA async，避免阻塞计算流水线
- **Semaphore 分配**：对于多种数据类型的 DMA（k, v, gk, h0, h_out, ht_out），建议使用二维信号量 `pltpu.SemaphoreType.DMA((6, 2))` 统一管理
