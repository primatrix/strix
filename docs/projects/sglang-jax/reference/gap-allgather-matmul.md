---
title: 实现差距：通信-计算重叠
---

# 实现差距：AllGather+MatMul 通信-计算重叠

> tpu-inference 实现了 AllGather 与 MatMul 融合的 Pallas Kernel，通过双向环通信和多阶段管线将设备间通信延迟隐藏在计算之中。sglang-jax 无此实现。

---

## 一、差距总览

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| AllGather+MatMul 融合 Kernel | ✅ Pallas 实现 | ❌ 无 |
| 双向环通信 | ✅ 左半左传、右半右传 | ❌ 无 |
| 多阶段管线 (tp_size+2) | ✅ prologue/steady/epilogue | ❌ 无 |
| `pltpu.make_async_remote_copy` | ✅ 非阻塞 inter-device DMA | ❌ 无 |
| 调优 block sizes | ✅ 预计算查找表 | ❌ 无 |
| TP 通信方式 | AllGather+MatMul 融合 | `jax.lax.psum` 归约 |

---

## 二、原理

在 Tensor Parallelism 下，列并行 MatMul 需要先 AllGather 输入（或 ReduceScatter 输出）。标准做法是通信和计算串行执行。AllGather+MatMul 融合将通信和计算重叠，有效隐藏通信延迟。

---

## 三、tpu-inference 实现

### 3.1 管线结构

管线共 `tp_size + 2` 步：

```text
Step 0 (prologue):
  → 启动与邻居设备的远程 DMA 复制
  → 启动本地 HBM → VMEM 复制
  → 执行 local_barrier 同步

Step 1..tp_size (steady state):
  → 等待上一步远程 DMA 完成（wait）
  → 发起下一步远程 DMA（start）        ← 通信
  → 从 VMEM 读取当前 shard 的 x
  → MatMul: x_shard @ y → output_shard  ← 计算
  → 写回上一步结果到 HBM output        ← 写回
  三者同时进行

Step tp_size+1 (epilogue):
  → 等待最后的 output 写回完成
```

### 3.2 双向环通信

每个设备将数据分为左半和右半，分别向左和向右邻居传递。有效带宽翻倍：

```python
# all_gather_matmul.py: 双向环 — 左半向左传
left_remote_copy_op = pltpu.make_async_remote_copy(
    src_ref=x_hbm_ref.at[0:m_per_device_per_direction],
    dst_ref=x_hbm_scratch_ref.at[
        x_hbm_receiving_slot, 0:m_per_device_per_direction],
    send_sem=send_sems.at[0, outer_step],
    recv_sem=recv_sems.at[0, outer_step],
    device_id=(left_neighbor,),
    device_id_type=pltpu.DeviceIdType.MESH,
)

# 右半向右传
right_remote_copy_op = pltpu.make_async_remote_copy(
    src_ref=x_hbm_ref.at[m_per_device_per_direction:m_per_device],
    dst_ref=x_hbm_scratch_ref.at[
        x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
    send_sem=send_sems.at[1, outer_step],
    recv_sem=recv_sems.at[1, outer_step],
    device_id=(right_neighbor,),
    device_id_type=pltpu.DeviceIdType.MESH,
)
```

### 3.3 Scratch 内存布局

```python
# Kernel Scratch Shapes
scratch_shapes = (
    pltpu.SemaphoreType.DMA,                    # x_local_copy_sem
    pltpu.SemaphoreType.DMA,                    # y_local_copy_sem
    pltpu.SemaphoreType.DMA,                    # o_local_copy_sem
    pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # left+right send sems
    pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # left+right recv sems
    pltpu.VMEM((2, m_per_device, k), x.dtype),  # x VMEM 双缓冲
    pltpu.VMEM(y_vmem_shape, y.dtype),          # y VMEM
    pltpu.VMEM((2, m_per_device, bn), x.dtype), # output VMEM 双缓冲
    pltpu.VMEM(acc_shape, jnp.float32),         # 累加器
)
```

- **x 双缓冲**：`(2, m_per_device, k)` — 一个用于当前计算，一个用于接收下一个 shard
- **output 双缓冲**：`(2, m_per_device, bn)` — 一个用于当前写入，一个用于上一步写回
- **Semaphore 数组**：`(2, tp_size-1)` — 方向(左/右) × 步数

### 3.4 Grid 设计

```python
grid = (tp_size + 2, grid_n, grid_k)
# dim 0: 管线步骤 (prologue + tp_size 个 steady + epilogue)
# dim 1: 输出 N 维度的 tile 数
# dim 2: K 维度的 tile 数（支持 K 累积）
```

### 3.5 Steady-state MatMul

```python
# 单步 K (grid_k == 1): 直接写入 output scratch
o_vmem_scratch_ref.at[o_receiving_slot][...] = jnp.dot(
    x_vmem_scratch_ref.at[x_vmem_working_slot][...],
    y_vmem_scratch_ref.at[:, n_slice][...],
    preferred_element_type=jnp.float32,
).astype(x_vmem_scratch_ref.dtype)

# 多步 K (grid_k > 1): 累积到 accumulator
acc_vmem_scratch_ref[...] += jnp.dot(
    x_vmem_scratch_ref.at[x_vmem_working_slot, :, k_slice][...],
    y_vmem_scratch_ref.at[k_slice, n_slice][...],
    preferred_element_type=jnp.float32,
)
```

### 3.6 Output 写回

将左半和右半结果写到 AllGather 后的正确位置：

```python
# 左半结果写入
o_left_local_copy_op = pltpu.make_async_copy(
    src_ref=o_vmem_scratch_ref.at[o_working_slot, :m_per_device_per_direction],
    dst_ref=o_hbm_ref.at[
        pl.ds(m_per_device_per_direction * left_o_idx,
              m_per_device_per_direction),
        n_slice,
    ],
    sem=o_local_copy_sem,
)

# 右半结果写入
o_right_local_copy_op = pltpu.make_async_copy(
    src_ref=o_vmem_scratch_ref.at[o_working_slot, m_per_device_per_direction:],
    dst_ref=o_hbm_ref.at[
        pl.ds(m_per_device_per_direction * right_o_idx,
              m_per_device_per_direction),
        n_slice,
    ],
    sem=o_local_copy_sem,
)
```

### 3.7 公共 API

```python
def all_gather_matmul(
    x: jax.Array,       # [m_per_device, k] — 本设备的 shard
    y: jax.Array,       # [k, n] — 完整权重（每设备一份）
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,      # mesh 轴名
    collective_id: int | None = 0,
    bn: int | None = None,   # output tile N
    bk: int | None = None,   # K tile
    rhs_transpose: bool = False,
) -> jax.Array:  # [m_total, n] — AllGather 后的完整输出
```

### 3.8 调优 Block Sizes

```python
tuned_bn, tuned_bk = all_gather_matmul_tuned_block_sizes.get_tuned_block_sizes(
    m, n, k, jnp.dtype(x.dtype).name, tp_size
)
if bn is None:
    bn = tuned_bn if tuned_bn is not None else n
if bk is None:
    bk = tuned_bk if tuned_bk is not None else k
```

---

## 四、sglang-jax 实现建议

### 4.1 集成点

在 `srt/layers/linear.py` 的列并行 MatMul 中替换 `jax.lax.psum`：

```python
# 当前 sglang-jax: 串行 AllGather + MatMul
output = jnp.dot(x, weight)  # 本地 matmul
output = jax.lax.psum(output, axis_name="tensor")  # AllReduce

# 替换为: AllGather+MatMul 融合
output = all_gather_matmul(x, weight, mesh, "tensor")
```

### 4.2 实现步骤

1. 在 `srt/kernels/collectives/` 下创建 `all_gather_matmul.py`
2. 实现 `_all_gather_kernel` Pallas kernel（双向环 + 多阶段管线）
3. 实现 `all_gather_matmul` 公共 API（包装 `shard_map` + `pallas_call`）
4. 创建 `tuned_block_sizes.py` 调优配置
5. 集成到 `linear.py` 的列并行路径

### 4.3 前置条件

- `pltpu.make_async_remote_copy` API 支持
- `pltpu.DeviceIdType.MESH` 设备寻址
- 需要 TP ≥ 2 的环境测试

**优先级**: 中。TP ≥ 8 时通信开销显著，值得实现。

---

## 五、参考文件

| 文件 | 内容 |
|---|---|
| `tpu_inference/kernels/collectives/all_gather_matmul.py` | 完整 Kernel 实现 |
| `tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py` | 调优参数 |
| `tpu_inference/kernels/collectives/util.py` | `local_barrier` 等工具 |
