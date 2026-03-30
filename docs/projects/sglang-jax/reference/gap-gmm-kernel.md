---
title: 实现差距：GMM Kernel
---

# 实现差距：GMM Kernel（V1 vs V2）

> tpu-inference 和 sglang-jax 都实现了 GMM (Grouped Matrix Multiply) Kernel 的 V1 和 V2 版本，但 tpu-inference 的 V2 拥有更多高级特性。本文详细对比两个版本的差异。

---

## 一、差距总览

| 特性 | tpu-inference V2 | sglang-jax V2 |
|---|---|---|
| `emit_pipeline` 三重缓冲 | ✅ RHS 权重 `Buffered(buffer_count=3)` | ✅ 已有 |
| VMEM 感知 Tiling | ✅ 动态计算 `VMEM / 3` | ✅ 已有 |
| 动态 LHS 量化 | ✅ 内核内 per-block abs-max 量化 | ✅ 已有 |
| Sub-byte 类型 (int4/MXFP4) | ✅ `pltpu.bitcast` 解包 | ❌ 无 |
| 激活函数融合 (Gate+Up) | ✅ `fuse_act` 参数 | ❌ 无 |
| 异步 DMA 置零 | ✅ `zero_out_start/end` | ❌ 无 |
| 内核内 Metadata 填充 | ✅ `fill_metadata` via `lax.fori_loop` | ✅ 已有 |
| `BoundedSlice` 动态行 | ✅ 变长 group tiling | ✅ 已有 |

---

## 二、架构对比

### 2.1 V1 Kernel — 经典 Pallas

V1 使用 `PrefetchScalarGridSpec` + 显式 `BlockSpec` 索引映射，metadata 在内核外部通过 host-side JAX ops 计算。

```python
# tpu-inference V1: gmm.py
call_gmm = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[lhs_block_spec, rhs_block_spec, rhs_scale_block_spec, ...],
        out_specs=out_block_spec,
        grid=(tiles_n, num_active_tiles, tiles_k),
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
    ),
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary", "arbitrary"),
    ),
)
```

**V1 独有优势**：支持**子通道量化**（`rhs_scale.shape[1] > 1`），在每个 k-block 内独立缩放：

```python
# V1: 子通道量化 — 每个 quant block 独立 dot + scale
for b_i in range(num_quant_blocks_per_tk):
    partial_result = jnp.dot(
        loaded_lhs[..., b_i * quant_block_size:(b_i + 1) * quant_block_size],
        loaded_rhs[b_i * quant_block_size:(b_i + 1) * quant_block_size, ...],
        preferred_element_type=jnp.float32,
    )
    if rhs_scale is not None:
        partial_result *= jnp.broadcast_to(rhs_scale[b_i], partial_result.shape)
    acc = acc + partial_result
```

### 2.2 V2 Kernel — `emit_pipeline` 驱动

V2 使用 `pltpu.emit_pipeline` 自动管理 DMA-计算重叠，metadata 在内核内部通过 `lax.fori_loop` 动态填充。

```python
# tpu-inference V2: gmm_v2.py — 核心 emit_pipeline 调用
pipeline_fn = pltpu.emit_pipeline(
    functools.partial(inner_kernel, cfgs=cfgs),
    grid=(num_n, num_gm, num_k),
    in_specs=(lhs_spec, rhs_spec),
    out_specs=out_spec,
)
pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)
```

---

## 三、V2 独有特性详解

### 3.1 三重缓冲

RHS 权重使用 `pl.Buffered(buffer_count=3)` 实现三重缓冲，确保 MXU 计算、DMA 读取下一块、写回上一块可以同时进行：

```python
# gmm_v2.py line 282-285
rhs_weight_spec = pl.BlockSpec(
    (None, tile_k_rhs, cfgs.tiles.tile_n),
    index_map.rhs_weight_index_map,
    pipeline_mode=pl.Buffered(buffer_count=3),  # 三重缓冲
)
```

### 3.2 VMEM 感知 Tiling

`calculate_tiling` 根据 TPU 的 VMEM 容量动态计算 tile 大小，目标是每个 RHS buffer 占 VMEM 的 1/3（为三重缓冲留空间）：

```python
# gmm_v2.py: calculate_tiling
num_rhs_buffers = 3
rhs_vmem_target = vmem_limit_bytes // num_rhs_buffers
base_rhs_size_bytes = dims.size_k * dims.size_n * rhs_bits // 8

tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2

# 先缩小 tile_n，再缩小 tile_k
while (pl.cdiv(base_rhs_size_bytes, num_n_tiles) > rhs_vmem_target
       and tile_n > tile_n_limit):
    num_n_tiles += 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles
```

### 3.3 动态 LHS 量化

V2 在内核内部对 LHS 激活进行 per-block 动态量化（abs-max → scale → 量化 → matmul → rescale）：

```python
# gmm_v2.py: inner_kernel._matmul — 动态 LHS 量化
for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
    block_lhs = tiled_lhs[:, start_k:end_k]

    # Per-block abs-max 量化
    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
    block_scale = block_abs_max / dtype_max
    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
    block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

    # 量化 MatMul
    block_acc = jnp.matmul(
        block_lhs_q, block_rhs,
        preferred_element_type=preferred_element_type,
    ).astype(acc_ref.dtype)

    # Rescale
    block_acc *= block_scale.astype(acc_ref.dtype)
    if cfgs.rhs_cfgs.has_scale:
        block_acc *= rhs_scale_slice[b_id, :, start_n:end_n].astype(acc_ref.dtype)
    acc_n += block_acc
```

### 3.4 Sub-byte 类型支持

V2 通过 `pltpu.bitcast` 支持 int4、float4_e2m1fn 等低于 8-bit 的类型：

```python
# gmm_v2.py: sub-byte 支持
@property
def should_bitcast(self) -> bool:
    bits = jax.dtypes.itemsize_bits(self.dtype)
    return bits < 8  # int4, float4_e2m1fn 等

# 内核中解包
tiled_rhs = tiled_rhs_ref.get_weight()
if cfgs.rhs_cfgs.should_bitcast:
    tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
```

### 3.5 激活函数融合

Gate+Up 投影合并到单次 MatMul 中，内核内部应用激活函数：

```python
# gmm_v2.py: 激活融合
if cfgs.fuse_act is not None:
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:], rhs_ref)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
    # inner_kernel 中: gate_result = silu(gate_matmul) * up_matmul
```

### 3.6 异步 DMA 置零

V2 使用 `pltpu.make_async_copy` 异步清零未使用的输出区域，与计算重叠：

```python
# gmm_v2.py: 异步置零
zero_size = zero_out_start(out_ref, zero_ref, semaphore_ref, ...)
# ... 计算 ...
zero_out_end(out_ref, semaphore_ref, zero_size, ...)
```

---

## 四、sglang-jax 缺失特性的实现建议

### 4.1 Sub-byte 类型 (MXFP4)

**优先级**: 高（GPT-OSS 模型依赖 MXFP4）

```python
# 需要在 sglang-jax 的 gmm_v2.py 中添加:
# 1. InputConfigs 添加 should_bitcast 属性
# 2. 内核中添加 pltpu.bitcast 解包逻辑
# 3. calculate_tiling 考虑 sub-byte 的 lhs_bits 调整
```

### 4.2 激活函数融合

**优先级**: 中（减少一次 HBM 读写）

需要将 Gate 和 Up 的 RHS 权重拼接，在单次 GMM 中同时计算，然后在内核内部应用 `silu(gate) * up`。

### 4.3 异步 DMA 置零

**优先级**: 低（性能微优化）

使用 `pltpu.make_async_copy` 将零值异步写入输出的空白区域。

---

## 五、V1/V2 选择逻辑

两个代码库的分发逻辑类似：

```python
# sglang-jax: megablox_gmm_backend.py
# V2 条件: 非 interpret 模式 且 rhs_scale 为 per-channel (shape[1] == 1)
# 否则回退到 V1

# tpu-inference: 类似逻辑
# V2 不支持子通道量化 → 回退 V1
```

---

## 六、参考文件

| 仓库 | 文件 | 内容 |
|---|---|---|
| tpu-inference | `kernels/megablox/gmm_v2.py` | V2 Kernel + emit_pipeline |
| tpu-inference | `kernels/megablox/gmm.py` | V1 Kernel + 子通道量化 |
| sglang-jax | `srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py` | V2 Kernel |
| sglang-jax | `srt/kernels/gmm/megablox_gmm_kernel/gmm.py` | V1 Kernel |
| sglang-jax | `srt/kernels/gmm/megablox_gmm_backend.py` | V1/V2 分发 |
