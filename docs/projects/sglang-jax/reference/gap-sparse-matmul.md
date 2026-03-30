---
title: 实现差距：N:M 结构化稀疏
---

# 实现差距：N:M 结构化稀疏矩阵乘

> tpu-inference 实现了软件模拟的 N:M 结构化稀疏 MatMul（`Sparsifier` 编码/解码 + Pallas Kernel）。sglang-jax 无此实现。

---

## 一、差距总览

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| `Sparsifier` 编码/解码 | ✅ | ❌ |
| 稀疏 MatMul Pallas Kernel | ✅ | ❌ |
| 支持 N:M 配置 | 任意 N:M（M ≤ 16） | ❌ |
| 支持 dtype | f32 / bf16 / int8 | ❌ |
| 稀疏维度 | LHS 或 RHS，收缩或非收缩 | ❌ |

---

## 二、N:M 稀疏原理

N:M 结构化稀疏指每 M 个连续值中保留 N 个非零值。例如 2:4 稀疏意味着每 4 个值中只有 2 个非零，节省约 50% 存储和计算。

存储格式：**非零值数组 + 压缩索引 metadata**。

---

## 三、tpu-inference 实现

### 3.1 Sparsifier 类

`Sparsifier` 负责稀疏矩阵的编码（dense → sparse）和解码（sparse → dense）：

```python
class Sparsifier:
    """N:M 结构化稀疏编码器/解码器

    Attributes:
        sparsity: (N, M) 稀疏度配置
        nonzeros: 提取的非零值
        metadata: 压缩的非零值索引
    """
    def __init__(self, data, mask, *, sparsity, sparse_dim, stride):
        self.nonzeros, self.metadata = self.encode(data, mask)
```

#### encode — 提取非零值 + 压缩索引

```python
def encode(self, data, mask):
    x, y = self.sparsity  # e.g., (2, 4)
    h, w = data.shape

    # 1. Reshape 为 (groups, M, stride) 布局
    data = data.reshape(-1, y, self.stride).transpose(0, 2, 1)
    mask = mask.reshape(-1, y, self.stride).transpose(0, 2, 1)

    # 2. Top-K 选择非零值索引
    _, nz_idx = jax.lax.top_k(mask, x)

    # 3. 按索引 gather 非零值
    nz = data[
        np.arange(nz_idx.shape[0])[:, None, None],
        np.arange(nz_idx.shape[1])[:, None],
        nz_idx,
    ]

    # 4. 压缩索引 metadata（位打包）
    metadata = self.compress(nz_idx, next_log2(y))
    return nz, metadata
```

#### compress — 位打包压缩索引

将多个索引值打包到单个 int32 中：

```python
def compress(self, data, bitwidth):
    """Pack indices based on bitwidth. e.g., 2:4 → 2-bit indices, 16 per int32"""
    packing = 32 // bitwidth
    comp = np.zeros((d0, d1 // packing, d2), dtype=jnp.int32)
    for i in range(d1):
        shift = i % packing * bitwidth
        comp_i = i // packing
        comp[:, comp_i, :] |= data[:, i, :] << shift
    return comp
```

#### decode — 从稀疏格式恢复

```python
def decode(self):
    """从 nonzeros + metadata 恢复为 dense matrix + mask"""
    nz_idx = self.decompress(self.metadata, next_log2(y))
    data = np.zeros((h, w // y, y), dtype=self.dtype)
    data[
        np.arange(nz_idx.shape[0])[:, None, None],
        np.arange(nz_idx.shape[1])[:, None],
        nz_idx,
    ] = nz
    return data.reshape(h, w), mask.reshape(h, w)
```

### 3.2 Pallas 稀疏 MatMul Kernel

内核在每个 tile 内动态解压缩 metadata → scatter 非零值 → 标准 dense MatMul：

```python
def _kernel(nz_tile_ref, md_tile_ref, mat_tile_ref, out_tile_ref, acc_tile_ref):
    # 1. 初始化累加器
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_tile_ref[...] = jnp.zeros_like(acc_tile_ref, acc_dtype)

    # 2. 解压 metadata → 非零值索引
    nonzeros_idx = _decompress_metadata(md_tile_ref, md_packing)

    # 3. 从索引重建 dense tile
    nonzeros = nz_tile_ref[...].astype(acc_dtype)
    decompressed_tile = _decompress_nonzeros(
        sparsity, nonzeros, nonzeros_idx, sparse_dim, stride, default_value,
    )

    # 4. 标准 dense dot_general
    mat_tile = mat_tile_ref[...]
    if rhs_sparse:
        lhs_tile, rhs_tile = mat_tile, decompressed_tile
    else:
        lhs_tile, rhs_tile = decompressed_tile, mat_tile

    acc_tile_ref[...] += jax.lax.dot_general(
        lhs_tile, rhs_tile, dim_nums, preferred_element_type=acc_dtype
    )

    # 5. 最后一个 k-tile 写出结果
    @pl.when(pl.program_id(2) == k // block_k - 1)
    def _():
        out_tile_ref[...] = acc_tile_ref[...].astype(out_dtype)
```

#### 内核内 metadata 解压

```python
def _decompress_metadata(md_tile_ref, packing):
    """在 Pallas 内核中解压位打包的索引"""
    bitwidth = 32 // packing
    decompressed_md = []
    for i in range(h):
        unpacked_md = jnp.broadcast_to(md_tile_ref[:, pl.ds(i, 1), :], (x, packing, w))
        shift = jax.lax.broadcasted_iota(jnp.int32, unpacked_md.shape, 1) * bitwidth
        unpacked_md = jax.lax.bitwise_and(
            jax.lax.shift_right_logical(unpacked_md, shift),
            jnp.broadcast_to(2**bitwidth - 1, unpacked_md.shape),
        )
        decompressed_md.append(unpacked_md)
    return jnp.concatenate(decompressed_md, axis=1)
```

#### 非零值重建为 dense tile

```python
def _decompress_nonzeros(sparsity, nonzeros, nonzeros_idx, sparse_dim, stride, default_value):
    """从稀疏非零值 + 索引重建 dense tile"""
    x, y = sparsity
    target_size = nonzeros.shape[1 + sparse_dim] * y
    tiles = [None] * (target_size // stride)
    for i in range(y):
        lhs_tile_part = default_value
        for xi in range(x):
            lhs_tile_part = jnp.where(
                nonzeros_idx[xi] == i, nonzeros[xi], lhs_tile_part
            )
        for j in range(target_size // (y * stride)):
            tiles[j * y + i] = lhs_tile_part[:, j * stride:(j + 1) * stride]
    return jnp.concatenate(tiles, axis=sparse_dim).astype(nonzeros.dtype)
```

### 3.3 Pallas Call 配置

```python
pl.pallas_call(
    _kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(m // block_m, n // block_n, k // block_k),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((block_m, block_n), acc_dtype)],
    ),
    out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
)(nonzeros, metadata, matrix)
```

---

## 四、支持的配置

| 参数 | 支持范围 |
|---|---|
| N:M | 任意，M ≤ 16（如 2:4, 4:8, 1:4） |
| dtype | f32, bf16, int8 |
| 稀疏侧 | LHS 或 RHS (`rhs_sparse` 参数) |
| 稀疏维度 | 收缩维或非收缩维 (`sparse_dim`) |
| RHS 转置 | 支持 (`rhs_transpose`) |

---

## 五、sglang-jax 实现建议

**优先级**: 低。当前为独立 Kernel，需模型提供稀疏权重才能使用。

### 5.1 实现步骤

1. 在 `srt/kernels/` 下创建 `structured_sparse_matmul/`
2. 实现 `Sparsifier` 类（encode/decode/compress/decompress）
3. 实现 `_structured_spmm` Pallas Kernel
4. 可选：集成到 MoE 专家计算路径，对稀疏专家权重使用

### 5.2 集成方式

```python
# 1. 离线阶段：对模型权重进行 N:M 剪枝
mask = prune_to_nm(weight, n=2, m=4)
sparsifier = Sparsifier(weight, mask, sparsity=(2, 4), sparse_dim=1, stride=1)
# 存储 sparsifier.nonzeros + sparsifier.metadata

# 2. 推理阶段：使用稀疏 MatMul
output = structured_spmm(sparsifier.nonzeros, sparsifier.metadata, input,
                          sparsity=(2, 4), sparse_dim=1, rhs_sparse=True)
```

---

## 六、参考文件

| 文件 | 内容 |
|---|---|
| `tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py` | 完整实现 |
| `tpu_inference/kernels/structured_sparse_matmul/v1/spmm_test.py` | 测试用例 |
