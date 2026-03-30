---
title: 实现差距：高级优化
---

# 实现差距：高级优化

> 本文覆盖 tpu-inference 已实现但 sglang-jax 缺失的优化特性：AllGather+MatMul 融合、N:M 结构化稀疏、MXFP4/AWQ 量化、YARN RoPE、GPT-OSS 模型。

---

## 一、差距总览

| 特性 | tpu-inference | sglang-jax |
|---|---|---|
| AllGather+MatMul 融合 | ✅ 双向环通信 Pallas Kernel | ❌ 无 |
| N:M 结构化稀疏 MatMul | ✅ `Sparsifier` + Pallas Kernel | ❌ 无 |
| MXFP4 量化 | ✅ `float4_e2m1fn` + `e8m0` scale | ❌ 无 |
| AWQ 量化 | ✅ 特殊 packing 顺序 | ❌ 无 |
| YARN RoPE with mscale | ✅ `DeepseekScalingRotaryEmbedding` | ❌ 仅辅助函数 |
| GPT-OSS 模型 | ✅ `GptOss` + sliding window + sinks | ❌ 无 |
| SparseCore Gather-Reduce | ✅ TPU v7 硬件加速 | ❌ 无 |
| N-gram 投机采样 | ✅ 基于 token 历史模式匹配 | ❌ 无 |

---

## 二、AllGather+MatMul 融合

### 2.1 原理

在 TP 配置下，将设备间通信（AllGather）与矩阵乘法重叠，隐藏通信延迟。

### 2.2 tpu-inference 实现

**管线阶段** (`tp_size + 2` 步)：

```text
Step 0 (prologue):
  → 启动远程 DMA (send local shard to neighbors)
  → 启动本地 HBM→VMEM 复制

Step 1..tp_size-1 (steady state):
  → DMA 下一个 shard from neighbor     // 通信
  → MXU 计算当前 shard 的 matmul       // 计算
  → 写入上一个结果到 HBM output        // 写回

Step tp_size+1 (epilogue):
  → 写入最终结果
```

**双向环通信**：数据分半，左半向左传递、右半向右传递，有效带宽 ×2。

**TPU 特定 API**：

```python
pltpu.make_async_remote_copy()  # 非阻塞 inter-device DMA
pltpu.DeviceIdType.MESH         # mesh-aware 设备寻址
```

**关键文件**: `tpu_inference/kernels/collectives/all_gather_matmul.py`

### 2.3 sglang-jax 实现建议

**优先级**: 中。TP 度较高时（≥8）通信开销显著，值得实现。

**实现步骤**：

1. 在 `srt/kernels/collectives/` 下创建 `all_gather_matmul.py`
2. 实现 `tp_size + 2` 步管线 Pallas Kernel
3. 双缓冲 VMEM scratch
4. 双向环通信
5. 集成到 `srt/layers/linear.py` 的列并行 MatMul 中

**前置条件**：需要 `pltpu.make_async_remote_copy` API 支持。

---

## 三、N:M 结构化稀疏 MatMul

### 3.1 原理

每 M 个连续值中保留 N 个非零值（如 2:4、4:8），以压缩存储和减少计算量。

### 3.2 tpu-inference 实现

**软件模拟方案**（非硬件稀疏指令）：

```text
Sparsifier:
  encode(dense_matrix) → (nonzero_values, compressed_metadata)
  decode(nonzero_values, metadata) → dense_matrix

Kernel:
  for each tile (m, n, k):
    1. 解压 metadata → 恢复非零索引
    2. Scatter 非零值到 dense tile
    3. 标准 dense dot_general
    4. 跨 k tiles 累积
```

**支持配置**：

- 任意 N:M（M ≤ 16）
- f32 / bf16 / int8 dtype
- LHS 或 RHS 稀疏
- 收缩维或非收缩维

**关键文件**: `tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py`

### 3.3 sglang-jax 实现建议

**优先级**: 低。当前为独立 Kernel，需显式集成到 MoE pipeline，实际收益取决于模型是否提供稀疏权重。

**实现步骤**：

1. 在 `srt/kernels/` 下创建 `structured_sparse_matmul/`
2. 实现 `Sparsifier` 编码/解码
3. 实现 Pallas sparse MatMul Kernel
4. 集成到 MoE 专家计算路径（可选）

---

## 四、MXFP4 量化

### 4.1 原理

Microscaling FP4：使用 `float4_e2m1fn`（2 位指数 + 1 位尾数）+ `e8m0` 共享 scale，实现 4-bit 量化。

### 4.2 tpu-inference 实现

```python
# 两个 e2m1 值打包到一个 uint8
# e8m0 scale 每 block 共享
# GPT-OSS 模型使用

# GMM V2 Kernel 中支持 sub-byte 类型：
pltpu.bitcast(packed_uint32, jnp.float4_e2m1fn)  # 解包
```

**关键文件**:

- `tpu_inference/layers/common/quantization/__init__.py`
- `tpu_inference/kernels/megablox/gmm_v2.py`（sub-byte 支持）

### 4.3 sglang-jax 实现建议

**优先级**: 高（如果要支持 GPT-OSS 模型）。

**实现步骤**：

1. 在 `srt/configs/quantization_config.py` 的 `DTYPE_MAP` 中添加 `float4_e2m1fn`
2. 在 `srt/utils/quantization/` 中实现 MXFP4 打包/解包
3. 在 GMM Kernel 中添加 sub-byte 类型支持（`pltpu.bitcast`）
4. 创建 `mxfp4.yaml` 量化配置

---

## 五、AWQ 量化

### 5.1 原理

Activation-aware Weight Quantization (4-bit)，使用特殊打包顺序 `(0,2,4,6,1,3,5,7)` 优化 TPU 访存模式。

### 5.2 tpu-inference 实现

**关键文件**: `tpu_inference/layers/common/quantization/__init__.py`

### 5.3 sglang-jax 实现建议

**优先级**: 低。Int8/FP8 已覆盖大部分场景，AWQ 主要用于极致压缩。

---

## 六、YARN RoPE with mscale

### 6.1 差距详情

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| `_yarn_get_mscale()` | ✅ | ✅ 已有 |
| `_yarn_find_correction_range()` | ✅ | ✅ 已有 |
| `DeepseekScalingRotaryEmbedding` 类 | ✅ 完整实现 | ❌ 无 |
| `GptOssRotaryEmbedding` 类 | ✅ NTK-by-parts | ❌ 无 |
| `get_rope()` YARN 分支 | ✅ | ❌ 仅支持 `llama3` scaling |
| Interleaved RoPE pairing | ✅ even/odd 配对 | ❌ 仅 split-half |

### 6.2 tpu-inference 实现

**DeepseekScalingRotaryEmbedding**：

```python
# 频率计算
inv_freq = base ** (-2k/dim)
interpolation_freq = inv_freq / scaling_factor
extrapolation_freq = inv_freq

# 校正范围混合
low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max_pos)
ramp = _yarn_linear_ramp_mask(low, high, dim // 2)
inv_freq = interpolation_freq * (1 - ramp) + extrapolation_freq * ramp

# mscale 校正
mscale = _yarn_get_mscale(scaling_factor, mscale_value)
       / _yarn_get_mscale(scaling_factor, mscale_all_dim)
sin, cos = jnp.sin(freqs) * mscale, jnp.cos(freqs) * mscale
```

**注意力 scale 额外 mscale²**：

```python
yarn_mscale = 0.1 * mscale_all_dim * log(scaling_factor) + 1.0
scale = qk_head_dim ** (-0.5) * yarn_mscale ** 2
```

**特殊处理**：使用 interleaved (even/odd) RoPE 配对，sin/cos cache 对齐到 128 倍数，自定义内存布局 `major_to_minor=(1, 0)` 优化 TPU。

**关键文件**: `tpu_inference/layers/jax/rope.py:81`

### 6.3 sglang-jax 实现建议

**优先级**: 高（DeepSeek V3 前置依赖）。

**实现步骤**：

1. 在 `srt/layers/embeddings.py` 中创建 `DeepseekScalingRotaryEmbedding` 类
2. 复用已有的 `_yarn_get_mscale()` 和 `_yarn_find_correction_range()` 辅助函数
3. 在 `get_rope()` 工厂函数中添加 `scaling_type == "deepseek"` 分支
4. 支持 interleaved RoPE pairing（even/odd 而非 split-half）
5. 可选：创建 `GptOssRotaryEmbedding`（NTK-by-parts 变体）

---

## 七、GPT-OSS 模型

### 7.1 架构特点

- **全层 MoE**（无 Dense 层）
- **Sliding Window + Full Attention 交替**（奇偶层）
- **Attention Sinks**（per-head sink 参数，保留初始 token 注意力）
- **MXFP4 量化**
- **`swigluoai` 激活函数**
- **head_dim = 64**（使用 RPA hd64 Kernel）

### 7.2 sglang-jax 实现建议

**优先级**: 中。

**前置依赖**：

- MXFP4 量化支持
- `GptOssRotaryEmbedding` (NTK-by-parts YARN)
- RPA hd64 Kernel（sglang-jax 的 RPA 已支持 head_dim=64？需确认）
- Attention Sinks 支持

**实现步骤**：

1. 在 `srt/models/` 下创建 `gpt_oss.py`
2. 实现全 MoE 架构 + sliding window/full attention 交替
3. 添加 `swigluoai` 激活到 MoE Kernel
4. 集成 MXFP4 量化
5. 注册到 `registry.py`

---

## 八、SparseCore Gather-Reduce (TPU v7)

### 8.1 原理

TPU v7 SparseCore 硬件加速 MoE 的 gather+reduce 步骤。

### 8.2 tpu-inference 实现

```python
# 启用条件
def is_supported_by_sc_gather_reduce():
    return tpu_generation == 7 and batch_size > threshold

# 算法
# 1. tpu.enqueue_indirect_dma: 硬件间接 gather
# 2. 加权归约: reduce_group_size (top-k) 行 → 2 行
# 3. bf16 packing
# 4. 双缓冲管线
```

**关键文件**: `tpu_inference/kernels/gather/gather_reduce.py`

### 8.3 sglang-jax 实现建议

**优先级**: 低（TPU v7 专用，需硬件才能测试）。

---

## 九、N-gram 投机采样

### 9.1 差距

tpu-inference 支持基于请求 token 历史的 N-gram 模式匹配投机采样。sglang-jax 仅有 EAGLE / EAGLE3 / STANDALONE 方法。

### 9.2 sglang-jax 实现建议

**优先级**: 低。EAGLE3 已是更高效的方法，N-gram 主要作为无模型回退。

---

## 十、优先级排序

| 特性 | 优先级 | 理由 |
|---|---|---|
| YARN RoPE with mscale | **高** | DeepSeek V3 前置依赖 |
| MXFP4 量化 | **高** | GPT-OSS 前置依赖 |
| AllGather+MatMul 融合 | **中** | 高 TP 度下性能关键 |
| GPT-OSS 模型 | **中** | Q2 计划模型 |
| N:M 结构化稀疏 | **低** | 独立 Kernel，需显式集成 |
| AWQ 量化 | **低** | Int8/FP8 已覆盖多数场景 |
| SparseCore | **低** | TPU v7 专用 |
| N-gram 投机 | **低** | EAGLE3 更优 |
