# RFC: KDA (Kimi Delta Attention) MaxText 实现设计文档

**Author:** @qiaotonggg  
**Status:** Draft  
**Related:** PR #64 (RFC-0007), Megatron KDA, FLA Library  

---

## 1. 背景与动机

### 1.1 什么是 KDA

KDA (Kimi Delta Attention) 是 Kimi 团队提出的一种改进的线性注意力机制，结合了：

- **短卷积 (Short Convolution)**：通过 1D 因果卷积建模局部依赖
- **可学习 Decay (A_log)**：每头独立的衰减参数
- **门控机制 (Gate & Beta)**：全秩投影 g_proj / b_proj
- **线性注意力**：O(T) 复杂度，支持长序列

### 1.2 为什么需要迁移

| 方面 | Megatron 实现 | MaxText 目标 |
|-----|---------------|-------------|
| 框架 | PyTorch | JAX/Flax NNX |
| 硬件 | GPU/NPU | TPU (Pallas) |
| 并行 | Megatron 并行 | JAX sharding |
| 集成 | BailingMoE V3 | MaxText Ling3 |

### 1.3 参考实现

- **Megatron KDA**: `bailing_moe_v3/modeling_bailing_moe_v3.py:757-950`
- **MaxText GLA**: `layers/attention_gla.py` (架构参考)
- **FLA Kernel**: `fla.ops.kda.chunk_kda` (同学已实现)

---

## 2. 设计目标

### 2.1 功能目标

- [ ] 完整的 KDA 前向/反向传播
- [ ] Training 模式 (chunk mode)
- [ ] 支持 TPU Pallas kernels（直接调用 `tops.ops.kda.chunk_kda`）

> **注意**: MaxText 仅关注训练模式，Inference 模式不在本 RFC 范围内。直接接入 Pallas kernel 即可，无需 fallback 路径（参考 GLA 实现）。
>
> **范围外**：Decoder 层装配由 RFC-0012 负责；KDA 配置字段已在 RFC-0007 / PR #72 中落地，本 RFC 仅消费；Megatron → MaxText checkpoint 转换由独立 RFC 跟进。

---

## 3. 架构设计

### 3.1 整体架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                         KDA Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  Input: [B, T, hidden_size]                                      │
│         │                                                        │
│         ├── pad to chunk_size=64 multiples (if needed)           │
│         │                                                        │
│  ├─> q_proj ─> q_conv1d ─> SiLU ─> (L2 norm) ──┐               │
│  ├─> k_proj ─> k_conv1d ─> SiLU ─> (L2 norm) ──┼─ q,k,v       │
│  ├─> v_proj ─> v_conv1d ─> SiLU ────────────────┘               │
│  │                                               │               │
│  ├─> g_proj ─────────────────────> g ────────────┤               │
│  ├─> b_proj ──> sigmoid ─────────> beta ─────────┤               │
│  │                                               │               │
│  │      A_log, dt_bias (learnable params) ───────┤               │
│  │                                               ▼               │
│  │                             ┌──────────────────────┐          │
│  │                             │   pallas_chunk_kda   │          │
│  │                             │     (直接调用)        │          │
│  │                             └──────────────────────┘          │
│  │                                         │                     │
│  └─> gate_proj ──> output_gate ──┐         │                     │
│                                  ▼         ▼                     │
│               per-head RMSNorm(o) * sigmoid(output_gate)         │
│                                  │                               │
│                               o_proj                             │
│                                  │                               │
│                    unpad (if padded) ──> [B, T, hidden]          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

| 组件 | 文件路径 | 功能描述 |
|-----|---------|---------|
| `KdaAttention` | `src/maxtext/layers/attention_kda.py` | 主 KDA 模块 |
| `ShortConvolution` | `src/maxtext/layers/attention_kda.py` | 内嵌于同文件，KDA 专用 |
| `chunk_kda` | `tops.ops.kda` | 第三方 Pallas kernel，直接 import |
| `RMSNorm` | `src/maxtext/layers/normalizations.py` | 输出归一化（复用） |

### 3.3 QK L2 Normalization

> **重要**: 参考 Megatron-LM `megatron/core/ssm/kda.py:824-828`，KDA 在计算 attention 时对 Q 和 K 应用 L2 normalization。

**设计决策**: L2 norm **始终在 kernel 外部完成**，`use_qk_l2norm_in_kernel` 固定传 `False`。这与 Megatron 实现一致（`kda.py:878`），由已有配置字段 `use_qk_norm` 控制（无需 KDA 专用字段）。

**Megatron 参考**:

```python
# megatron/core/ssm/kda.py:824-828
if self.use_qk_l2norm:
    query = l2norm(query)   # from fla.modules.l2norm
    key = l2norm(key)

# megatron/core/ssm/kda.py:878
use_qk_l2norm_in_kernel=False,  # 始终 False
```

**MaxText 实现**:

```python
# layers/attention_kda.py — SiLU 之后、chunk_kda 调用之前
from tops.cpu.ops.common.l2norm import l2norm_fwd

if cfg.use_qk_norm:
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

# chunk_kda 调用时固定传 False
o, _ = chunk_kda(..., use_qk_l2norm_in_kernel=False, ...)
```

---

## 4. 详细实现方案

### 4.1 Short Convolution (短卷积)

**目的**: 在 Q/K/V 投影后添加局部依赖建模

**Megatron 参考**: 使用 `nn.Conv1d` with `groups=in_channels` 实现 depthwise 因果卷积，SiLU 通过 `causal_conv1d_fn` 或单独 `act_fn` 在外部应用。

**MaxText 实现**: 内联于 `layers/attention_kda.py`（非独立文件）。使用手动 depthwise loop 替代 `nnx.Conv`，SiLU 在主模块的 `__call__` 中统一应用（与 conv 解耦）。仅支持训练模式，无 cache。

```python
# layers/attention_kda.py
class ShortConvolution(nnx.Module):
    """Depthwise causal 1D convolution for local dependency modeling in KDA."""

    def __init__(
        self,
        kernel_size: int,
        features: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.kernel_size = kernel_size
        self.features = features
        self.dtype = dtype
        self.kernel = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (kernel_size, features), weight_dtype)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply depthwise causal 1D convolution. Input/output: [B, T, F]."""
        B, T, F = x.shape
        x_padded = jnp.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        output = jnp.zeros((B, T, F), dtype=x.dtype)
        for k in range(self.kernel_size):
            offset = self.kernel_size - 1 - k
            output = output + x_padded[:, offset : offset + T, :] * self.kernel[k]
        return output.astype(self.dtype)
```

> **注意**: SiLU 不在 ShortConvolution 内部应用，而是在主模块 `__call__` 中对 Q/K/V 统一执行 `jax.nn.silu()`，与 Megatron 行为等价。

### 4.2 Naive KDA 实现

**文件**: `kernels/kda/naive.py`

纯 JAX 参考实现，用于单元测试中与 Pallas kernel 对比验证精度。接口与 `chunk_kda` 一致，逐 token 执行 Delta Rule 递推。

```python
def naive_chunk_kda(
    q, k, v,          # [B, T, H, K/V]
    g,                 # [B, T, H, K] gate
    beta,              # [B, T, H]
    scale=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    A_log=None,        # [H]
    dt_bias=None,      # [H, K]
    use_gate_in_kernel=False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    ...
```

### 4.3 Kernel 入口

**文件**: `kernels/kda/__init__.py` → `kernels/kda/pallas.py` → `tops.ops.kda.chunk_kda`

参照 GLA 的 `chunk_gla` 模式，两层薄 wrapper 直接委托给 tops Pallas kernel。`tops.ops.kda` 是必需依赖，不提供 naive fallback。

```python
# kernels/kda/__init__.py
def chunk_kda(
    q, k, v,           # [B, T, H, K/V]
    g,                  # [B, T, H, K] gate (raw, transform done in kernel)
    beta,               # [B, T, H] sigmoid(b_proj(x))
    A_log=None,         # [H] learnable decay (log space)
    dt_bias=None,       # [H*K] gate bias
    scale=None,         # default 1/sqrt(K)
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,  # 固定 False
    use_gate_in_kernel=True,        # A_log/dt_bias 在 kernel 内应用
    safe_gate=False,                # Ling3: True
    lower_bound=None,               # Ling3: -5.0
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Returns (o: [B,T,H,V], final_state: [B,H,K,V] | None)."""
    ...

# kernels/kda/pallas.py — 直接委托给 tops.ops.kda.chunk_kda
def pallas_chunk_kda(...) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    ...
```

### 4.4 主 KDA 模块

```python
# layers/attention_kda.py

class KimiDeltaAttention(nnx.Module):
    """Kimi Delta Attention for Ling3."""
    
    def __init__(self, config, layer_idx, mesh, *, rngs):
        cfg = config
        
        # Q/K/V projections (separate, not fused like GLA)
        # 均为 DenseGeneral: hidden -> (num_heads, head_dim), kernel_axes=("embed","heads","kv")
        self.q_proj = linears.DenseGeneral(...)
        self.k_proj = linears.DenseGeneral(...)
        self.v_proj = linears.DenseGeneral(...)
        
        # Short Convolutions (Q, K, V each have their own conv)
        if cfg.linear_conv_kernel_dim > 0:
            self.q_conv = ShortConvolution(kernel_size=cfg.linear_conv_kernel_dim, features=H*K, ...)
            self.k_conv = ShortConvolution(...)
            self.v_conv = ShortConvolution(...)
        else:
            self.q_conv = self.k_conv = self.v_conv = None
        
        # A_log: learnable decay [num_heads], init log(Uniform(1, 16))
        self.A_log = nnx.Param(jnp.log(jax.random.uniform(..., minval=1.0, maxval=16.0)))
        
        # dt_bias: [num_heads * head_dim], init inverse-softplus of Uniform(dt_min, dt_max)
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = jnp.exp(jax.random.uniform(...) * (log(dt_max) - log(dt_min)) + log(dt_min))
        dt = jnp.clip(dt, min=dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = nnx.Param(inv_dt)
        
        # Beta projection: hidden -> (num_heads,)
        self.b_proj = linears.DenseGeneral(...)
        
        # Gate projection: hidden -> (num_heads, head_dim), use_bias=True
        self.g_proj = linears.DenseGeneral(...)
        
        # Output gate projection: hidden -> (num_heads, head_dim)
        self.gate_proj = linears.DenseGeneral(...)
        
        # Output norm (per-head RMSNorm)
        self.out_norm = RMSNorm(num_features=head_dim, ...)
        
        # Output projection: (num_heads, head_dim) -> hidden
        self.o_proj = linears.DenseGeneral(...)
    
    def __call__(self, hidden_states, decoder_positions=None,
                 deterministic=True, model_mode=MODEL_MODE_TRAIN, *,
                 layer_idx=None, decoder_segment_ids=None):
        cfg = self.config
        B, T, _ = hidden_states.shape
        
        # Pad to chunk_size=64 multiples (if needed)
        # TODO: 当前 padding 放在框架层；待算子层完全实现后，应下沉到算子层处理
        chunk_size = 64
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
            T = hidden_states.shape[1]
            needs_unpad = True
        else:
            needs_unpad = False
        
        # Q/K/V projections
        q = self.q_proj(hidden_states)  # [B, T, H, K]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Apply short convolution if enabled
        if self.q_conv is not None:
            q_flat = self.q_conv(q.reshape(B, T, -1))  # [B, T, H*K]
            k_flat = self.k_conv(k.reshape(B, T, -1))
            v_flat = self.v_conv(v.reshape(B, T, -1))
            q = q_flat.reshape(B, T, self.num_heads, self.head_dim)
            k = k_flat.reshape(B, T, self.num_heads, self.head_dim)
            v = v_flat.reshape(B, T, self.num_heads, self.head_dim)
        
        # SiLU activation after conv (matching Megatron)
        q = jax.nn.silu(q)
        k = jax.nn.silu(k)
        v = jax.nn.silu(v)
        
        # Q/K L2 normalization (applied outside kernel, matching Megatron)
        if cfg.use_qk_norm:
            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)
        
        # Gate and beta
        g = self.g_proj(hidden_states)           # [B, T, H, K]
        output_gate = self.gate_proj(hidden_states)  # [B, T, H, V]
        beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))  # [B, T, H]
        
        # KDA kernel
        scale = self.head_dim ** -0.5
        safe_gate = cfg.use_kda_safe_gate
        lower_bound = cfg.kda_lower_bound if safe_gate else None

        o, _ = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=self.A_log.value,
            dt_bias=self.dt_bias.value,
            scale=scale, chunk_size=64,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            safe_gate=safe_gate, lower_bound=lower_bound,
        )
        
        # Output gated norm (matching Megatron _apply_gated_norm)
        o_normed = self.out_norm(o.reshape(-1, self.head_dim))
        gate_flat = output_gate.reshape(-1, self.head_dim)
        o = (o_normed * jax.nn.sigmoid(gate_flat.astype(jnp.float32))).reshape(o.shape)
        
        # Output projection + unpad
        output = self.o_proj(o)
        if needs_unpad:
            output = output[:, :T - pad_len, :]
        
        return output, None
```

---

## 5. 配置参数

### 5.1 KDA 相关配置参数

以下配置参数由 RFC-0007 定义，本节仅作说明：

**RFC-0007 新增字段（KdaAttention 类）：**

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `linear_conv_kernel_dim` | int | 4 | 短卷积核大小 |
| `use_kda_lora` | bool | False | 是否使用 LoRA 分解 (Ling3 设 False) |
| `use_kda_safe_gate` | bool | False | 安全门控 (Ling3 设 True) |
| `kda_lower_bound` | float | 0.0 | gate 值下界 (Ling3: -5.0) |
| `enable_gated_attention` | bool | False | MLA 输出门控开关 (Ling3 设 True) |

**复用已有字段：**

| 参数名 | 类型 | Ling3 值 | 说明 |
|-------|------|---------|------|
| `use_qk_norm` | bool | True | 是否对 q/k 应用 L2 norm（KDA 层外部执行） |
| `base_num_query_heads` | int | 16 | KDA 头数（直接复用，无需 KDA 专用字段） |

> **注意**: 根据 RFC-0007，这些参数不直接修改 `configs/base.yml`，而是通过模型配置文件或代码中定义。chunk_size 在实现中硬编码为 64，不作为配置字段暴露。

---

## 6. Decoder 集成说明

> **注意**: Decoder 层集成由 RFC-0012 负责。本节仅说明 KDA 模块的使用方式，不定义具体的集成路径。

KDA 模块通过 `KimiDeltaAttention` 类提供，由 RFC-0012 的 `Ling3GenericLayer` 在 `inhomogeneous_layer_cycle` 中调用：

```python
# RFC-0012 中的使用示例
from maxtext.layers.attention_kda import KimiDeltaAttention

# 在 Ling3GenericLayer 中根据 cycle_interval 决定使用 KDA 或 MLA
if use_kda_layer:
    self.attention = KimiDeltaAttention(
        config=self.config,
        layer_idx=layer_idx,
        mesh=self.mesh,
        rngs=rngs,
    )
```

**不使用 AttentionType 枚举**: RFC-0012 设计中，KDA/MLA 的选择由 `inhomogeneous_layer_cycle_interval` 决定，而非 `attention_type` 配置。

---

## 7. Checkpoint 转换

### 7.1 Megatron -> MaxText 权重映射

Megatron 使用 fused `in_proj` 将 Q/K/V/g/gate 打包在一个投影中，转换时需要
按 `[qk_dim, qk_dim, v_dim, qk_dim, v_dim]` split 后分别映射。

> **LoRA 说明**: Megatron 的 `no_kda_lora` 控制 g/gate 是否经过 LoRA 瓶颈
> (`in_proj` 输出小维度 → `f_b_proj`/`g_b_proj` 上投影)。MaxText 当前仅实现
> `no_kda_lora=True` 路径（全秩投影），因此无 `f_b_proj`/`g_b_proj` 权重。

| Megatron 权重 | MaxText 权重 | 形状 | 说明 |
|--------------|-------------|------|------|
| `in_proj.weight` (q split) | `q_proj.kernel` | [hidden, H, K] | 转置 + reshape |
| `in_proj.weight` (k split) | `k_proj.kernel` | [hidden, H, K] | 转置 + reshape |
| `in_proj.weight` (v split) | `v_proj.kernel` | [hidden, H, V] | 转置 + reshape |
| `in_proj.weight` (g split) | `g_proj.kernel` | [hidden, H, K] | 转置 + reshape |
| `in_proj.weight` (gate split) | `gate_proj.kernel` | [hidden, H, V] | 转置 + reshape |
| `conv1d.weight` (q channel) | `q_conv.kernel` | [kernel_size, H*K] | 按 channel split |
| `conv1d.weight` (k channel) | `k_conv.kernel` | [kernel_size, H*K] | 按 channel split |
| `conv1d.weight` (v channel) | `v_conv.kernel` | [kernel_size, H*V] | 按 channel split |
| `A_log` | `A_log.value` | [H] | 直接复制 |
| `dt_bias` | `dt_bias.value` | [H*K] | 直接复制 |
| `beta_proj.weight` | `b_proj.kernel` | [hidden, H] | 转置 |
| `out_proj.weight` | `o_proj.kernel` | [H*V, hidden] | 转置 |
| `pre_output_norm.weight` | `out_norm.scale` | [V] | 直接复制 |

---

## 8. 测试计划

### 8.1 单元测试

测试文件: `tests/unit/kda_attention_test.py`

**TestShortConvolution** — 短卷积层

- `test_output_shape`: 输出形状正确
- `test_causality`: 因果性（未来 token 不影响当前输出）
- `test_against_naive`: 与 naive 逐位置实现对比

**TestKimiDeltaAttention** — KDA 模块基本功能

- `test_init_head_dims`: head 维度初始化正确
- `test_init_no_conv`: `linear_conv_kernel_dim=0` 时无 conv 层
- `test_init_has_gate_and_norm`: gate_proj / out_norm 存在
- `test_forward_shape`: 输出形状 [B, T, emb_dim]
- `test_forward_no_nan_inf`: 前传无 NaN/Inf
- `test_sequence_padding`: 非 chunk_size 整数倍序列正确 pad/unpad
- `test_deterministic`: 两次前传结果一致
- `test_packed_sequences_not_supported`: 拒绝 packed sequences
- `test_autoregressive_not_supported`: 拒绝 autoregressive mode

**TestChunkKda** — chunk_kda kernel

- `test_basic`: 基本前传无 NaN，输出非全零
- `test_chunk_vs_recurrent`: chunk_kda 与 fused_recurrent_kda 对比（atol=5e-3, rtol=5e-3）

**TestNaiveKda** — kernel 精度与行为

- `test_chunk_kda_vs_naive`: chunk_kda 与 naive 实现对比（fp32, atol=5e-3, rtol=1e-3）
- `test_fused_recurrent_vs_naive`: fused_recurrent 与 naive 对比（fp32, atol=1e-2, rtol=1e-3）
- `test_chunk_kda_vs_naive_bf16`: chunk_kda bf16 精度（atol=1e-2, rtol=1e-2）
- `test_fused_recurrent_vs_naive_bf16`: fused_recurrent bf16 精度（atol=1e-2, rtol=1.6e-2）
- `test_naive_kda_basic_properties`: 输出因果性、非零
- `test_naive_kda_zero_gate_accumulates`: g=0 时状态累积
- `test_naive_kda_large_negative_gate_decays`: 大负 g 时状态衰减

**TestQkL2Norm** — Q/K L2 归一化

- `test_qk_l2norm_applied_outside_kernel`: 开启时 L2 norm 生效
- `test_qk_l2norm_skipped_when_disabled`: 关闭时跳过
- `test_l2norm_changes_output`: L2 norm 改变最终输出

**TestKdaBackward** — 反传测试

- `test_backward_no_nan`: activation gradient 无 NaN/Inf 且非全零
- `test_backward_deterministic`: 两次 VJP 结果一致
- `test_weight_grads_no_nan`: 所有参数梯度无 NaN/Inf 且非全零
- `test_backward_bf16`: bf16 下 activation gradient 无 NaN/Inf

**TestKdaWithRealConfig** — Ling3 配置集成

- `test_ling3_use_qk_norm_is_true`: 配置正确启用 QK norm
- `test_ling3_kda_config_fields`: KDA 相关配置字段存在
- `test_ling3_kda_forward`: Ling3 配置下前传正常

### 8.2 集成测试（TODO）

- KDA + MoE 组合训练
- KDA + MLA 交替层
- 长序列 (8k+)
- Megatron dump 对比测试（参考 `gla_compare_test.py` 模式）

---

## 9. 参考文档

1. **FLA Library**: `fla.ops.kda.chunk_kda` API
2. **Megatron KDA**: `bailing_moe_v3/modeling_bailing_moe_v3.py`
3. **MaxText GLA**: `layers/attention_gla.py`
4. **KDA Paper**: (如有)

---

## 10. 附录

### 10.1 KDA 公式

```text
# 1. Short Convolution
q = Silu(Conv1d(W_q @ x))
k = Silu(Conv1d(W_k @ x))
v = Silu(Conv1d(W_v @ x))

# 2. Gate and beta
g = W_g @ x                    # raw gate, A_log/dt_bias applied in kernel
beta = sigmoid(W_b @ x)

# 3. KDA recurrence (Delta Rule)
S'    = S_{t-1} * exp(g_t)           # gated decay
res_t = v_t - S'^T @ k_t             # Delta Rule residual
S_t   = S' + beta_t * k_t ⊗ res_t   # state update with mixing coefficient
o_t   = scale * q_t^T @ S_t          # output query

# 4. Output (gated norm)
y = W_o @ (sigmoid(gate) * RMSNorm(o))
```

### 10.2 文件清单

```text
src/maxtext/
├── layers/
│   ├── attention_kda.py          # 主要实现（含 ShortConvolution）
│   └── attentions.py             # 添加 KDA 分支
├── kernels/
│   └── kda/                      # KDA kernels 目录
│       ├── __init__.py           # 导出接口（chunk_kda）
│       └── pallas.py             # Pallas kernel wrapper（tops.ops.kda）
├── checkpoint/
│   └── (TODO) convert_kda_megatron.py   # 权重转换
└── tests/
    └── unit/
        └── kda_attention_test.py # 单元测试
```
