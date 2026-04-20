# RFC: Refactor PR1 - Ling3 配置扩展

| 字段     | 值                                 |
| -------- | ---------------------------------- |
| **作者** | @QiaoTong                          |
| **日期** | 2026-04-15                         |
| **状态** | Draft                              |
| **PR**   | PR1（基础 PR，后续实现的前置依赖） |

## 动机

Ling3 是 Ling2 架构的演进版本，同样采用混合 Transformer 架构（MLA + 线性注意力 + MoE），但将线性注意力层从 **GLA（Gated Linear Attention）** 升级为 **KDA（Kimi Delta Attention）**。本次重构旨在向 MaxText 的 Pydantic 配置系统中添加 Ling3 所需的新配置字段，以支持在 MaxText 框架下运行 Ling3 预训练任务。

本 PR 是**基础 PR** —— 仅添加配置字段定义和模型配置文件，不涉及任何行为代码变更。后续所有 PR（KDA 实现、decoder、checkpoint 转换等）都依赖这些字段的存在。

**设计约束：**

- **不修改** `base.yml` —— 所有 Ling3 专有的配置值放在 `ling3-tiny.yml` 中
- 每个新字段都有向后兼容的默认值（不改变现有用户的行为）
- **复用现有字段** —— 凡是能用已有字段表达的语义，不新增字段
- 参考 Ling2 的迁移经验，保持命名风格一致

## Ling3 与 Ling2 的架构差异

| 特性 | Ling2 | Ling3 |
|------|-------|-------|
| 线性注意力类型 | GLA (Gated Linear Attention) | KDA (Kimi Delta Attention) |
| 线性注意力配置 | `group_norm_size` | `linear_conv_kernel_dim`, `use_kda_lora`, `use_kda_safe_gate`, `kda_lower_bound` |
| 门控注意力 | 无 | **新增**：`enable_gated_attention` |
| MoE 层控制 | `first_num_dense_layers` + `interleave_moe_layer_step` | `first_k_dense_replace` + `moe_layer_freq`（可选） |
| 混合注意力分组 | `inhomogeneous_layer_cycle_interval` 设为 5（4 GLA + 1 MLA） | 复用同一字段设为 `4`（Megatron 中为 `layer_group_size`：3 KDA + 1 MLA） |
| 模型规模 | 20 层, hidden_size=2048 | 24 层, hidden_size=1536 |
| MoE 专家数 | 256 | 128 |
| 共享专家维度 | 2048 | 512 |
| 序列长度 | 4096 | 8192 |
| 位置编码扩展 | YaRN (rope_factor=32) | 原始 RoPE (rope_factor=1) |

## KDA（Kimi Delta Attention）说明

KDA 是 Kimi 团队提出的一种改进的线性注意力机制，是 Ling3 相比 Ling2 的核心升级。

**KDA 与 GLA 的主要区别：**

- KDA 不使用 GLA 的 `group_norm_size` 分组归一化
- KDA 使用 1D 卷积进行局部依赖建模（`linear_conv_kernel_dim`）
- KDA 使用 LoRA 分解控制（`use_kda_lora`）
- KDA 有专门的安全门控机制（`use_kda_safe_gate`, `kda_lower_bound`）

## 复用现有字段（不新增）

以下 Ling3 所需的语义已由上游现有字段覆盖，不需要在 PR1 中新增：

| 现有字段 | Ling3 用法 | 说明 |
|---------|-----------|------|
| `inhomogeneous_layer_cycle_interval` | 设为 `4`（每组 3 层 KDA + 1 层 MLA） | 已被 Llama4、Qwen3-Next、GPT-OSS 使用。Ling3 的混合注意力分组语义与之完全一致，对应 Megatron `layer_group_size` |
| `first_num_dense_layers` | 设为 `1`（第一层为 dense，其余为 MoE） | 配合 `interleave_moe_layer_step=1` 表达 Ling3 的 dense+MoE 层排布，对应 Megatron `first_k_dense_replace` |
| `mtp_final_layernorm` | 设为 `True` | 在 MTP（Multi-Token Prediction）层中应用最终层归一化，已在 Ling2 PR 中添加 |
| `mtp_per_layer_loss_norm` | 设为 `True` | 对 MTP 损失进行逐层归一化，已在 Ling2 PR 中添加 |

**关于 `inhomogeneous_layer_cycle_interval` 的说明：** Megatron 中使用 `layer_group_size=4`，功能与 `inhomogeneous_layer_cycle_interval` 等价。推上游时统一复用后者。

## 涉及文件

| 文件 | 变更 |
|------|------|
| `src/maxtext/common/common_types.py` | 在 `DecoderBlockType` 中添加 `LING3` 枚举值 |
| `src/maxtext/layers/nnx_decoders.py` | 在 `layer_map` 字典中添加 `DecoderBlockType.LING3` 占位条目 |
| `src/maxtext/configs/types.py` | 在 `KdaAttention` 类中添加 5 个 KDA 相关字段；在 `MlaAttention` 类中添加门控注意力开关字段；在 `ModelName` 字面量中添加 `"ling3"` |
| `src/maxtext/configs/models/ling3-tiny.yml` | **新文件** —— Ling3 模型配置 |
| `src/maxtext/configs/base.yml` | **不变** |

## 详细变更

### 1. `common_types.py` —— 新增枚举值

在 `DecoderBlockType` 中添加 `LING3`：

```python
class DecoderBlockType(enum.Enum):
    # ... 现有值 ...
    LING2 = "ling2"
    LING3 = "ling3"  # 新增：混合 MLA/KDA + MoE 架构
```

**向后兼容性：** 无影响。`LING3` 仅在通过 `decoder_block: "ling3"` 显式配置时才会激活。

### 2. `types.py` —— 新增 `ModelName` 条目

在 `ModelName` 字面量类型中添加 `"ling3"`：

```python
type ModelName = Literal[
    # ... 现有条目 ...
    "ling2",
    "ling3",  # 新增
]
```

### 3. `types.py` —— 新增配置字段

#### 3.1 `KdaAttention`（4 个字段）—— KDA 相关

**设计说明：** KDA（Kimi Delta Attention）是一种新的线性注意力机制，与 GLA 同属于线性注意力变体。为保持职责清晰，KDA 相关字段放在独立的 `KdaAttention` 类中（类似于 `MlaAttention`），而非复用 `LinearAttention` 或放在 `Qwen3Next` 中。GLA 相关字段保留在 `Qwen3Next` 类中不动。

```python
class KdaAttention(BaseModel):
    """KDA (Kimi Delta Attention) configuration.

    KDA is a linear attention mechanism used by Ling3, distinct from GLA (used by Ling2).
    These fields are placed in a separate class from MlaAttention for clear responsibility separation.
    """

    linear_conv_kernel_dim: int = Field(
        4,
        description=(
            "Convolution kernel dimension for linear attention layers (KDA). "
            "This specifies the size of the 1D convolution applied to keys for local dependency modeling. "
            "Ling3 uses 4 to match Megatron reference implementation."
        ),
    )
    use_kda_lora: bool = Field(
        False,
        description=(
            "Whether to use LoRA (Low-Rank Adaptation) style decomposition in KDA layers. "
            "When True, uses low-rank factorization for KDA computation. "
            "When False, uses full-rank projections. "
            "Ling3 uses this default to match Megatron reference implementation."
        ),
    )
    use_kda_safe_gate: bool = Field(
        False,
        description=(
            "Whether to use numerically safe gate computation in KDA layers. "
            "When True, applies value clamping and safe operations to prevent gate value explosion "
            "during training. Ling3 uses True for improved training stability."
        ),
    )
    kda_lower_bound: float = Field(
        0.0,
        description=(
            "Lower bound for gate values in KDA layers. Prevents gate values from "
            "becoming too small (highly negative) during training, which can cause numerical instability. "
            "Ling3 uses -5.0 as the lower bound."
        ),
    )
```

| 字段 | 默认值 | 向后兼容 | 说明 |
|------|--------|---------|------|
| `linear_conv_kernel_dim` | `4` | 是 —— KDA 卷积核维度 | KDA 卷积核维度 |
| `use_kda_lora` | `False` | 是 —— 默认不使用 LoRA 分解 | KDA LoRA 分解控制（默认关闭） |
| `use_kda_safe_gate` | `False` | 是 —— 默认不使用安全门控 | KDA 数值安全门控 |
| `kda_lower_bound` | `0.0` | 是 —— 0 表示无下界限制 | 门控值下界 |

#### 3.2 `MlaAttention`（1 个字段）—— 门控注意力相关

这是 **Ling3 特有的 MLA 门控输出**开关。与 GLA/KDA 的"门控"（Gated）不同，这里的门控是在 **MLA 层的输出上添加可学习的门控机制**。

**设计说明：** 经过对原始配置的分析，Ling3 的门控注意力配置固定为：

- `gated_attention_proj_granularity_type`: `head_wise`（固定值）
- `gated_attention_input_tensor_type`: `linear_qkv_input`（固定值）

因此，只需一个布尔开关 `enable_gated_attention` 来控制是否启用此功能。

```python
class MlaAttention(BaseModel):
    # ... 现有字段 ...
    enable_gated_attention: bool = Field(
        False,
        description=(
            "Whether to enable gated attention output for MLA layers in Ling3. "
            "When True, applies a learnable gating mechanism (sigmoid-activated projection) "
            "to the MLA attention output with per-head granularity (head_wise) using "
            "linear_qkv_input as the gate input. "
            "This is a Ling3-specific feature that adds a gate to MLA layer output, different from "
            "the 'Gated' in GLA (which gates the attention computation itself). "
            "Ling3 uses True for improved training stability and model quality. "
            "Note: When enabled, the granularity is fixed to 'head_wise' and input tensor "
            "is fixed to 'linear_qkv_input' as per Ling3 design."
        ),
    )
```

| 字段 | 默认值 | 向后兼容 | 说明 |
|------|--------|---------|------|
| `enable_gated_attention` | `False` | 是 —— 默认禁用 | MLA 输出门控开关（Ling3 设为 True） |

**与 GLA "Gated" 的区别：**

- GLA 的 "Gated" 是指注意力计算内部使用门控机制（Gated Linear Unit）
- Ling3 MLA 的 "Gated Attention" 是指在**注意力输出后**添加可学习的门控乘法（固定使用 head_wise 粒度和 linear_qkv_input 输入）
- 两者在不同的位置、服务于不同的目的

**注：** `mtp_final_layernorm` 和 `mtp_per_layer_loss_norm` 已在 Ling2 PR 中添加（`types.py` 第 467-477 行），本 PR 仅复用这些字段，无需新增。

### 4. 新增模型配置：`ling3-tiny.yml`

新文件用于 ling3_tiny 预训练 `src/maxtext/configs/models/ling3-tiny.yml`。当训练命令传入 `model_name=ling3` 时自动加载。

```yaml
# Ling3 model config (hybrid MLA/KDA + MoE architecture)
#
# Fields marked [NEW] require PR1 config extension in types.py.
# All other fields already exist in upstream MaxText.

# Core architecture
decoder_block: "ling3"
weight_dtype: 'float32'
dtype: 'bfloat16'
base_emb_dim: 1536
base_mlp_dim: 4608
base_num_decoder_layers: 24
mtp_num_layers: 1
vocab_size: 157184
first_num_dense_layers: 1

# Tokenizer IDs (tied to vocab, used by mmap data pipeline)
mmap_eod_id: 156892                    # <|endoftext|>
bos_id: 156891                        # <|startoftext|>

mlp_activations: ["silu", "linear"]
normalization_layer_epsilon: 1.0e-6

# Attention
base_num_query_heads: 16
base_num_kv_heads: 16
head_dim: 128
use_qk_norm: true
enable_dropout: false
logits_via_embedding: false

## MLA (Multi-Head Latent Attention)
q_lora_rank: 256
kv_lora_rank: 512
qk_nope_head_dim: 128
qk_rope_head_dim: 64
v_head_dim: 128
mscale: 1.0
mla_interleaved_rope: true             # de-interleave RoPE dims before rotary embedding
dropout_rate: 0.0
## Gated Attention [NEW] - Ling3 specific MLA output gating
enable_gated_attention: true                           # [NEW] enable gate on MLA output

# Hybrid attention (MLA + KDA)
# Every inhomogeneous_layer_cycle_interval layers, the last layer uses MLA (full attention),
# the rest use KDA (linear attention).
inhomogeneous_layer_cycle_interval: 4  # 3 KDA + 1 MLA per group

## KDA (Kimi Delta Attention) [NEW] - Ling3 linear attention
linear_conv_kernel_dim: 4             # [NEW] 1D conv kernel size for KDA
use_kda_lora: false                   # [NEW] LoRA in KDA (default=False)
use_kda_safe_gate: true               # [NEW] numerically safe gate
kda_lower_bound: -5.0                 # [NEW] gate value lower bound

# MoE (Mixture of Experts)
num_experts: 128
num_experts_per_tok: 8
base_moe_mlp_dim: 512
moe_shared_expert_dim: 512            # shared expert FFN width
shared_experts: 1
routed_scaling_factor: 2.5
n_routing_groups: 8
topk_routing_group: 4

# Routing
routed_score_func: "sigmoid"
routed_bias: true
routed_bias_dtype: "float32"          # keep bias in fp32 for stability
enable_routed_bias_grad: false        # freeze bias from optimizer; loss-free balancing only
routed_bias_update_rate: 0.001
routed_bias_zero_mean_update: true    # recenter bias to zero-mean after each update
norm_topk_prob: false

# Numerics
float32_logits: true

# RoPE
rope_min_timescale: 1
rope_max_timescale: 10000
max_position_embeddings: 8192
original_max_position_embeddings: 8192
rope_factor: 1                        # Ling3 uses original RoPE (no YaRN)
partial_rotary_factor: 0.5

# Sequence
max_target_length: 8192
max_prefill_predict_length: 8192
scan_layers: true

# Training
add_bos: false
calculate_per_token_loss: false       # micro-batch independent averaging for GA

## MTP (Multi-Token Prediction)
mtp_final_layernorm: true             # apply final LayerNorm in MTP
mtp_per_layer_loss_norm: true         # normalize loss per MTP layer
mtp_loss_scaling_factor: 0.1          # MTP loss weight
```

标注 `[NEW]` 的 key 需要 PR1 的 types.py 变更支持。其余 key 在上游已存在。

## Megatron 参数名称映射

### 新增字段映射（6 total）

| MaxText 字段 | Megatron 参数 | 说明 |
|--------------|--------------|------|
| `linear_conv_kernel_dim` | `linear_conv_kernel_dim` | 新增字段，KDA 卷积核维度 |
| `use_kda_lora` | `no_kda_lora` | 新增字段（反向语义），LoRA 分解（默认 false） |
| `use_kda_safe_gate` | `kda_safe_gate` | 新增字段，KDA 安全门控 |
| `kda_lower_bound` | `kda_lower_bound` | 新增字段，门控值下界 |
| `enable_gated_attention` | `enable_gated_attention` | 新增字段，MLA 输出门控开关（固定使用 head_wise + linear_qkv_input） |

### 字段命名差异（复用现有字段）

| MaxText 字段 | Megatron 参数 | 说明 |
|--------------|--------------|------|
| `base_emb_dim` | `hidden_size` | 命名不同 |
| `base_mlp_dim` | `ffn_hidden_size` | 命名不同 |
| `base_num_decoder_layers` | `num_layers` | 命名不同 |
| `base_num_query_heads` | `num_attention_heads` | 命名不同 |
| `base_num_kv_heads` | `num_query_groups` | 命名不同 |
| `head_dim` | `kv_channels` | 命名不同 |
| `inhomogeneous_layer_cycle_interval` | `layer_group_size` | 命名不同，功能等价 |
| `first_num_dense_layers` | `first_k_dense_replace` | 命名不同 |
| `num_experts_per_tok` | `moe_router_topk` | 命名不同 |
| `base_moe_mlp_dim` | `moe_ffn_hidden_size` | 命名不同 |
| `moe_shared_expert_dim` | `moe_shared_expert_intermediate_size` | 命名不同 |
| `n_routing_groups` | `moe_router_num_groups` | 命名不同 |
| `topk_routing_group` | `moe_router_group_topk` | 命名不同 |
| `routed_scaling_factor` | `moe_router_topk_scaling_factor` | 命名不同 |
| `routed_score_func` | `moe_router_score_function` | 命名不同 |
| `routed_bias` | `moe_router_enable_expert_bias` | 命名不同 |
| `routed_bias_dtype` | `moe_router_dtype` | 命名不同（fp32 vs float32） |
| `routed_bias_update_rate` | `moe_router_bias_update_rate` | 命名不同 |
| `routed_bias_zero_mean_update` | `bias_zero_mean_update` | 命名不同 |
| `moe_z_loss_weight` | `moe_z_loss_coeff` | 后缀不同 |
| `partial_rotary_factor` | `rotary_percent` | 命名不同 |
| `rope_max_timescale` | `rotary_base` | 语义等价 |
| `normalization_layer_epsilon` | `norm_epsilon` | 命名不同 |
| `use_qk_norm` | `qk_layernorm` | 命名不同 |

### Ling2 vs Ling3 关键参数值差异

| 参数 | Ling2 值 | Ling3 值 | Megatron 来源 |
|------|---------|----------|---------------|
| `base_num_decoder_layers` | 20 | 24 | `num_layers` |
| `base_emb_dim` | 2048 | 1536 | `hidden_size` |
| `base_mlp_dim` | 5120 | 4608 | `ffn_hidden_size` |
| `num_experts` | 256 | 128 | `num_experts` |
| `moe_shared_expert_dim` | 2048 | 512 | `moe_shared_expert_intermediate_size` |
| `inhomogeneous_layer_cycle_interval` | 5 | 4 | `layer_group_size` |
| `max_target_length` | 4096 | 8192 | `seq_length` |
| `max_position_embeddings` | 4096 | 8192 | `max_position_embeddings` |
| `rope_factor` | 32 | 1 | Ling3 不使用 YaRN |
| `enable_gated_attention` | `false` | `true` | Ling3 特有 MLA 输出门控开关 |
| `moe_z_loss_weight` | 3.5e-6 | 2.9e-6 | 7e-5 / num_layers |

## 向后兼容性

| 保证 | 机制 |
|------|------|
| 不破坏现有配置 | 所有 5 个新字段都有保持现有行为的安全默认值 |
| `base.yml` 不变 | 不修改共享默认配置文件 |
| `enable_gated_attention=False` 保持现有 MLA 行为 | 默认禁用门控，确保现有 MLA 模型不受影响 |
| 新枚举值 `LING3` 为惰性 | 仅在显式设置 `decoder_block: "ling3"` 时激活 |

## 测试方案

1. **现有测试套件通过**：运行 `python3 -m pytest -m "cpu_only"` —— 所有现有测试必须不受影响地通过

2. **配置加载验证**：验证 `ling3-tiny.yml` 能成功加载

   ```bash
   python3 -c "
   from maxtext.pyconfig import initialize_pydantic
   cfg = initialize_pydantic(['', 'src/maxtext/configs/base.yml', 'model_name=ling3'])
   assert cfg.decoder_block.value == 'ling3'
   assert cfg.linear_conv_kernel_dim == 4
   assert cfg.use_kda_lora == False
   assert cfg.use_kda_safe_gate == True
   assert cfg.kda_lower_bound == -5.0
   assert cfg.enable_gated_attention == True
   assert cfg.mtp_final_layernorm == True
   assert cfg.mtp_per_layer_loss_norm == True
   assert cfg.inhomogeneous_layer_cycle_interval == 4
   assert cfg.first_num_dense_layers == 1
   assert cfg.base_emb_dim == 1536
   assert cfg.base_num_decoder_layers == 24
   assert cfg.max_target_length == 8192
   print('OK')
   "
   ```

3. **默认值透传验证**：验证现有模型配置不受影响

   ```bash
   python3 -c "
   from maxtext.pyconfig import initialize_pydantic
   cfg = initialize_pydantic(['', 'src/maxtext/configs/base.yml', 'model_name=ling2'])
   assert cfg.enable_gated_attention == False  # 默认值（Ling2 无门控注意力）
   print('OK')
   "
   ```

4. **LING3 decoder block 注册验证**

   ```bash
   python3 -c "
   from maxtext.common.common_types import DecoderBlockType
   from maxtext.layers.nnx_decoders import Decoder
   # 验证 layer_map 包含 LING3 条目
   assert DecoderBlockType.LING3 in Decoder.layer_map, 'LING3 missing from layer_map'
   print('OK: LING3 registered in layer_map')
   "
   ```

## 后续工作

本 PR 仅添加配置字段，后续 PR 需要实现：

1. **KDA 注意力层实现**：在 `layers/attention.py` 中添加 KDA 注意力计算逻辑
2. **门控注意力实现**：在 MLA 层中添加门控机制
3. **Ling3 解码器实现**：在 `nnx_decoders.py` 中实现 `LING3` 解码器块
4. **Checkpoint 转换**：支持从 Megatron checkpoint 加载 Ling3 权重
5. **端到端测试**：验证 Ling3 模型的训练和推理正确性

## 附录：Ling3 架构图解

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ling3 层结构示意                                │
│                         (layer_group_size=4)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Layer 0 (Dense)     ┌───────────────────────────────────────────────┐  │
│                      │  Dense FFN (非 MoE)                           │  │
│                      └───────────────────────────────────────────────┘  │
│                                                                         │
│  Layers 1-3 (KDA)    ┌───────────────────────────────────────────────┐  │
│  ┌──────────────┐    │  KDA 线性注意力                               │  │
│  │  KDA Block 1 │    │  - linear_conv_kernel_dim=4                   │  │
│  │  KDA Block 2 │───▶│  - use_kda_lora=False                         │  │
│  │  KDA Block 3 │    │  - use_kda_safe_gate=True                     │  │
│  └──────────────┘    │  - kda_lower_bound=-5.0                       │  │
│                      └───────────────────────────────────────────────┘  │
│                                                                         │
│  Layer 4 (MLA+Gate)  ┌───────────────────────────────────────────────┐  │
│                      │  MLA 全注意力 + 输出门控（Ling3 特有）        │  │
│                      │  - enable_gated_attention=True                │  │
│                      │  - granularity: head_wise (固定)              │  │
│                      │  - input: linear_qkv_input (固定)             │  │
│                      └───────────────────────────────────────────────┘  │
│                              ↑                                          │
│                              │ 在 layers 4-23 重复此模式               │
│                              ↓                                          │
│                      （3 个循环，每组 3 KDA + 1 Gated MLA）             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

关键区分：
┌──────────────────┬──────────────────────────────────────────────────────┐
│ GLA "Gated"      │ 注意力计算内部使用门控机制（Gated Linear Attention） │
│ Ling3 "Gated"    │ 在 MLA 层输出后添加可学习的门控乘法（Output Gating） │
│ KDA              │ Kimi Delta Attention，线性注意力的一种新实现         │
└──────────────────┴──────────────────────────────────────────────────────┘
```
