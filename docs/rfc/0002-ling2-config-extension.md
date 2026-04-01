# RFC: Refactor PR1 - Ling2 配置扩展

| 字段     | 值                                 |
| -------- | ---------------------------------- |
| **作者** | @Garrybest                         |
| **日期** | 2026-03-26                         |
| **状态** | Draft                              |
| **PR**   | PR1（基础 PR，PR2-PR8 的前置依赖） |

## 动机

Ling2 是一种混合 Transformer 架构，结合了 MLA（Multi-Head Latent Attention）+ GLA（Gated Linear Attention）+ MoE（Mixture of Experts），最初在 ant-pretrain fork 中以 **ALModel**（`ALModel/al_model.py`，NNX 实现）的名称开发。本次重构将其统一命名为 **Ling2**，推回上游 MaxText。ant-pretrain 中与 ALModel 相关的代码（`DecoderBlockType.AL_MODEL`、`al_model.yml` 等）将在上游以 `LING2` / `ling2` 的名称出现。

为了将 Ling2 支持推回上游，需要向 MaxText 的 Pydantic 配置系统中添加 16 个新配置字段。

PR1 是**基础 PR** —— 仅添加配置字段定义和模型配置文件，不涉及任何行为代码变更。后续所有 PR（MoE 对齐、MMap 数据管道、GLA、MLA、Ling2 解码器、MTP、checkpoint 转换）都依赖这些字段的存在。

**设计约束：**

- **不修改** `base.yml` —— 所有 Ling2 专有的配置值放在 `ling2.yml` 中
- 每个新字段都有向后兼容的默认值（不改变现有用户的行为）
- 排除无用配置字段（已定义但未被任何模型代码引用的字段）
- **复用现有字段** —— 凡是能用已有字段表达的语义，不新增字段

## 复用现有字段（不新增）

以下 Ling2 所需的语义已由上游现有字段覆盖，不需要在 PR1 中新增：

| 现有字段                             | Ling2 用法                             | 说明                                                                                                        |
| ------------------------------------ | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `inhomogeneous_layer_cycle_interval` | 设为 `5`（每组 4 层 GLA + 1 层 MLA）   | 已被 Llama4、Qwen3-Next、GPT-OSS 使用。Ling2 的混合注意力分组语义与之完全一致                               |
| `first_num_dense_layers`             | 设为 `1`（第一层为 dense，其余为 MoE） | 已在 `DeepSeekMoE` 类中定义，配合 `interleave_moe_layer_step=1`（默认值）即可表达 Ling2 的 dense+MoE 层排布 |

**关于 `inhomogeneous_layer_cycle_interval` 的说明：** ant-pretrain fork 中曾使用 `layer_group_size` 字段，功能与 `inhomogeneous_layer_cycle_interval` 等价（`al_model.yml` 注释："equivalent to inhomogeneous_layer_cycle_interval"）。推上游时统一复用后者，避免引入冗余字段。checkpoint 转换代码中从 HF config 读取 `layer_group_size` 后映射到 `inhomogeneous_layer_cycle_interval` 即可。

**关于不引入 `moe_layer_freq` 的说明：** ant-pretrain fork 中曾使用 `moe_layer_freq` 列表提供逐层 MoE/dense 控制。但 Ling2 的实际模式是"第一层 dense，其余全 MoE"，完全可用 `first_num_dense_layers=1` + `interleave_moe_layer_step=1` 表达。引入第三种机制会导致三者优先级关系混乱，且需要额外扩展验证逻辑。

## 涉及文件

| 文件                                   | 变更                                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------- |
| `src/maxtext/common/common_types.py`   | 在 `DecoderBlockType` 中添加 `LING2` 枚举值                                               |
| `src/maxtext/layers/nnx_decoders.py`   | 在 `layer_map` 字典中添加 `DecoderBlockType.LING2` 占位条目（指向后续 PR 实现的解码器类） |
| `src/maxtext/configs/types.py`         | 在 7 个现有 Pydantic 类中添加 16 个字段；在 `ModelName` 字面量中添加 `"ling2"`            |
| `src/maxtext/configs/models/ling2.yml` | **新文件** —— Ling2 模型配置                                                              |
| `src/maxtext/configs/base.yml`         | **不变**                                                                                  |

## 详细变更

### 1. `common_types.py` —— 新增枚举值

在 `DecoderBlockType` 中添加 `LING2`：

```python
class DecoderBlockType(enum.Enum):
    # ... 现有值 ...
    LLAMA4 = "llama4"
    OLMO3 = "olmo3"
    LING2 = "ling2"  # 新增：混合 MLA/GLA + MoE 架构
```

**向后兼容性：** 无影响。现有代码路径匹配已知枚举值；`LING2` 仅在通过 `decoder_block: "ling2"` 显式配置时才会激活。

### 2. `types.py` —— 新增 `ModelName` 条目

在 `ModelName` 字面量类型中添加 `"ling2"`：

```python
type ModelName = Literal[
    # ... 现有条目 ...
    "olmo3-32b",
    "ling2",  # 新增
]
```

### 3. `types.py` —— 新增配置字段（7 个类共 16 个字段）

#### 3.1 `MoEGeneral`（1 个字段）

```python
class MoEGeneral(BaseModel):
    # ... 现有字段 ...
    moe_z_loss_weight: NonNegativeFloat = Field(
        0.0,
        description=(
            "Weight for MoE router z-loss. Adds a penalty on the log-sum-exp of "
            "router logits to encourage uniform routing. "
            "0.0 disables z-loss entirely (zero compute overhead when disabled)."
        ),
    )
```

| 字段                | 默认值 | 向后兼容                | 说明                                                         |
| ------------------- | ------ | ----------------------- | ------------------------------------------------------------ |
| `moe_z_loss_weight` | `0.0`  | 是 —— 0 表示禁用 z-loss | Ling2 使用 z-loss 稳定 MoE 路由；在 DeepSeek/Megatron 中常见 |

**与 `load_balance_loss_weight` 的区别：** 后者是负载均衡辅助损失的权重，`moe_z_loss_weight` 是 router logits z-loss 的权重（Switch Transformer 风格），两者独立。命名后缀 `_weight` 与 `load_balance_loss_weight` 保持一致。

#### 3.2 `DeepSeekMoE`（4 个字段）

```python
class DeepSeekMoE(BaseModel):
    # ... 现有字段 ...
    moe_shared_expert_dim: int = Field(
        0,
        description=(
            "Intermediate dimension for shared experts. When 0, falls back to "
            "base_moe_mlp_dim. Allows shared experts to have a different FFN width "
            "than routed experts."
        ),
    )
    routed_bias_dtype: str = Field(
        "",
        description=(
            "Data type for the routed gate bias parameter. Empty string means "
            "follow weight_dtype. Set to 'float32' to keep bias in higher precision "
            "for loss-free load balancing stability."
        ),
    )
    enable_routed_bias_grad: bool = Field(
        True,
        description=(
            "Whether the routed gate bias receives gradient-based optimizer updates. "
            "When False, the bias is frozen from optimizer updates and only modified "
            "via loss-free bias update (routed_bias_update_rate). Setting to False "
            "matches Megatron's default loss-free balancing behavior."
        ),
    )
    routed_bias_zero_mean_update: bool = Field(
        False,
        description=(
            "Whether to recenter routed expert bias to zero-mean after each "
            "loss-free bias update step. Prevents bias drift over long training runs."
        ),
    )
```

| 字段                           | 默认值  | 向后兼容                          | 说明                                                                 |
| ------------------------------ | ------- | --------------------------------- | -------------------------------------------------------------------- |
| `moe_shared_expert_dim`        | `0`     | 是 —— 0 回退到 `base_moe_mlp_dim` | Ling2 的共享专家使用不同的 FFN 宽度（2048）而非路由专家的宽度（512） |
| `routed_bias_dtype`            | `""`    | 是 —— 空字符串跟随 `weight_dtype` | Ling2 将 bias 保持在 fp32 以提高稳定性                               |
| `enable_routed_bias_grad`      | `True`  | 是 —— 现有行为允许梯度更新        | Ling2 禁用梯度更新，仅使用 loss-free 平衡                            |
| `routed_bias_zero_mean_update` | `False` | 是 —— 默认不做零均值重心化        | Ling2 启用以防止 bias 漂移                                           |

**`enable_routed_bias_grad` 命名说明：** 选择 `enable_*` 而非 `freeze_*` 是因为现有 `freeze_*` 仅出现在视觉编码器场景（`freeze_vision_encoder_params`），并非 MoE 路由领域的惯例。`routed_bias_*` 命名族群（`routed_bias`、`routed_bias_update_rate`、`routed_bias_dtype`）已统一使用正向语义。

#### 3.3 `MlaAttention`（1 个字段）

```python
class MlaAttention(BaseModel):
    # ... 现有字段 ...
    mla_interleaved_rope: bool = Field(
        True,
        description=(
            "Whether to de-interleave RoPE dimensions before applying rotary embedding "
            "in MLA. When True, reorders [d0, d1, d2, d3, ...] -> [d0, d2, ..., d1, d3, ...] "
            "to match the non-interleaved RoPE convention."
        ),
    )
```

| 字段                   | 默认值 | 向后兼容                                     | 说明                            |
| ---------------------- | ------ | -------------------------------------------- | ------------------------------- |
| `mla_interleaved_rope` | `True` | 是 —— 仅影响 `attention_type="mla"` 代码路径 | 控制 MLA 中 RoPE 维度的排列约定 |

**与 `rope_interleave` 的区别：** 两者控制不同层面。`rope_interleave`（`YarnRope` 类）控制 YarnRotaryEmbedding **内部**的 sin/cos 交错/拼接方式；`mla_interleaved_rope`（`MlaAttention` 类）控制 MLA 注意力层在调用 RoPE **之前**是否对输入维度做 de-interleave 重排。两者有组合关系：当 `mla_interleaved_rope=True` 且 YarnRotaryEmbedding 未自行处理 interleave 时，才做外部 de-interleave。

#### 3.4 `Qwen3Next`（2 个字段）

这些字段供 GLA（Gated Linear Attention）层使用。GLA 与 Gated Delta Net 架构同源，共享 `Qwen3Next` 配置命名空间。

```python
class Qwen3Next(BaseModel):
    # ... 现有字段 ...
    group_norm_size: int = Field(
        1,
        description=(
            "Group size for GroupRMSNorm applied to GLA output. "
            "A value of 1 is equivalent to standard RMSNorm (per-element normalization)."
        ),
    )
    use_linear_silu: bool = Field(
        False,
        description=(
            "Whether to apply SiLU activation on the fused QKV projection output "
            "in GLA layers before splitting into Q, K, V tensors."
        ),
    )
```

| 字段              | 默认值  | 向后兼容                     | 说明                                       |
| ----------------- | ------- | ---------------------------- | ------------------------------------------ |
| `group_norm_size` | `1`     | 是 —— `1` 等价于标准 RMSNorm | Ling2 使用 `4` 对 GLA 输出做分组归一化     |
| `use_linear_silu` | `False` | 是 —— 默认不应用 SiLU        | 精度对齐：Ling2 和 Megatron 均使用 `False` |

**`use_linear_silu` 与 `mlp_activations` 的区别：** `mlp_activations`（默认 `["silu", "linear"]`）控制 MLP（FFN）层的激活函数；`use_linear_silu` 控制 **GLA 注意力层的 QKV 投影**是否应用 SiLU（`attention_gla.py` 中使用）。两者作用域完全不同。

#### 3.5 `GrainDataset`（5 个字段）

这些字段为 Megatron mmap 数据管道（PR3）提供支持，在此预先声明以便 PR3 直接引用。

```python
class GrainDataset(BaseModel):
    # ... 现有字段 ...
    blend_cache_dir: PathStr = Field(
        "",
        description=(
            "Cache directory for auto-generated Megatron blend indices. "
            "When non-empty, generated index files are cached here for reuse across runs."
        ),
    )
    blend_index_dir: PathStr = Field(
        "",
        description=(
            "Directory containing pre-generated Megatron dataset_index.npy files. "
            "When set, skips index generation and loads from this directory directly."
        ),
    )
    reset_attention_mask: bool = Field(
        True,
        description=(
            "Controls segment ID generation when converting Megatron mmap packed data "
            "to MaxText format. When True, different documents within a packed sample "
            "receive separate segment IDs, preventing cross-document attention via "
            "MaxText's existing segment ID mechanism. When False, all tokens in a packed "
            "sample share the same segment ID, allowing cross-document attention."
        ),
    )
    eod_mask_loss: bool = Field(
        False,
        description=(
            "When True, end-of-document (EOD) tokens are excluded from loss calculation. "
            "Matches Megatron-LM's default behavior for mmap datasets."
        ),
    )
    mmap_split_sentences: bool = Field(
        False,
        description="Enable sentence-level splitting when loading mmap format data.",
    )
```

| 字段                   | 默认值  | 向后兼容                       | 说明                         |
| ---------------------- | ------- | ------------------------------ | ---------------------------- |
| `blend_cache_dir`      | `""`    | 是 —— 空表示不缓存             | 加速相同数据混合比的重复运行 |
| `blend_index_dir`      | `""`    | 是 —— 空表示即时生成索引       | 预生成索引避免启动开销       |
| `reset_attention_mask` | `True`  | 是 —— 现有行为阻止跨文档注意力 | packed 训练标准做法          |
| `eod_mask_loss`        | `False` | 是 —— EOD token 默认计入 loss  | 启用时匹配 Megatron 约定     |
| `mmap_split_sentences` | `False` | 是 —— 默认不分句               | 用于特定 mmap 数据预处理模式 |

**`reset_attention_mask` 与现有 segment ID 机制的关系：** 该字段不替代 MaxText 的 `decoder_segment_ids` 机制，而是控制 mmap 数据管道在将 Megatron 格式的 packed 数据转换为 MaxText 输入时，如何生成 segment ID。当 `True` 时，为不同文档分配不同 segment ID（复用 MaxText 已有的跨文档注意力隔离机制）；当 `False` 时，所有 token 共享同一 segment ID。

#### 3.6 `Tokenizer`（2 个字段）

```python
class Tokenizer(BaseModel):
    # ... 现有字段 ...
    pad_id: int = Field(
        0,
        description=(
            "Padding token ID. Used when grain_file_type='mmap' to avoid loading "
            "the tokenizer solely for the pad token ID."
        ),
    )
    bos_id: int = Field(
        1,
        description=(
            "Beginning-of-sentence token ID. Used when grain_file_type='mmap' with "
            "concat_then_split packing to insert BOS tokens between documents."
        ),
    )
```

| 字段     | 默认值 | 向后兼容                     | 说明                                       |
| -------- | ------ | ---------------------------- | ------------------------------------------ |
| `pad_id` | `0`    | 是 —— 仅在 mmap 代码路径使用 | 避免仅为获取 pad token ID 而加载 tokenizer |
| `bos_id` | `1`    | 是 —— 仅在 mmap 代码路径使用 | Llama 系列 tokenizer 的标准 BOS token      |

#### 3.7 `Optimizer`（1 个字段）

```python
class Optimizer(BaseModel):
    # ... 现有字段 ...
    calculate_per_token_loss: bool = Field(
        True,
        description=(
            "Controls gradient accumulation loss normalization. When True (default), "
            "accumulated gradients are divided by total token count across all "
            "micro-batches (global per-token average). When False, each micro-batch "
            "independently computes per-token average, then gradients are averaged "
            "with equal weight (1/N)."
        ),
    )
```

| 字段                       | 默认值 | 向后兼容                        | 说明                                            |
| -------------------------- | ------ | ------------------------------- | ----------------------------------------------- |
| `calculate_per_token_loss` | `True` | 是 —— 保留 MaxText 现有 GA 行为 | Ling2 设为 `False` 以匹配 Megatron 的 loss 对齐 |

**命名说明：** 虽然 MaxText 本身已做 per-token 归一化（`loss = total_loss / total_weights`），该字段控制的是梯度累积（GA）场景下的归一化策略差异：全局 token 平均 vs 微批次独立平均。

### 4. 新增模型配置：`ling2.yml`

新文件位于 `src/maxtext/configs/models/ling2.yml`。当训练命令传入 `model_name=ling2` 时自动加载。所有 key 均为 `MaxTextConfig` 字段的子集。

```yaml
# Ling2 模型配置（混合 MLA/GLA + MoE 架构）

# 核心架构参数
decoder_block: "ling2"
weight_dtype: 'float32'
dtype: 'bfloat16'
base_emb_dim: 2048
base_mlp_dim: 5120
base_num_decoder_layers: 20
mtp_num_layers: 1
vocab_size: 157184
first_num_dense_layers: 1
mlp_activations: ["silu", "linear"]
normalization_layer_epsilon: 1.0e-6

# 注意力配置
base_num_query_heads: 16
base_num_kv_heads: 16
head_dim: 128
use_qk_norm: true
enable_dropout: false
logits_via_embedding: false

## MLA（Multi-Head Latent Attention）参数
attention_type: "mla"
q_lora_rank: 256
kv_lora_rank: 512
qk_nope_head_dim: 128
qk_rope_head_dim: 64
v_head_dim: 128
mscale: 1.0
mla_interleaved_rope: true             # [NEW] 显式声明 RoPE de-interleave
dropout_rate: 0.0

# 混合注意力参数（MLA + GLA）
# 每 inhomogeneous_layer_cycle_interval 层中，最后一层使用 MLA（完整注意力）
# 其余层使用 GLA（线性注意力）
inhomogeneous_layer_cycle_interval: 5  # 每组 4 层 GLA + 1 层 MLA
group_norm_size: 4                     # [NEW] GLA 输出的分组归一化大小

# MoE（专家混合）参数
num_experts: 256
num_experts_per_tok: 8
base_moe_mlp_dim: 512
moe_shared_expert_dim: 2048            # [NEW] 共享专家 FFN 宽度
shared_experts: 1
routed_scaling_factor: 2.5
n_routing_groups: 8
topk_routing_group: 4

# 路由参数
routed_score_func: "sigmoid"
routed_bias: true
routed_bias_dtype: "float32"           # [NEW] 将 bias 保持在 fp32
enable_routed_bias_grad: false         # [NEW] 仅使用 loss-free 平衡
routed_bias_update_rate: 0.001
routed_bias_zero_mean_update: true     # [NEW] 防止 bias 漂移
norm_topk_prob: false

# 归一化与数值设置
float32_logits: true
use_linear_silu: false                 # [NEW]

# RoPE 设置
rope_min_timescale: 1
rope_max_timescale: 10000
max_position_embeddings: 4096
original_max_position_embeddings: 4096
rope_factor: 32
partial_rotary_factor: 0.5

# 序列设置
max_target_length: 4096
max_prefill_predict_length: 4096
scan_layers: false

# 训练配置
add_bos: false
calculate_per_token_loss: false        # [NEW] GA 归一化策略
```

标注 `[NEW]` 的 key 需要 PR1 的 types.py 变更支持。其余 key 在上游已存在。

## Megatron 参数名称映射

以下表格列出 Ling2 配置字段与 Megatron 对应参数的映射关系，供跨框架对照参考：

| MaxText 字段                   | Megatron 参数                           | 说明                            |
| ------------------------------ | --------------------------------------- | ------------------------------- |
| `moe_z_loss_weight`            | `--moe-z-loss-coeff`                    | 后缀对齐 MaxText `_weight` 惯例 |
| `calculate_per_token_loss`     | `--no-calculate-per-token-loss`         | 布尔语义相反                    |
| `use_linear_silu`              | `--no-linear-silu`                      | 布尔语义相反                    |
| `enable_routed_bias_grad`      | loss-free balancing 默认行为            | Megatron 默认不更新 bias 梯度   |
| `moe_shared_expert_dim`        | `--moe-shared-expert-intermediate-size` | 后缀对齐 MaxText `_dim` 惯例    |
| `routed_bias_zero_mean_update` | bias zero-mean recentering              | —                               |

## 向后兼容性

| 保证                                             | 机制                                                                  |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| 不破坏现有配置                                   | 所有 16 个新字段都有保持现有行为的安全默认值                          |
| `base.yml` 不变                                  | 不修改共享默认配置文件                                                |
| `moe_z_loss_weight=0.0` 零开销                   | 权重为 0 时完全跳过 z-loss 计算                                       |
| `calculate_per_token_loss=True` 保留现有 GA 行为 | 仅 Ling2 覆盖为 `False`                                               |
| 新枚举值 `LING2` 为惰性                          | 仅在显式设置 `decoder_block: "ling2"` 时激活                          |
| 不引入冗余字段                                   | 复用 `inhomogeneous_layer_cycle_interval` 和 `first_num_dense_layers` |

## 测试方案

1. **现有测试套件通过**：运行 `python3 -m pytest -m "cpu_only"` —— 所有现有测试必须不受影响地通过，因为新字段仅有向后兼容的默认值
1. **配置加载验证**：验证 `ling2.yml` 能成功加载：

   ```bash
   python3 -c "
   from maxtext.pyconfig import initialize_pydantic
   cfg = initialize_pydantic(['', 'src/maxtext/configs/base.yml', 'model_name=ling2'])
   assert cfg.decoder_block.value == 'ling2'
   assert cfg.inhomogeneous_layer_cycle_interval == 5
   assert cfg.first_num_dense_layers == 1
   assert cfg.moe_shared_expert_dim == 2048
   print('OK')
   "
   ```

1. **默认值透传验证**：验证现有模型配置不受影响：

   ```bash
   python3 -c "
   from maxtext.pyconfig import initialize_pydantic
   cfg = initialize_pydantic(['', 'src/maxtext/configs/base.yml', 'model_name=llama2-7b'])
   assert cfg.moe_z_loss_weight == 0.0
   assert cfg.calculate_per_token_loss == True
   assert cfg.moe_shared_expert_dim == 0
   print('OK')
   "
   ```

1. **LING2 decoder block 注册验证**：验证 `layer_map` 中已注册 `LING2` 条目（即使实现推迟到后续 PR）：

   ```bash
   python3 -c "
   from maxtext.common.common_types import DecoderBlockType
   from maxtext.layers.nnx_decoders import Decoder
   # 验证 layer_map 包含 LING2 条目
   assert DecoderBlockType.LING2 in Decoder.layer_map, 'LING2 missing from layer_map'
   print('OK: LING2 registered in layer_map')
   "
   ```
