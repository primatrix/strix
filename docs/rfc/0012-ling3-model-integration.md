# RFC-0012: Ling3 模型集成实现方案

| 字段     | 值                              |
| -------- | ------------------------------- |
| **作者** | @Garrybest                      |
| **日期** | 2026-04-15                      |
| **状态** | Draft                           |
| **前置** | Ling2 系列 PR（PR1-PR10）已合并 |

## 概述

在 MaxText 中集成 Ling3 模型，包括 Ling3 解码层组装、ScannableBlock scan 层支持、Decoder 分发逻辑及 MTP 适配。本 RFC 聚焦于模型的**组装和适配接入**，KDA 注意力层实现、MLA 门控输出修改及配置扩展分别在独立 PR 中完成。

## 背景

Ling3 是 Ling2（百灵 v2.5）的架构演进，两者的核心差异：

1. **KDA 替换 GLA**：非 MLA 层从 GLA（Gated Linear Attention）切换到 KDA（Kimi Delta Attention）
2. **MLA 门控输出**：MLA 注意力输出新增 sigmoid 门控机制（`head_wise` 粒度）
3. **层配置差异**：Ling3-tiny 24 层（group_size=4），Ling3-flash 42 层（group_size=6），Ling3-flash 前 2 层 dense
4. **QK_Clip**：训练时在 MLA 中启用 QK Clipping
5. **Scan 层支持**：Ling2 不支持 `scan_layers=True`，Ling3 通过 ScannableBlock 支持
6. **Muon 优化器**：Ling3 使用不同优化器（不在本 RFC 范围）

**设计目标**：最小改动、最大复用现有架构，同时保持 Ling2 完全向后兼容。

### 模型配置对比

| 参数 | Ling2 | Ling3-tiny | Ling3-flash |
|------|-------|------------|-------------|
| `num_decoder_layers` | 20 | 24 | 42 |
| `inhomogeneous_layer_cycle_interval` | 5（4 GLA + 1 MLA） | 4（3 KDA + 1 MLA） | 6（5 KDA + 1 MLA） |
| `first_num_dense_layers` | 1 | 1 | 2 |
| 非 MLA 注意力 | GLA | KDA | KDA |
| MLA 门控 | 无（`enable_gated_attention: false`） | 启用（`enable_gated_attention: true`） | 启用（`enable_gated_attention: true`） |
| `use_qk_clip` | false | true | true |
| `scan_layers` | false（不支持 true） | true/false | true/false |

### 层排布示意

**Ling3-tiny**（24 层，group_size=4，first_num_dense_layers=1，unscan_prefix=4）：

```text
Unscan 前缀（4 层）：
  Layer 0  [KDA + Dense]   Layer 1  [KDA + MoE]   Layer 2  [KDA + MoE]   Layer 3  [MLA + MoE]
---
Scan ScannableBlock（5 个 block × 4 层）：
  Block 0 (scan iter 0): Layer 4  [KDA + MoE]  Layer 5  [KDA + MoE]  Layer 6  [KDA + MoE]  Layer 7  [MLA + MoE]
  Block 1 (scan iter 1): Layer 8  [KDA + MoE]  Layer 9  [KDA + MoE]  Layer 10 [KDA + MoE]  Layer 11 [MLA + MoE]
  ...
  Block 4 (scan iter 4): Layer 20 [KDA + MoE]  Layer 21 [KDA + MoE]  Layer 22 [KDA + MoE]  Layer 23 [MLA + MoE]
```

> Unscan 前缀扩展到 cycle 边界（`ceil(1/4)*4 = 4` 层），确保 scan 区域能被 interval 整除。详见 4.3.3 节。

**Ling3-flash**（42 层，group_size=6，first_num_dense_layers=2，unscan_prefix=6）：

```text
Unscan 前缀（6 层）：
  Layer 0  [KDA + Dense]  Layer 1  [KDA + Dense]  Layer 2-5  [KDA/MLA + MoE]
---
Scan ScannableBlock（6 个 block × 6 层）：
  Block 0: Layer 6-11   (5 KDA + MoE, 1 MLA + MoE)
  Block 1: Layer 12-17  (5 KDA + MoE, 1 MLA + MoE)
  ...
```

## 前置依赖

本 RFC 聚焦于模型层的组装和适配接入。以下组件由独立 PR 实现，本 RFC 假设它们已就绪：

| 依赖 | 说明 | 参考 |
|------|------|------|
| **KDA 注意力层** (`attention_kda.py`) | `KDAAttention` 类，与 GLA (`BailingMoeV2LinearAttention`) 接口兼容。初期复制 GLA 代码作占位，KDA 算子就绪后替换内部实现。接口契约：`__call__(hidden_states, decoder_positions, deterministic, model_mode, *, layer_idx, decoder_segment_ids) -> (output, None)` | 独立 PR |
| **MLA 门控输出** (`attention_mla.py` 修改) | `MLA` 类新增 `mla_g_proj` 门控投影，在注意力输出后、输出投影前应用 `sigmoid` 门控（固定 `head_wise` 粒度）。由 `enable_gated_attention` 配置控制，默认 `False` 禁用。 | 独立 PR |
| **配置扩展** (`types.py` + `ling3.yml`) | `LING3` 枚举值、KDA/MLA 门控相关配置字段、模型 YAML 配置 | [RFC-0007](./0007-ling3-config-extension.md)、[PR #64](https://github.com/primatrix/wiki/pull/64) |

## 方案

### 涉及文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/maxtext/common/common_types.py` | 修改 | 添加 `LING3` 枚举值 |
| `src/maxtext/models/ling3.py` | **新文件** | Ling3 解码层 + ScannableBlock |
| `src/maxtext/layers/decoders.py` | 修改 | 添加 `LING3` 分发逻辑（scan + unscan） |
| `src/maxtext/layers/multi_token_prediction.py` | 修改 | 添加 `LING3` MTP 适配 |
| `tests/unit/ling3_decoder_test.py` | **新文件** | 单元测试 |

---

### 4.1 配置扩展

#### `common_types.py` — 新增枚举值

```python
class DecoderBlockType(enum.Enum):
    # ... 现有值 ...
    LING2 = "ling2"
    LING3 = "ling3"  # 新增：KDA + 门控 MLA + MoE 架构
```

> **注**：其他配置字段（`enable_gated_attention`、KDA 相关字段、`ling3-tiny.yml` / `ling3-flash.yml` 模型配置）的设计见 [RFC-0007: Ling3 配置扩展](./0007-ling3-config-extension.md)（[PR #64](https://github.com/primatrix/wiki/pull/64)），不在本 RFC 范围内。

---

### 4.2 Ling3 解码层设计（`ling3.py`）

#### 类层次结构

```text
Ling3GenericLayer(nnx.Module)           # 基类：KDA/MLA 注意力 + Pre/Post RMSNorm + 残差
├── Ling3DenseDecoderLayer               # 子类：Dense MLP（MlpBlock）
└── Ling3MoEDecoderLayer                 # 子类：MoE MLP（RoutedAndSharedMoE）

Ling3ScannableBlock(nnx.Module)          # ScannableBlock：捆绑 N 个异构层（N = inhomogeneous_layer_cycle_interval，Ling3-tiny 为 4，Ling3-flash 为 6）

# Linen 包装器
Ling3DenseDecoderLayerToLinen            # to_linen_class(Ling3DenseDecoderLayer)
Ling3MoEDecoderLayerToLinen              # to_linen_class(Ling3MoEDecoderLayer)
Ling3ScannableBlockToLinen               # to_linen_class(Ling3ScannableBlock)
```

#### `Ling3GenericLayer` — 与 `Ling2GenericLayer` 的差异

唯一的结构差异在 `__init__` 中的非 MLA 注意力选择：

```python
class Ling3GenericLayer(nnx.Module):
  """Ling3 基础解码层。

  与 Ling2GenericLayer 的差异:
    1. 非 MLA 层使用 KDA（KDAAttention）而非 GLA（BailingMoeV2LinearAttention）
    2. MLA 层门控输出由 config.enable_gated_attention 自动控制（MLA 内部实现）
  """

  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    # ... 与 Ling2GenericLayer.__init__ 结构相同 ...

    is_full_attention_layer = (
        (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
        or self.layer_idx >= cfg.num_decoder_layers
    )
    if is_full_attention_layer:
      # MLA — 门控由 config.enable_gated_attention 在 MLA 内部自动启用
      self.attention = attention_mla.MLA(
          config=cfg,
          # ... 与 Ling2 完全相同的参数列表 ...
      )
    else:
      # *** 关键差异：KDA 而非 GLA ***
      self.attention = attention_kda.KDAAttention(
          config=cfg,
          layer_idx=self.layer_idx,
          mesh=mesh,
          rngs=rngs,
      )

  def __call__(self, inputs, decoder_segment_ids, decoder_positions,
               deterministic, model_mode, ..., global_layer_idx=None, ...):
    # 前向传播逻辑与 Ling2GenericLayer.__call__ 完全相同
    # 因为 KDA 和 GLA 具有相同的调用接口
    if isinstance(self.attention, attention_mla.MLA):
      attention_output, kv_cache = self.attention(hidden_states, hidden_states, ...)
    else:
      attention_output, _ = self.attention(
          hidden_states, decoder_positions, deterministic, model_mode,
          layer_idx=global_layer_idx,
      )
      kv_cache = None
    # ... 残差 → PostNorm → MLP → 残差 → post_process ...
```

**设计决策 — 为什么不继承 Ling2GenericLayer**：

1. **注意力类型差异**：`__init__` 中 KDA vs GLA 是核心分支差异。继承后覆盖会比平行实现更脆弱。
2. **向后兼容**：独立文件确保 Ling2 代码零改动，降低回归风险。
3. **Scan 支持**：Ling3 的 ScannableBlock 需要在 `ling3.py` 中定义，独立模块更清晰。
4. **代码量可控**：`Ling2GenericLayer` 本身约 300 行，核心逻辑（`__call__` + `post_process` + properties）模式固定，平行实现的维护成本低。

#### `Ling3DenseDecoderLayer` 和 `Ling3MoEDecoderLayer`

与 Ling2 版本结构完全相同，仅父类改为 `Ling3GenericLayer`：

```python
class Ling3DenseDecoderLayer(Ling3GenericLayer):
  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    super().__init__(config, mesh, model_mode, layer_idx, quant, rngs=rngs)
    cfg = self.config
    self.mlp = linears.MlpBlock(
        config=cfg, mesh=mesh, in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim, activations=cfg.mlp_activations,
        # ... 与 Ling2DenseDecoderLayer 完全相同 ...
    )

class Ling3MoEDecoderLayer(Ling3GenericLayer):
  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    super().__init__(config, mesh, model_mode, layer_idx, quant, rngs=rngs)
    cfg = self.config
    self.mlp = moe.RoutedAndSharedMoE(
        config=cfg, mesh=mesh,
        # ... 与 Ling2MoEDecoderLayer 完全相同 ...
    )

# Linen 包装器
Ling3DenseDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Ling3DenseDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
Ling3MoEDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Ling3MoEDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
```

---

### 4.3 ScannableBlock 设计（Scan 层支持）

#### 4.3.1 参考模式分析

MaxText 中已有三种 scan 模式：

| 模式 | 代表模型 | 特点 |
|------|----------|------|
| 同构 scan | Llama2、Gemma | 所有层结构相同，直接 scan |
| ScannableBlock | Llama4、OLMo3、Qwen3-Next | 捆绑 N 个异构层为一个 block，scan 重复 block（此处 N = `inhomogeneous_layer_cycle_interval`，非固定值） |
| 两组 scan | DeepSeek | 分别 scan dense 层和 MoE 层 |

Ling3 的层排布兼具两种异构性：MLP 异构（Dense vs MoE，与 DeepSeek 相同）和注意力异构（KDA vs MLA，与 Llama4 类似）。因此采用**混合方案**：借鉴 DeepSeek 的 dense/MoE 分组命名，同时在 MoE 区域引入 ScannableBlock 解决注意力异构。

#### 4.3.2 核心挑战

**挑战 1：MoE 区域的注意力异构**。DeepSeek 所有层使用同一种注意力（MLA），MoE 层可逐层 scan。Ling3 的 MoE 层交替使用 KDA 和 MLA，参数 shape 不同，必须用 ScannableBlock 把一个周期打包。

**挑战 2：MoE 区域不能整除 cycle interval**。`(24-1)/4=5.75`，`(42-2)/6=6.67`，均不整除。

#### 4.3.3 解决方案：两阶段 scan

Dense 前缀层数量很少（Ling3-tiny 1 层、Ling3-flash 2 层），scan 收益微乎其微，反而引入参数堆叠/解包的编译开销。因此 dense 前缀和 MoE 过渡层统一走 unscan 路径，仅对 MoE ScannableBlock 做 scan。

将 unscan 前缀扩展到下一个 `interval` 边界，使 ScannableBlock 区域能整除：

```python
interval = cfg.inhomogeneous_layer_cycle_interval
if cfg.first_num_dense_layers > 0:
  unscan_prefix = ((cfg.first_num_dense_layers + interval - 1) // interval) * interval
else:
  unscan_prefix = 0
num_moe_prefix = unscan_prefix - cfg.first_num_dense_layers
scan_layers_count = cfg.num_decoder_layers - unscan_prefix
scan_length = scan_layers_count // interval
```

两阶段执行流程（层名前缀与 DeepSeek 保持一致，便于权重转换）：

```text
Phase 1: unscan 前缀层
  → dense_layers_0, dense_layers_1, ...  （Dense 层，数量少无需 scan）
  → moe_layers_0, moe_layers_1, ...     （MoE 过渡层，到 cycle 边界）

Phase 2: scan("moe_layers", ScannableBlock, scan_length)
  → MoE 区域的 ScannableBlock，每 block 含 interval 个 MoE 层
```

以 Ling3-tiny 为例（24 层，dense=1，interval=4）：

```text
Phase 1: unscan 前缀                    dense_layers_0 [KDA+Dense], moe_layers_0 [KDA+MoE], moe_layers_1 [KDA+MoE], moe_layers_2 [MLA+MoE]
Phase 2: scan("moe_layers", Block, 5)   Layers 4-23, 每 block [KDA,KDA,KDA,MLA]+MoE
```

ScannableBlock 内部统一使用 MoE MLP（dense 前缀已在 Phase 1 处理完毕）。

```python
class Ling3ScannableBlock(nnx.Module):
  """Ling3 可扫描块，捆绑 inhomogeneous_layer_cycle_interval 个 MoE 层。

  注意力类型按组内位置决定：前 (interval-1) 个为 KDA，最后一个为 MLA。
  MLP 统一为 MoE（dense 前缀层在 Decoder.__call__ 中 unscan 处理）。
  """

  def __init__(self, config, mesh, model_mode, rngs, quant=None):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    for layer_id in range(config.inhomogeneous_layer_cycle_interval):
      layer = Ling3MoEDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=layer_id,  # 组内索引，决定注意力类型
          quant=quant,
          rngs=rngs,
      )
      setattr(self, f"layers_{layer_id}", layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
  ):
    cfg = self.config
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    y = inputs
    for layer_id in range(cfg.inhomogeneous_layer_cycle_interval):
      y = getattr(self, f"layers_{layer_id}")(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
      )
      y = y[0]  # 解包 (output, None) tuple，无论 scan/unscan 始终解包以保证健壮性
    if cfg.scan_layers:
      return y, None
    else:
      return y

Ling3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Ling3ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
```

**注意 `layer_idx` 的处理**：ScannableBlock 内部的 `layer_idx` 是组内索引（0 到 interval-1），用于决定注意力类型（KDA vs MLA）。KDA 的 slope 计算需要全局层索引，但在 scan 模式下无法直接获取。解决方案：

- 在 scan 模式下，KDA 的 slope 使用组内相对索引计算（或使用固定 slope）
- 在 unscan 模式下，通过 `global_layer_idx` kwarg 传递真实全局索引
- 这与 Ling2 当前 GLA 的处理方式一致（scan 模式下 slope 使用构造时的 layer_idx）

---

### 4.4 Decoder 层分发逻辑（`decoders.py`）

#### `get_decoder_layers()`

scan 和 unscan 模式均返回 Dense + MoE 两种层类型（scan 模式额外需要 ScannableBlock）：

```python
case DecoderBlockType.LING3:
    if self.config.scan_layers:
        # 顺序约定：MoE 层必须在最后，使 layer_types[-1] 始终是
        # Ling3MoEDecoderLayerToLinen。MTP（见 4.5）依赖 [-1] 拿到带 MLA
        # 门控的 MoE 层，而非 ScannableBlock（其内部含 KDA+MLA 异构层）。
        return [ling3.Ling3DenseDecoderLayerToLinen,
                ling3.Ling3ScannableBlockToLinen,
                ling3.Ling3MoEDecoderLayerToLinen]
    return [ling3.Ling3DenseDecoderLayerToLinen, ling3.Ling3MoEDecoderLayerToLinen]
```

#### Scan 路径 — `_apply_ling3_scan_layers()`

将 Ling3 的两阶段 scan 逻辑抽为独立方法：

```python
def _apply_ling3_scan_layers(self, cfg, mesh, policy, model_mode, y, broadcast_args):
    """Ling3 两阶段 scan：unscan 前缀 → ScannableBlock scan。"""
    interval = cfg.inhomogeneous_layer_cycle_interval

    # 计算 unscan 前缀（扩展到 cycle 边界）
    if cfg.first_num_dense_layers > 0:
        unscan_prefix = ((cfg.first_num_dense_layers + interval - 1) // interval) * interval
    else:
        unscan_prefix = 0
    assert unscan_prefix < cfg.num_decoder_layers, (
        f"unscan_prefix ({unscan_prefix}) >= num_decoder_layers ({cfg.num_decoder_layers}): "
        f"first_num_dense_layers ({cfg.first_num_dense_layers}) 不能覆盖全部解码层"
    )
    num_moe_prefix = unscan_prefix - cfg.first_num_dense_layers

    dense_layer, scannable_block, moe_layer = self.set_remat_policy(
        [ling3.Ling3DenseDecoderLayerToLinen,
         ling3.Ling3ScannableBlockToLinen,
         ling3.Ling3MoEDecoderLayerToLinen], policy
    )

    # Phase 1: unscan 前缀层（dense 层数量少，不值得 scan）
    for idx in range(cfg.first_num_dense_layers):
        y, _ = dense_layer(
            config=cfg, mesh=mesh,
            name=f"dense_layers_{idx}",
            quant=self.quant,
            model_mode=model_mode,
            layer_idx=idx,
        )(y, *broadcast_args)

    for idx in range(num_moe_prefix):
        global_idx = cfg.first_num_dense_layers + idx
        y, _ = moe_layer(
            config=cfg, mesh=mesh,
            name=f"moe_layers_{idx}",
            quant=self.quant,
            model_mode=model_mode,
            layer_idx=global_idx,
        )(y, *broadcast_args)

    # Phase 2: scan MoE ScannableBlock
    scan_layers_count = cfg.num_decoder_layers - unscan_prefix
    scan_length = scan_layers_count // interval
    y, _ = self.scan_decoder_layers(
        cfg, scannable_block, scan_length,
        "moe_layers", mesh,
        in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
        model_mode=model_mode,
    )(y, *broadcast_args)

    return y
```

在 `Decoder.__call__` 中调用：

```python
elif cfg.decoder_block == DecoderBlockType.LING3 and cfg.scan_layers:
    y = self._apply_ling3_scan_layers(cfg, mesh, policy, model_mode, y, broadcast_args)
```

#### Unscan 路径

复用 DeepSeek 的 dense+MoE 两组遍历逻辑，使用 `dense_layers` / `moe_layers` 命名：

```python
# 在现有的 DEEPSEEK/LING2 unscan 分支中添加 LING3
if cfg.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2, DecoderBlockType.LING3):
    layers = [dense_layer, moe_layer]
    layer_prefixes = ["dense_layers", "moe_layers"]  # 与 DeepSeek 一致
    num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
    num_layers_list = [cfg.first_num_dense_layers, num_moe_layers]
    # ... 按 prefix 遍历各组 ...
    if cfg.decoder_block in (DecoderBlockType.LING2, DecoderBlockType.LING3):
        layer_call_kwargs["global_layer_idx"] = global_layer_idx
```

#### `get_norm_layer()`

添加 LING3：

```python
def get_norm_layer(self, num_features):
    if self.config.decoder_block in (
        # ... 现有 block types ...
        DecoderBlockType.LING2,
        DecoderBlockType.LING3,  # 新增
    ):
        return functools.partial(rms_norm, ...)
```

---

### 4.5 MTP 适配

#### 当前 Ling2 行为

MTP 使用 `get_decoder_layers()` 返回的最后一个层类型（`Ling2MoEDecoderLayerToLinen`）。`layer_idx = cfg.num_decoder_layers + k - 1` 确保 MTP 层始终使用 MLA 注意力。

> 在 Ling3 的 scan 模式下，`get_decoder_layers()` 返回 `[Dense, ScannableBlock, MoE]`（见 4.4），保证 `layer_types[-1] == Ling3MoEDecoderLayerToLinen`。这是必要的——MTP 需要单层 MoE 蓝图，而不是包含 KDA+MLA 异构层的 ScannableBlock。

#### Ling3 适配

```python
# multi_token_prediction.py
# 在设置 MTP layer_idx 的条件中添加 LING3
if cfg.decoder_block in (DecoderBlockType.LING2, DecoderBlockType.LING3):
    layer_idx_kwargs["layer_idx"] = cfg.num_decoder_layers + k - 1
```

MTP 层行为：

- 使用 `Ling3MoEDecoderLayerToLinen`（`layer_types[-1]`）
- `layer_idx >= num_decoder_layers` → 触发 MLA 注意力（带 head_wise 门控）
- 门控行为与主模型一致，因为 `enable_gated_attention` 是全局配置

---

### 4.6 权重命名与 Checkpoint 转换映射

> **注**：Checkpoint 转换的具体实现不在本 RFC 范围，但层命名直接影响权重路径映射，需在此明确约定。

#### 命名规则

MaxText 中 scan/unscan 两种模式下参数路径不同：

| 模式 | 路径格式 | 参数 shape |
|------|----------|-----------|
| Unscan（逐层） | `decoder/{prefix}_{i}/...` | 原始 shape |
| Scan（堆叠） | `decoder/{scan_name}/...` | 在 `param_scan_axis=1` 处插入 scan 维度 |
| Scan + ScannableBlock | `decoder/{scan_name}/layers_{intra_idx}/...` | 在 `param_scan_axis=1` 处插入 scan 维度 |

#### HF → MaxText 层索引映射公式

```python
unscan_prefix = ceil(first_num_dense_layers / interval) * interval
num_moe_prefix = unscan_prefix - first_num_dense_layers

for hf_layer_idx in range(num_decoder_layers):
    if hf_layer_idx < first_num_dense_layers:
        # Dense 前缀层
        maxtext_name = f"dense_layers_{hf_layer_idx}"
    elif hf_layer_idx < unscan_prefix:
        # MoE 过渡层（unscan）
        maxtext_name = f"moe_layers_{hf_layer_idx - first_num_dense_layers}"
    else:
        if scan_layers:
            # ScannableBlock 内部层
            block_idx = (hf_layer_idx - unscan_prefix) // interval
            intra_idx = (hf_layer_idx - unscan_prefix) % interval
            maxtext_name = f"moe_layers/layers_{intra_idx}"  # scan_axis[block_idx]
        else:
            # 普通 MoE 层（unscan）
            maxtext_name = f"moe_layers_{hf_layer_idx - first_num_dense_layers}"
```

#### Ling3-tiny 具体映射（24 层，dense=1，interval=4）

**Unscan 模式**（所有层独立命名）：

| HF 层 | 全局索引 | 注意力 | MLP | MaxText 参数路径 |
|--------|---------|--------|-----|-----------------|
| `layers.0` | 0 | KDA | Dense | `decoder/dense_layers_0/...` |
| `layers.1` | 1 | KDA | MoE | `decoder/moe_layers_0/...` |
| `layers.2` | 2 | KDA | MoE | `decoder/moe_layers_1/...` |
| ... | ... | ... | MoE | `decoder/moe_layers_{i-1}/...` |
| `layers.23` | 23 | MLA | MoE | `decoder/moe_layers_22/...` |

**Scan 模式**（前缀 unscan + ScannableBlock scan）：

| HF 层 | 全局索引 | 阶段 | MaxText 参数路径 |
|--------|---------|------|-----------------|
| `layers.0` | 0 | Phase 1 unscan | `decoder/dense_layers_0/...` |
| `layers.1` | 1 | Phase 1 unscan | `decoder/moe_layers_0/...` |
| `layers.2` | 2 | Phase 1 unscan | `decoder/moe_layers_1/...` |
| `layers.3` | 3 | Phase 1 unscan | `decoder/moe_layers_2/...` |
| `layers.4` | 4 | Phase 2 scan | `decoder/moe_layers/layers_0/...` `[scan_idx=0]` |
| `layers.5` | 5 | Phase 2 scan | `decoder/moe_layers/layers_1/...` `[scan_idx=0]` |
| `layers.6` | 6 | Phase 2 scan | `decoder/moe_layers/layers_2/...` `[scan_idx=0]` |
| `layers.7` | 7 | Phase 2 scan | `decoder/moe_layers/layers_3/...` `[scan_idx=0]` |
| `layers.8` | 8 | Phase 2 scan | `decoder/moe_layers/layers_0/...` `[scan_idx=1]` |
| ... | ... | ... | ... |
| `layers.23` | 23 | Phase 2 scan | `decoder/moe_layers/layers_3/...` `[scan_idx=4]` |

> `[scan_idx=i]` 表示参数张量在 `param_scan_axis=1` 维度上的第 i 个切片。scan 模式下，`moe_layers/layers_{k}` 的每个参数 shape 从 `[d1, d2, ...]` 变为 `[d1, scan_length, d2, ...]`，其中 `scan_length=5`。

#### 命名冲突说明

Unscan 前缀的 `moe_layers_0`、`moe_layers_1` 等与 scan 的 `moe_layers` 是不同的 Flax module name（带下标 vs 不带下标），不会产生命名冲突。

---

## 向后兼容性

| 保证 | 机制 |
|------|------|
| Ling2 零影响 | 独立 `DecoderBlockType.LING3`，不修改 Ling2 任何代码 |
| `base.yml` 不变 | 新默认值通过 `types.py` Field 默认值设置 |
| 现有 scan 模型零影响 | `LING3` 的 scan 路径是独立的 case 分支 |

## 测试方案

### 单元测试（`tests/unit/ling3_decoder_test.py`）

```python
class Ling3DecoderLayerTest:
  def test_kda_layer_construction(self):
    """非 MLA 位置正确构造 KDA 注意力层"""

  def test_mla_gated_layer_construction(self):
    """MLA 位置正确构造带门控的 MLA 注意力层"""

  def test_scannable_block_layer_count(self):
    """ScannableBlock 包含 inhomogeneous_layer_cycle_interval 个层"""

  def test_scannable_block_attention_types(self):
    """ScannableBlock 内 KDA/MLA 分布正确（前 interval-1 个 KDA，最后 1 个 MLA；interval = inhomogeneous_layer_cycle_interval）"""

  def test_unscan_dense_moe_split(self):
    """unscan 模式下 dense 前缀 + MoE 后缀正确构造"""

  def test_scan_prefix_and_block(self):
    """scan 模式下 unscan 前缀 + ScannableBlock 组合正确执行"""

  def test_unscan_prefix_boundary_zero_dense(self):
    """first_num_dense_layers=0 时 unscan_prefix=0，全部层进入 scan"""

  def test_unscan_prefix_boundary_exceeds_total(self):
    """first_num_dense_layers >= num_decoder_layers 时应触发断言错误"""

  def test_mtp_uses_gated_mla(self):
    """MTP 层使用带门控的 MLA 注意力"""

  def test_mtp_layer_blueprint_is_moe_in_scan_mode(self):
    """scan_layers=True 时，layer_types[-1] 必须是 Ling3MoEDecoderLayerToLinen，
    而不是 Ling3ScannableBlockToLinen（否则 MTP 会拿到含 KDA 的异构 block）"""
```

### 集成测试

1. **前向传播**：小规模 Ling3 配置完整前向传播
2. **训练 step**：验证一个训练 step 的 loss 计算
3. **Scan vs Unscan 对齐**：比较 `scan_layers=True/False` 的输出一致性
4. **Ling2 回归**：确保现有 Ling2 测试全部通过

---

## PR 拆分建议

| PR | 范围 | 依赖 |
|----|------|------|
| PR1 | Ling3 解码层 + unscan 路径（`ling3.py` + `decoders.py` unscan + `multi_token_prediction.py`） | 前置依赖（KDA、MLA 门控、配置扩展）已合并 |
| PR2 | ScannableBlock + scan 路径（`ling3.py` ScannableBlock + `decoders.py` scan） | PR1 |
| PR3 | 测试（`ling3_decoder_test.py`） | PR2 |

---

## 开放问题

1. **Scan 中 KDA 全局层索引**：KDA 是否需要全局层索引取决于其具体实现。GLA 使用 ALiBi 风格的逐层 slope 衰减（`slope_scale = 1 - layer_idx / (num_layers - 1)`），依赖全局索引；而同源的 Gated Delta Net（Qwen3-Next）采用可学习衰减参数，不依赖层索引。待 KDA 实现确定后明确。即使需要，也有成熟方案：在 scan 调用时传入 `block_indices = jnp.arange(scan_length)` 作为 `in_axes=0` 的非广播输入，ScannableBlock 内部即可通过 `unscan_prefix + block_idx * interval + intra_idx` 计算全局索引，无需修改 carry 结构。
2. **Checkpoint 转换**：Ling3 的 HF checkpoint 到 MaxText checkpoint 的转换需要单独 RFC（涉及 `mla_g_proj` 权重映射和 KDA 层权重映射）。
3. **Unscan 前缀层数量**：scan 路径中 unscan 前缀包含 dense 层和 MoE 过渡层（Ling3-tiny 共 4 层、Ling3-flash 共 6 层），对编译时间和运行效率的影响需 benchmark 验证。
