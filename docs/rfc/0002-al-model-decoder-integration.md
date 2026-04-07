---
title: "RFC-0002: Ling2 Decoder 集成"
status: implemented (Phase 1)
author: ClawSeven
date: 2026-03-31
reviewers:
  - jimoosciuc
---

# RFC-0002: Ling2 Decoder 集成

> 关联 Issue: [primatrix/ant-pretrain#234](https://github.com/primatrix/ant-pretrain/issues/234)
>
> 上游重构总体设计: [upstream-refactoring RFC](https://github.com/primatrix/ant-pretrain/blob/main/docs/rfcs/2026-03-23-upstream-refactoring.md) — PR6
>
> 实现 PR: [primatrix/maxtext#25](https://github.com/primatrix/maxtext/pull/25)

## 概述

将 Ling2 Decoder（混合 MLA/GLA + MoE 架构）集成到上游 MaxText。重构后模型统一命名为 **Ling2**（原 `ALModel/` 下的实现迁移重命名，原 `ling2.py` 废弃实现删除）。分两阶段实施：Phase 1 完成代码重构与 unscanned 模式验证；Phase 2 实现 `nn.scan` 支持。

## 背景

Ling2 是一种混合注意力解码器架构，每 `inhomogeneous_layer_cycle_interval`（默认 5）层中：前 4 层使用 GLA（Gated Linear Attention / Lightning Attention-2）线性注意力，最后 1 层使用 MLA（Multi-Head Latent Attention）全注意力。MLP 部分采用首层 Dense + 其余 MoE 的布局。

当前存在以下问题：

1. **命名与目录不规范**：实际实现代码位于项目根目录 `ALModel/`，使用 `al_model` 命名，不在标准的 `src/MaxText/layers/` 下
2. **旧实现冗余**：`src/MaxText/layers/ling2.py` 是早期实验实现，功能与 `ALModel/` 重叠且不再维护
3. **仅支持 unscanned 模式**：当前使用 `for_loop` 逐层展开，XLA 编译图规模大（~20 层），HBM 占用高
4. **Argus dump 耦合**：`@_dumpable` / `_dump_tensor` 调用散布在代码中，需剥离到独立目录管理
5. **配置字段冗余**：`ALModel/` 使用 `layer_group_size`、`moe_layer_freq` 等自定义字段，与上游已有的 `inhomogeneous_layer_cycle_interval`、`first_num_dense_layers` 语义重复

**前置依赖**（假设已合入）：
- PR2: MoE 对齐（z-loss、router bias、5-value 返回签名、`moe_shared_expert_dim` 支持）
- PR4: GLA 线性注意力（`attention_gla.py`、`gla_pallas.py`、`GroupRMSNorm`）
- PR5: MLA 注意力（已在上游，含 `mla_interleaved_rope`）
- PR1: 配置扩展（已在上游，所有新增字段已存在）

## 方案

### 配置复用策略

Ling2 所需语义由上游现有字段覆盖，**不新增配置字段**：

| 现有字段 | Ling2 用法 | 说明 |
|---------|-----------|------|
| `inhomogeneous_layer_cycle_interval` | 设为 `5`（每组 4 层 GLA + 1 层 MLA） | 已被 Llama4、Qwen3-Next、GPT-OSS 使用，语义完全一致 |
| `first_num_dense_layers` | 设为 `1`（第一层为 dense，其余为 MoE） | 已在 DeepSeek 中使用，配合 `interleave_moe_layer_step=1` 表达 dense+MoE 排布 |

**不引入的字段：**

- `layer_group_size` — 与 `inhomogeneous_layer_cycle_interval` 等价，checkpoint 转换中做映射即可
- `moe_layer_freq` — Ling2 实际模式为"第一层 dense，其余全 MoE"，用 `first_num_dense_layers=1` + `interleave_moe_layer_step=1` 即可表达，引入第三种机制会导致优先级混乱

### 命名变更

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `ALModel/al_model.py` | `src/maxtext/models/ling2.py` | 目录迁移 + 重命名 |
| `DecoderBlockType.AL_MODEL` | `DecoderBlockType.LING2` | 复用已有枚举值 |
| `ALGenericLayer` / `ALDenseLayer` / `ALMoELayer` | `Ling2DecoderLayer` | 统一为单一类 |
| `HybridAttention` | 内联到 `Ling2DecoderLayer.__init__` | 根据 `is_mla()` 直接选择 MLA 或 GLA |
| `ALScannableBlock`（Phase 2） | `Ling2ScannableBlock` | 重命名 |
| `al_model.yml` | `ling2.yml` | 配置文件重命名 |

`common_types.py` 中保留 `LING2 = "ling2"`，移除 `AL_MODEL = "al_model"`。

---

## Phase 1: 代码迁移（unscanned 模式）— 已实现

**目标**：完成代码迁移、命名统一，保证 unscanned 模式下功能正确。scan 模式暂时抛出 `NotImplementedError`。

### 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/maxtext/models/ling2.py` | **新增** | 从 ant-pretrain 迁移，适配导入路径，仅含 Phase 1 类 |
| `src/maxtext/layers/decoders.py` | **修改** | 添加 import、get_decoder_layers 分发、get_norm_layer、unscanned 分支 |
| `src/maxtext/layers/linears.py` | **修改** | MlpBlock.get_norm_layer 添加 LING2 |
| `src/maxtext/configs/models/ling2.yml` | **修改** | 移除 [NEW] 标记，确认字段值 |
| `tests/unit/ling2_decoder_test.py` | **新增** | 单元测试 |
| `tests/unit/ling2_e2e_verify.py` | **新增** | TPU 端到端验证脚本 |

### Step 1.1: 创建 `src/maxtext/models/ling2.py`

**来源**: 从 `ant-pretrain/src/MaxText/layers/ling2.py`（407 行）迁移

**关键变更**:

1. **导入路径适配**（`MaxText.*` → `maxtext.*`）:
   ```python
   # 旧: from MaxText.common_types import Config, ...
   # 新: from maxtext.common.common_types import Config, ...
   # 旧: from MaxText.layers import attention_gla, attention_mla, linears, moe, ...
   # 新: from maxtext.layers import attention_gla, attention_mla, linears, moe, ...
   # 旧: from MaxText import max_utils
   # 新: from maxtext.utils import max_utils
   # 旧: from MaxText.sharding import create_sharding, maybe_shard_with_logical
   # 新: from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical
   ```

2. **仅保留 `Ling2DecoderLayer` 和 `Ling2DecoderLayerToLinen`**，不迁移 `Ling2ScannableBlock`（Phase 2）

3. **统一为 `Ling2DecoderLayer`**:
   ```python
   class Ling2DecoderLayer(nnx.Module):
       def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
           # 根据 layer_idx 选择注意力类型
           is_full_attention = (layer_idx + 1) % config.inhomogeneous_layer_cycle_interval == 0
           if is_full_attention:
               self.attention = attention_mla.MLA(...)
           else:
               self.attention = attention_gla.BailingMoeV2LinearAttention(...)

           # 根据 layer_idx 选择 MLP 类型
           if layer_idx < config.first_num_dense_layers:
               self.mlp = linears.MlpBlock(...)
           else:
               self.mlp = moe.RoutedAndSharedMoE(...)

       def __call__(self, inputs, ..., global_layer_idx=None):
           # global_layer_idx 用于 scan 模式下传递正确的运行时层索引
           # Phase 1 中由 decoders.py 传入 lyr
           ...
   ```

4. **Linen 包装**:
   ```python
   Ling2DecoderLayerToLinen = nnx_wrappers.to_linen_class(
       Ling2DecoderLayer,
       base_metadata_fn=initializers.variable_to_logically_partitioned,
   )
   ```

### Step 1.2: 修改 `decoders.py`

#### 添加 import

```python
from maxtext.models import (
    ...
    ling2,      # 新增
    ...
)
```

#### `get_decoder_layers()` 添加 LING2 分发

```python
case DecoderBlockType.LING2:
    if self.config.scan_layers:
        raise NotImplementedError(
            "Ling2 decoder does not support scan_layers=True yet. "
            "Please set scan_layers=False."
        )
    return [ling2.Ling2DecoderLayerToLinen]
```

#### `get_norm_layer()` 添加 LING2

将 `DecoderBlockType.LING2` 加入使用 RMSNorm 的类型列表。

#### unscanned 分支添加 LING2 处理

参考 QWEN3_NEXT 模式，在 unscanned 循环中添加：

```python
# 构造参数 + 调用参数
if cfg.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.LING2):
    layer_kwargs = {"layer_idx": lyr}
if cfg.decoder_block == DecoderBlockType.LING2:
    layer_call_kwargs = {"global_layer_idx": lyr}

# KV cache 读取：仅 MLA 层有 cache
elif kv_caches is not None and cfg.decoder_block == DecoderBlockType.LING2:
    if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
        kv_cache = kv_caches[lyr]
```

KV cache 写回使用默认 `if returned_cache is not None` 检查，GLA 层返回 `None` 自动跳过。

### Step 1.3: 修改 `linears.py`

`MlpBlock.get_norm_layer()` 中将 `DecoderBlockType.LING2` 加入 RMSNorm 类型列表。

### Step 1.4: 更新 `ling2.yml`

```yaml
decoder_block: "ling2"
inhomogeneous_layer_cycle_interval: 5  # 4 GLA + 1 MLA per group
first_num_dense_layers: 1
scan_layers: false                     # Phase 1 不支持 scan
# interleave_moe_layer_step 在 base.yml 中默认为 1，无需额外设置
```

移除所有 `[NEW]` 注释标记（PR1 已合入，字段已是正式字段）。

### ant-pretrain vs maxtext unscan mode 差异分析

| 差异 | 分析 | 处理 |
|------|------|------|
| `global_layer_idx` 传递 | ant-pretrain 通过 `layer_call_kwargs` 传递；maxtext 最初遗漏 | 已修复：添加 `layer_call_kwargs = {"global_layer_idx": lyr}` |
| KV cache 读取 | ant-pretrain 对所有层读 `kv_caches[lyr]`；maxtext 仅对 MLA 层读 | 功能等价：GLA 层 `__call__` 内部直接 `kv_cache = None`。maxtext 写法更清晰 |
| KV cache 写回 | 两边都用 `returned_cache is not None` 过滤 | 完全等价 |
| `hidden_state` 归一化时机 | ant-pretrain 返回归一化后；maxtext 返回 raw，norm 在 `apply_output_head` 内 | maxtext 架构差异，非 Ling2 特有。MTP 场景需注意 |

### Phase 1 验证结果

在 `multislice-job-zehuan-9xg4r` (8x TPU v4) 上通过全部 6 项测试：

| 测试 | 状态 | 说明 |
|------|------|------|
| Config loading | PASSED | LING2 config 正确加载 |
| Layer construction | PASSED | layer_idx=0: GLA+Dense; idx=1: GLA+MoE; idx=4: MLA+MoE; idx=9: MLA+MoE |
| Single-layer forward | PASSED | GLA 层 output shape=(8,256,128), kv_cache=None; MLA 层 output 正确 |
| scan_layers error | PASSED | `NotImplementedError` 正确抛出 |
| get_decoder_layers | PASSED | 返回 `[Ling2DecoderLayerToLinen]` |
| Full Decoder forward | PASSED | 10 层完整前向传播，logits shape=(8,256,157184)，所有值 finite |

---

## Phase 2: Scan 模式支持

**目标**：实现 `Ling2ScannableBlock`，支持 `nn.scan`，降低 XLA 编译图规模和 HBM 占用。移除 Phase 1 中的 `NotImplementedError`。

**前置条件**：Phase 1 已合入并验证。

### 涉及文件

| 文件 | 说明 |
|------|------|
| `src/maxtext/models/ling2.py` | 新增 `Ling2ScannableBlock` |
| `src/maxtext/layers/decoders.py` | 移除 `NotImplementedError`，添加 `_apply_ling2_scanned_blocks()` |
| `src/maxtext/configs/models/ling2.yml` | 将 `scan_layers` 默认改为 `true` |

### Step 2.1: 实现 `Ling2ScannableBlock`

将 `inhomogeneous_layer_cycle_interval` 个异构子层打包为一个可扫描单元：

```python
class Ling2ScannableBlock(nnx.Module):
    """可扫描的 Ling2 decoder block。

    层索引设计：
    - 构造时 layer_idx = block_base_idx + i → 决定层结构（MLA vs GLA, Dense vs MoE）
    - 运行时 global_layer_idx = block_base_idx + block_idx * cycle_interval + i
      → 传递给 GLA 的 slope scaling 等需要真实全局层索引的计算
    """

    def __init__(self, config, mesh, model_mode, quant=None, *, rngs, block_base_idx=0):
        self.config = config
        self.block_base_idx = block_base_idx
        cycle = config.inhomogeneous_layer_cycle_interval
        for i in range(cycle):
            layer = Ling2DecoderLayer(
                config=config, mesh=mesh, model_mode=model_mode,
                layer_idx=block_base_idx + i, quant=quant, rngs=rngs.fork(),
            )
            setattr(self, f"layer_{i}", layer)

    def __call__(self, carry, block_idx, decoder_segment_ids, ...):
        x = carry
        cycle = self.config.inhomogeneous_layer_cycle_interval
        block_idx = jnp.asarray(block_idx, dtype=jnp.int32)
        for i in range(cycle):
            layer = getattr(self, f"layer_{i}")
            global_layer_idx = self.block_base_idx + block_idx * cycle + i
            x, _ = layer(x, ..., global_layer_idx=global_layer_idx)
        return x, None
```

生成 Linen 包装：`Ling2ScannableBlockToLinen = nnx_wrappers.to_linen_class(Ling2ScannableBlock, ...)`

关键设计决策：

- `inhomogeneous_layer_cycle_interval`（默认 5）= 4 GLA + 1 MLA 构成一个 block
- `nn.scan` 要求同构参数结构，block 内子层各自独立命名（`layer_0` ~ `layer_4`）
- `block_idx` 由 scan 传入，用于计算运行时 `global_layer_idx`
- 优势：~20 层等价降至 ~4 个 scan 迭代，降低 XLA 编译图规模和 HBM 占用

### Step 2.2: 修改 `decoders.py` — 集成 scan 模式

#### 移除 `NotImplementedError`，更新 `get_decoder_layers()`

```python
case DecoderBlockType.LING2:
    if self.config.scan_layers:
        return [ling2.Ling2ScannableBlockToLinen]
    else:
        return [ling2.Ling2DecoderLayerToLinen]
```

#### 添加 `_apply_ling2_scanned_blocks()` 方法

三阶段执行：

| 阶段 | 说明 | 模式 |
|------|------|------|
| 1. Dense prefix | 前 `first_num_dense_layers` 层 | unscanned 逐层执行 |
| 2. MoE blocks | `remaining // cycle_interval` 个 block | `nn.scan` 扫描 |
| 3. MoE remainder | `remaining % cycle_interval` 个尾部层 | unscanned 逐层执行 |

#### 在 scan 分支添加分发

```python
elif cfg.decoder_block == DecoderBlockType.LING2:
    y = self._apply_ling2_scanned_blocks(
        y, decoder_segment_ids, decoder_positions,
        deterministic, model_mode, previous_chunk, page_state, slot,
    )
```

### Step 2.3: 更新配置

`ling2.yml` 中将 `scan_layers: false` 改为 `scan_layers: true`。

### Phase 2 测试计划

1. **`Ling2ScannableBlock` 测试**：单 block 前向传播、`global_layer_idx` 正确性
2. **Scan vs Unscanned 数值一致性**：相同输入/初始化，对比 `scan_layers=True` 和 `scan_layers=False` 的输出
3. **端到端训练测试**：`scan_layers=True` 运行少量步数，验证 loss 收敛
4. **性能验证**：对比 scan vs unscanned 的 XLA 编译时间和 HBM 占用

---

## 备选方案

1. **保留 `ALDenseLayer` + `ALMoELayer` 双类结构** — 不做合并，仅增加 scan 支持。未采用原因：`decoders.py` 中需保留双层循环特殊逻辑，增加维护负担。
2. **一次性完成 refactor + scan** — 不分阶段。未采用原因：变更量过大，review 难度高，出问题时难以定位是重构引入还是 scan 逻辑引入。
3. **新增 `layer_group_size` 配置字段** — 未采用原因：与上游已有的 `inhomogeneous_layer_cycle_interval` 语义完全一致，引入冗余字段。

## 影响范围

| 方面 | 影响 |
|------|------|
| 现有上游用户 | 无影响 — `LING2` decoder block type 为新增功能 |
| ant-pretrain 用户 | Phase 1 后 `scan_layers=false` 可用；Phase 2 后 `scan_layers=true` 可用 |
| 参数名/checkpoint | `Ling2DecoderLayer` 统一后参数名变化，需在 PR8（检查点转换）中更新映射；scan/unscanned checkpoint 不可直接互换，需通过转换路径处理 |
| API 签名 | `__call__` 新增可选参数 `global_layer_idx`，默认 None，不影响现有调用 |

## 风险

| 风险 | 应对措施 |
|------|---------|
| `Ling2DecoderLayer` 合并后参数名变化导致现有 checkpoint 不兼容 | PR8（检查点转换）中更新参数映射；checkpoint 转换代码从 HF config 读取 `layer_group_size` 后映射到 `inhomogeneous_layer_cycle_interval` |
| Phase 2 scan 模式下 `global_layer_idx` 计算错误导致 GLA slope 异常 | 专项测试覆盖 `global_layer_idx` 在三阶段的正确性；Phase 1 先验证 unscanned 模式正确 |
| 上游 rebase 冲突（`decoders.py` 变更较多） | 分阶段减小单次变更量；Phase 1 优先合入 |
