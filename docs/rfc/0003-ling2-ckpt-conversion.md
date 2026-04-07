# RFC: Refactor PR8 - Checkpoint Conversion Extension for Ling2

| 字段     | 值                              |
| -------- | ------------------------------- |
| **作者** | @aftersnow                      |
| **日期** | 2026-03-31                      |
| **状态** | Draft                           |
| **PR**   | PR8（依赖 PR6: Ling2 Decoder） |

## 动机

为了将 HF 预训练权重导入 MaxText 继续训练，需要在检查点转换框架中注册 Ling2 的参数映射。

MaxText 已有一套数据驱动的转换框架（`param_mapping.py` 中的 `PARAM_MAPPING` + `HOOK_FNS` 字典），本 PR 只需为 Ling2 新增一对 mapping/hook 函数并注册即可，不改动框架本身。

## 背景：Ling2 的参数结构特殊性

Ling2 与已支持的模型（Gemma/Qwen/LLaMA/DeepSeek 等）的关键差异在于三点，这决定了它无法复用已有映射：

1. **异构层**：前 `first_num_dense_layers` 层是 Dense MLP，其余是 MoE。两者在 MaxText 中分别编号（`dense_layers_0`, `moe_layers_0`），但 HF 侧用全局连续编号（`model.layers.0`, `model.layers.1`, ...）
2. **混合注意力**：每 `inhomogeneous_layer_cycle_interval` 层中，前几层用 GLA（Gated Linear Attention），最后一层用 MLA（Multi-Head Latent Attention）。两种注意力的参数集完全不同
3. **可选 MTP 层**：Multi-Token Prediction 层有独立的 norm、投影和内嵌 transformer + MoE，在其他模型中不存在

## 涉及文件

| 文件 | 变更 |
| --- | --- |
| `src/maxtext/utils/ckpt_conversion/utils/param_mapping.py` | 新增 `LING2_MAXTEXT_TO_HF_PARAM_MAPPING` 和 `LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN`，注册到 `PARAM_MAPPING` / `HOOK_FNS` 字典（约 295 行） |
| `src/maxtext/utils/ckpt_conversion/utils/utils.py` | 在 `HF_IDS` 字典中新增 `"ling2"` 条目（约 9 行） |
| `src/maxtext/utils/ckpt_conversion/to_maxtext.py` | **不变** — 框架通过 `PARAM_MAPPING[model_name]` 字典分发，自动支持新模型 |
| `src/maxtext/utils/ckpt_conversion/to_huggingface.py` | **不变** — 同上，通过 `HOOK_FNS[model_name]` 字典分发 |
| `src/maxtext/tools/ckpt_compare/compare_orbax_safetensors.py` | **新文件** — 从 ant-pretrain `scripts/` 迁入，shape 级对比脚本 |
| `src/maxtext/tools/ckpt_compare/compare_params.py` | **新文件** — 从 ant-pretrain `scripts/` 迁入，数值级对比脚本 |

## 详细设计

### 1. Mapping 函数：`LING2_MAXTEXT_TO_HF_PARAM_MAPPING`

**签名**（与其他模型一致）：

```python
def LING2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False) -> dict
```

**核心逻辑**：遍历所有层，根据层类型和注意力类型生成不同的 key 映射。

```text
for global_layer_idx in range(num_layers):
    ├── 确定 MaxText 前缀 (dense_layers_X 或 moe_layers_X)
    ├── 确定 HF 前缀 (model.layers.{global_idx})
    ├── 映射 LayerNorm 参数
    ├── if _is_mla(层):  映射 MLA 参数 (wq_a/wq_b/wkv_a/wkv_b/...)
    │   else:            映射 GLA 参数 (query_key_value/g_proj/...)
    └── if is_moe:       映射共享专家 + 路由器 + 路由专家
        else:            映射 Dense MLP (wi_0/wi_1/wo)
```

**注意力类型判断**：

```python
def _is_mla(global_layer_idx):
    # 每组最后一层 或 模型最后一层 → MLA，其余 → GLA
    interval = inhomogeneous_layer_cycle_interval
    return (global_layer_idx + 1) % interval == 0 or global_layer_idx == num_layers - 1
```

> **命名说明**：ant-pretrain 中使用 `layer_group_size`，上游复用已有的 `inhomogeneous_layer_cycle_interval`（PR1 已确认两者语义等价）。从 HF config 读取 `layer_group_size` 时映射到 `inhomogeneous_layer_cycle_interval`。

**MoE Expert 权重映射**：MaxText 将所有专家堆叠为一个 `[num_experts, ...]` 张量，HF 拆成独立文件：

```python
experts_wi_0 = [f"{hf_prefix}.mlp.experts.{e}.gate_proj.weight" for e in range(num_experts)]
mapping[f"{mt_prefix}-Ling2MoeBlock_0-MoeBlock_0-wi_0"] = experts_wi_0  # list value → expert stacking
```

> **命名说明**：ant-pretrain 中 MoE 子模块有多种历史命名（`ALMoeBlock_0`、`DeepSeekMoeBlock_0` 等），mapping 为每个参数注册多个候选 key 做兜底。推上游后 Ling2 decoder（PR6）统一使用 `Ling2MoeBlock_0`，不再需要多变体兜底。

### 2. MTP 层映射

当 `num_nextn_predict_layers > 0` 时，在 `model.layers.{num_layers}` 位置映射 MTP 特有参数。MTP 层包含：

- `embedding_norm` / `hidden_state_norm` / `final_layernorm` — 三个独立 RMSNorm
- `projection_layer` — 将 embedding 和 hidden state 拼接后投影
- 内嵌 transformer 层 — 包含完整的 MLA 注意力 + MoE MLP

```python
if has_mtp:
    mtp_hf_prefix = f"model.layers.{num_layers}"
    mapping["params-mtp_block-mtp_layer_1-embedding_norm-scale"] = f"{mtp_hf_prefix}.enorm.weight"
    mapping["params-mtp_block-mtp_layer_1-projection_layer-kernel"] = f"{mtp_hf_prefix}.eh_proj.weight"
    # ... 内嵌 transformer 的 MLA + MoE 参数（结构与主 decoder 层一致）
```

### 3. Hook 函数：`LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN`

Ling2 的 hook 非常简单 — 所有 kernel 只需一个 `reshape_kernel`（transpose + reshape），无 query scale、RoPE 重排等额外操作。这是因为 Ling2 的 MaxText 和 HF 实现对 Q/K scale、RoPE 布局的约定一致。

```python
def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
        return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)
```

结构与 mapping 函数平行：按层类型和注意力类型为所有 kernel 参数注册 `reshape_kernel`。

### 4. 注册

```python
PARAM_MAPPING["ling2"] = LING2_MAXTEXT_TO_HF_PARAM_MAPPING
HOOK_FNS["ling2"] = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN
HF_IDS["ling2"] = "local/ling2"  # 非公开模型，需通过 --hf_model_path 指定本地路径
```

### 5. 限制

- 当前仅支持 `scan_layers=False`（unscanned 模式）
- `scan_layers=True` 暂不支持，原因是 Ling2 的异构 scan 需要三段式处理（dense prefix unscanned → MoE blocks scanned → MoE remainder unscanned），与现有框架的单一 scan axis 假设不兼容，需要扩展框架逻辑，留作后续 PR

## Mapping 示例

以 20 层 decoder（1 dense + 19 MoE）+ 1 MTP、`inhomogeneous_layer_cycle_interval=5`、256 专家的 Ling2 为例：

| MaxText Key | HF Key | 说明 |
| --- | --- | --- |
| `params-token_embedder-embedding` | `model.word_embeddings.weight` | 词嵌入（Ling2 HF 模型定义使用 `word_embeddings` 而非 `embed_tokens`） |
| `params-decoder-dense_layers_0-mlp-wi_0-kernel` | `model.layers.0.mlp.gate_proj.weight` | Dense 层 MLP |
| `params-decoder-moe_layers_3-self_attention-wkv_a-kernel` | `model.layers.4.attention.kv_a_proj_with_mqa.weight` | MLA 层 KV 低秩投影（moe_layers_3 → global index 4） |
| `params-decoder-moe_layers_0-self_attention-query_key_value-kernel` | `model.layers.1.attention.query_key_value.weight` | GLA 层 QKV 融合（moe_layers_0 → global index 1） |
| `params-decoder-moe_layers_5-Ling2MoeBlock_0-MoeBlock_0-wi_0` | `model.layers.6.mlp.experts.{0..255}.gate_proj.weight` | MoE 路由专家（expert stacking，1 个 MaxText 张量 → 256 个 HF 张量） |
| `params-mtp_block-mtp_layer_1-projection_layer-kernel` | `model.layers.20.eh_proj.weight` | MTP 投影层（HF 将 MTP 放在 model.layers 末尾） |

## 对比脚本

以下两个验证脚本从 ant-pretrain 的 `scripts/` 目录迁入上游 `src/maxtext/tools/ckpt_compare/`：

| 脚本 | 功能 |
| --- | --- |
| `compare_orbax_safetensors.py` | Shape 级对比：检查 Orbax 和 HF safetensors 的 key 名和张量形状是否正确对应 |
| `compare_params.py` | 数值级对比：逐元素比对两侧的实际权重值，输出 max_diff / cosine similarity 等指标 |

这两个脚本独立于转换框架，不影响 `param_mapping.py` 的逻辑。

> **关于映射逻辑重复**：ant-pretrain 中 `compare_orbax_safetensors.py` 内有独立的硬编码映射函数 `mt_to_hf_keys()`，与 `param_mapping.py` 存在逻辑重复。推上游后应重构为复用 `param_mapping.py` 的映射，消除重复。

## 向后兼容性

| 保证 | 机制 |
| --- | --- |
| 不影响已有模型转换 | 仅在 `PARAM_MAPPING` / `HOOK_FNS` 字典中追加新条目，不修改已有模型的映射 |
| `to_maxtext.py` / `to_huggingface.py` 无行为变更 | 框架通过字典分发，新增注册项不改变分发逻辑 |
| 对比脚本独立于转换框架 | 作为独立工具放在 `tools/ckpt_compare/` 下，不影响任何现有代码路径 |

## 测试计划

### 1. 单元测试（CPU，无需 checkpoint 文件）

新增测试文件 `tests/ling2_ckpt_conversion_test.py`，标记 `@pytest.mark.cpu_only`，可在 CI 中快速运行。

#### 1.1 注册检查

验证 Ling2 已正确注册到转换框架的三个字典中：

```python
def test_ling2_registered():
    from maxtext.utils.ckpt_conversion.utils.param_mapping import PARAM_MAPPING, HOOK_FNS
    from maxtext.utils.ckpt_conversion.utils.utils import HF_IDS
    assert "ling2" in PARAM_MAPPING
    assert "ling2" in HOOK_FNS
    assert "ling2" in HF_IDS
```

#### 1.2 MLA/GLA 层分类

验证 `_is_mla()` 在不同 `inhomogeneous_layer_cycle_interval` 下对每个 layer index 的判断正确：

```python
@pytest.mark.parametrize("interval,num_layers,expected_mla_indices", [
    (5, 21, {4, 9, 14, 19, 20}),   # 标准 Ling2：每组第 5 层 + 最后一层
    (5, 20, {4, 9, 14, 19}),        # 恰好整除
    (1, 10, set(range(10))),        # interval=1：全部 MLA
])
def test_is_mla_classification(interval, num_layers, expected_mla_indices):
    for idx in range(num_layers):
        is_mla = (idx + 1) % interval == 0 or idx == num_layers - 1
        assert is_mla == (idx in expected_mla_indices), f"layer {idx}"
```

#### 1.3 层索引算术（global → dense/moe 前缀）

验证 `first_num_dense_layers=1` 时 global layer index 到 MaxText 前缀的映射：

```python
def test_layer_index_prefix_mapping():
    first_dense = 1
    num_layers = 21
    for global_idx in range(num_layers):
        if global_idx < first_dense:
            expected_prefix = f"dense_layers_{global_idx}"
        else:
            expected_prefix = f"moe_layers_{global_idx - first_dense}"
        # 调用 mapping 函数后，验证生成的 key 包含 expected_prefix
```

#### 1.4 Mapping key 完整性

给定 Ling2 默认配置，生成完整 mapping，验证 key 数量和模式：

```python
def test_mapping_key_completeness():
    mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config)
    # 基本数量检查
    assert len(mapping) > 0

    # 必须包含顶层参数
    top_level_keys = ["params-token_embedder-embedding", "params-decoder-decoder_norm-scale"]
    for key in top_level_keys:
        assert key in mapping or any(key in k for k in mapping)

    # Dense 层 key 存在
    assert any("dense_layers_0" in k for k in mapping)
    # MoE 层 key 存在
    assert any("moe_layers_0" in k for k in mapping)
    # MLA 层 key（wkv_a）存在
    assert any("wkv_a" in k for k in mapping)
    # GLA 层 key（query_key_value）存在
    assert any("query_key_value" in k for k in mapping)
    # MTP key 存在（当 mtp_num_layers > 0）
    assert any("mtp_block" in k for k in mapping)
```

#### 1.5 Hook 函数 shape 验证

验证 `reshape_kernel` 对小张量的输入输出 shape 正确：

```python
def test_reshape_kernel_roundtrip():
    import numpy as np
    # 模拟一个 [out_features, in_features] 的 MaxText kernel
    maxtext_shape = (128, 256)
    hf_shape = (256, 128)  # HF 约定：[in_features, out_features]
    tensor = np.random.randn(*maxtext_shape).astype(np.float32)

    # MaxText → HF（reshape by flipped target_shape, then transpose）
    hf_tensor = tensor.reshape(np.flip(np.array(hf_shape))).T
    assert hf_tensor.shape == (256, 128)

    # HF → MaxText（逆操作）
    restored = hf_tensor.T.reshape(maxtext_shape)
    np.testing.assert_array_equal(restored, tensor)
```

#### 1.6 Expert stacking mapping 结构

验证 MoE expert 映射值为正确长度的列表：

```python
def test_expert_stacking_mapping():
    num_experts = 256
    mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config)
    # 找到任一 expert stacking key
    expert_keys = [k for k in mapping if "MoeBlock_0-wi_0" in k]
    assert len(expert_keys) > 0
    for k in expert_keys:
        val = mapping[k]
        assert isinstance(val, list), f"{k} should map to a list for expert stacking"
        assert len(val) == num_experts, f"{k} should have {num_experts} entries"
```

### 2. E2E 测试（需要 checkpoint 文件）

#### 2.1 回归测试：已有模型转换不受影响

```bash
# 对任一已有模型（如 Qwen3-4B）执行转换，确认不受新注册项影响
python3 src/maxtext/utils/ckpt_conversion/to_maxtext.py \
    src/maxtext/configs/base.yml model_name="qwen3-4b" \
    base_output_directory="/tmp/qwen3_test" \
    hf_access_token=$HF_TOKEN hardware=cpu skip_jax_distributed_system=True \
    scan_layers=False
```

#### 2.2 Shape 级对比：转换后 key 和 shape 100% 匹配

```bash
python3 src/maxtext/tools/ckpt_compare/compare_orbax_safetensors.py \
    --orbax-items-dir /path/to/ling2-orbax/0/items/ \
    --hf-dir /path/to/ling2-hf-safetensors/ \
    --first-num-dense-layers 1 \
    --num-experts 256 \
    --mtp-num-layers 1 \
    --report-json shape_report.json
# 预期: mapped_ok_count == orbax_leaf_count, missing_mapping_count == 0
```

#### 2.3 数值级对比：所有参数精度验证

```bash
python3 src/maxtext/tools/ckpt_compare/compare_params.py \
    --orbax-items-dir /path/to/ling2-orbax/0/items/ \
    --hf-dir /path/to/ling2-hf-safetensors/ \
    --first-num-dense-layers 1 \
    --num-experts 256 \
    --mtp-num-layers 1 \
    --report-json value_report.json
# 预期: 所有参数 max_diff < 1e-6 (bf16 精度范围), cosine > 0.999999
```

### 3. 覆盖范围

测试须覆盖以下所有层类型的映射正确性：

- [ ] Dense 层（layer 0）：Dense MLP + GLA 注意力
- [ ] MoE + GLA 层（layer 1-3）：MoE 共享专家 + 路由专家 + GLA 注意力
- [ ] MoE + MLA 层（layer 4）：MoE + MLA 注意力（含 Q/KV LoRA 分解）
- [ ] MTP 层：norm + projection + 内嵌 transformer
- [ ] 顶层参数：embedding / decoder_norm / logits_dense
