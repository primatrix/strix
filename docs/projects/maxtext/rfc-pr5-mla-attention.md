---
title: "RFC-PR5: MLA Interleaved RoPE Alignment"
status: draft
author: fdz-1999
date: 2026-03-31
reviewers: []
---

# RFC-PR5: MLA Interleaved RoPE Alignment

**Depends on**: PR1 (Config Extensions — `mla_interleaved_rope` 字段)
**代码分支**: [`feature/mla-attention`](https://github.com/primatrix/maxtext/tree/feature/mla-attention)

---

## 1. 动机与目标

### 1.1 问题

Megatron-LM 和 MaxText 对 RoPE (Rotary Position Embedding) 维度的排列采用不同约定：

| 框架 | 排列方式 | 格式 |
|------|---------|------|
| Megatron-LM / HuggingFace | interleaved | `[x₀, y₀, x₁, y₁, ...]` |
| MaxText / JAX | non-interleaved | `[x₀, x₁, ..., y₀, y₁, ...]` |

两者的旋转语义相同（`(xᵢ, yᵢ)` 对应频率 `θᵢ`），但 tensor 中元素的物理排列不同。

当使用 Megatron-LM 训练的 MLA 模型权重（或 HuggingFace DeepSeek-V2/V3 权重）在 MaxText 中推理或继续训练时，Q/K 的 rope 维度仍保持 interleaved 排列。MaxText 的 `RotaryEmbedding` 假设输入为 non-interleaved 格式，**直接使用会导致 RoPE 旋转方向错误，attention score 与 Megatron 不一致**。

### 1.2 为什么需要运行时处理

标准 MHA/GQA 的 interleave 差异可以在权重转换阶段一次性重排解决。但 MLA 有其特殊性：

- `key_rope` 从 `wkv_a` 的投影输出中**动态分离**（每次 forward pass 都重新计算）
- `query_rope` 同样是动态计算的
- 不同 RoPE 实现（标准 RotaryEmbedding vs YaRN）对 interleave 的内部处理不同

因此 de-interleave 必须在 `apply_rotary_embedding` 中**运行时执行**，无法在权重转换时静态处理。

### 1.3 目标

在 MaxText 的 `Attention.apply_rotary_embedding()` 中为 MLA attention type 添加 de-interleave 逻辑，保证 **相同权重 + 相同输入 → 相同 attention output**。

---

## 2. 设计方案

### 2.1 核心变更

在 `Attention.apply_rotary_embedding()` 中，通过 `AttentionType.MLA` 守卫新增 MLA 分支：

```python
elif self.config.attention_type == AttentionType.MLA.value:
    if self.config.mla_interleaved_rope and not (
        isinstance(self.rotary_embedding, YarnRotaryEmbedding)
        and self.rotary_embedding.interleave
    ):
        # de-interleave: [x0, y0, x1, y1, ...] → [x0, x1, ..., y0, y1, ...]
        inputs = jnp.concatenate([inputs[..., 0::2], inputs[..., 1::2]], axis=-1)
    return self.rotary_embedding(inputs, inputs_positions)
```

当 `mla_interleaved_rope=True` 且 RoPE 实现未内部处理 interleave 时，在 RoPE 之前执行 de-interleave。

### 2.2 条件分支

| `mla_interleaved_rope` | RoPE 实现 | 行为 |
|:-:|------|------|
| `True` | 标准 RotaryEmbedding | de-interleave → RoPE |
| `True` | YaRN (`interleave=True`) | 跳过（YaRN 内部已处理） |
| `True` | YaRN (`interleave=False`) | de-interleave → RoPE |
| `False` | 任意 | 直接 RoPE（权重已是 non-interleaved） |

非 MLA attention type 不进入此分支（`AttentionType.MLA` 守卫）。

### 2.3 备选方案

| 方案 | 不采用的原因 |
|------|------------|
| 在权重转换时做维度重排 | MLA 的 key_rope/query_rope 是动态计算的，无法静态转换 |
| 修改 `RotaryEmbedding` 内部实现 | 会影响所有 attention 类型（MHA/GQA），侵入性过大 |

---

## 3. 涉及文件

| 文件 | 变更 | 说明 |
|------|:----:|------|
| `src/maxtext/layers/attentions.py` | +13/-1 | MLA interleaved RoPE de-interleave 分支 |
| `src/maxtext/configs/types.py` | +8 | `mla_interleaved_rope` 配置字段 |
| `src/maxtext/configs/base.yml` | +1 | `mla_interleaved_rope: True` 默认值 |
| `tests/unit/test_mla_interleaved_rope.py` | +209 (新增) | 6 个测试场景覆盖全部分支 |

**总计**：4 files, +231/-1 lines

注：`src/maxtext/layers/attention_mla.py`（766 行）为上游 fork 点已有文件，ant-pretrain 零修改，本 PR 不涉及。

---

## 4. Argus Dump 剥离清单

**本 PR 涉及的所有文件均无 Argus dump 调用**，无需剥离。

---

## 5. 向后兼容性

| 方面 | 影响 |
|------|------|
| MHA / GQA 用户 | 零影响 — `AttentionType.MLA` 守卫隔离 |
| `mla_interleaved_rope=True`（默认） | 仅对 MLA attention type 生效，匹配 Megatron 权重约定 |
| 现有 DeepSeek 模型 | `deepseek.py` 中 MLA 调用经过此分支，默认值与 Megatron 一致 |

---

## 6. 测试计划

### 6.1 单元测试

`tests/unit/test_mla_interleaved_rope.py`（6 个场景，覆盖全部分支）：

| 用例 | 覆盖场景 |
|------|---------|
| `test_yarn_interleave_true_skips_external_deinterleave` | YaRN interleave=True → 跳过外部 de-interleave |
| `test_mla_interleaved_rope_false_no_deinterleave` | 配置关闭 → 不做 de-interleave |
| `test_non_yarn_mla_interleaved_deinterleaves` | 标准 RoPE + True → 执行 de-interleave |
| `test_partial_rotary_non_yarn_deinterleaves_rot_portion` | partial_rotary → 仅对 rotary 部分 de-interleave |
| `test_yarn_interleave_false_deinterleaves` | YaRN interleave=False → 执行外部 de-interleave |
| `test_non_mla_no_deinterleave` | 非 MLA type → 不进入 MLA 分支 |

### 6.2 Argus 跨框架数值对比

使用 `tests/unit/mla_compare_test.py` 对 Megatron dump 数据和 MaxText 推理结果进行逐层对比：

| 对比项 | 指标 | 容差 |
|--------|------|------|
| MLA forward output（逐层） | cosine similarity | > 0.9999 |
| MLA backward output（逐层） | cosine similarity | > 0.9999 |
| 全精度 ULP 距离 | P99 ULP (bfloat16) | < 100 |

CI 实际结果（layer 4, Ling-2.5 权重, 4-chip TPU v7）：

| 方向 | cosine | rel_l2 | ulp_p99 |
|------|--------|--------|---------|
| forward | 0.999993 | 3.7e-03 | 6 |
| backward | 0.999936 | 1.1e-02 | 12 |

---

## 7. 风险

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| 修改 `apply_rotary_embedding` 影响其他 attention 类型 | 高 | 极低 | `AttentionType.MLA` 守卫 + 6 个单元测试全分支覆盖 |
| 上游 `attentions.py` rebase 冲突 | 中 | 低 | 变更仅 13 行，冲突可手动解决 |
| 现有 DeepSeek 权重兼容性 | 中 | 低 | 默认 `mla_interleaved_rope=True` 匹配 Megatron/HuggingFace 约定 |

---

## 8. References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — MLA 原始论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 维度排列约定
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — YaRN interleave 模式
- Megatron-LM `megatron/core/models/deepseek/deepseek_attention.py` — MLA RoPE 参考实现
