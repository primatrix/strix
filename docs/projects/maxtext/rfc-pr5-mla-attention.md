---
title: "RFC-PR5: MLA Interleaved RoPE Alignment"
status: draft
author: fdz-1999
date: 2026-03-31
reviewers: []
---

# RFC-PR5: MLA Interleaved RoPE Alignment

## 概述

在 MaxText 的 `apply_rotary_embedding` 中添加 MLA interleaved RoPE de-interleave 逻辑，对齐 Megatron-LM 的 RoPE 维度排列约定。

## 背景

Multi-Head Latent Attention（MLA）是 DeepSeek-V2 论文（arXiv:2405.04434）提出的注意力机制。上游 MaxText 在 fork 点 `d4ed2261` 已包含完整的 MLA 实现（`attention_mla.py`，766 行），ant-pretrain 对该文件 **零修改**。

### 问题：RoPE 维度排列不一致

Megatron-LM 和 MaxText 对 RoPE 维度有不同的排列约定：

```
Megatron-LM / HuggingFace (interleaved):
  rope_dims = [x0, y0, x1, y1, x2, y2, ...]
  旋转: (x0,y0) → θ_0,  (x1,y1) → θ_1, ...

MaxText / JAX (non-interleaved):
  rope_dims = [x0, x1, x2, ..., y0, y1, y2, ...]
  旋转: (x0,y0) → θ_0,  (x1,y1) → θ_1, ...
  但 x 和 y 在 tensor 中不相邻
```

使用 Megatron-LM 训练的权重转换到 MaxText 时，MLA 的 Q/K rope 部分维度是 interleaved 排列的。不做 de-interleave 会导致 RoPE 旋转方向错误，attention score 与 Megatron 不一致。

### 为什么不在权重转换阶段处理

MLA 的 `key_rope` 从 `wkv_a` 输出中**动态分离**，每次 forward 都需处理。Q 的 rope 部分也是动态计算的。因此需要在 `apply_rotary_embedding` 中**运行时**处理 de-interleave。

## 方案

### 实现位置

变更在 `Attention.apply_rotary_embedding()` 中，通过 `AttentionType.MLA` 守卫进入 MLA 分支：

```python
elif self.config.attention_type == AttentionType.MLA.value:
    if self.config.mla_interleaved_rope and not (
        isinstance(self.rotary_embedding, YarnRotaryEmbedding) and self.rotary_embedding.interleave
    ):
        inputs = jnp.concatenate([inputs[..., 0::2], inputs[..., 1::2]], axis=-1)
    return self.rotary_embedding(inputs, inputs_positions)
```

De-interleave 操作：`[x0, y0, x1, y1, ...] → [x0, x1, ..., y0, y1, ...]`

### 分支逻辑

| 条件 | 行为 | 原因 |
|------|------|------|
| `mla_interleaved_rope=True` + 非 YaRN | 外部 de-interleave 后 RoPE | 权重来自 Megatron，需维度重排 |
| `mla_interleaved_rope=True` + YaRN (`interleave=True`) | 跳过外部 de-interleave | YaRN 内部已处理 |
| `mla_interleaved_rope=False` | 不做变换 | 权重已是 non-interleaved 格式 |
| 非 MLA attention type | 不进入此分支 | `AttentionType.MLA` 守卫 |

### 备选方案

1. **在权重转换时处理** — 未采用，MLA 的 key_rope 是动态分离的，无法静态转换
2. **修改 RotaryEmbedding 内部** — 未采用，会影响所有 attention 类型

## 影响范围

| 方面 | 影响 |
|------|------|
| `attentions.py` 修改 | `AttentionType.MLA` 守卫确保仅 MLA 类型触发，不影响 MHA/GQA |
| `mla_interleaved_rope=True` 默认 | 仅影响 MLA attention type |
| 非 MLA 用户 | 零影响 |

### 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/maxtext/layers/attentions.py` | 修改 (+13/-1) | MLA interleaved RoPE de-interleave 分支 + 注释修复 |
| `src/maxtext/configs/types.py` | 修改 (+8) | `mla_interleaved_rope` 配置字段 |
| `src/maxtext/configs/base.yml` | 修改 (+1) | `mla_interleaved_rope` 默认值 |
| `tests/unit/test_mla_interleaved_rope.py` | 新增 (209 行) | 6 个测试场景覆盖所有分支 |

**总计**：4 files, +229/-1 lines

### 未修改文件

| 文件 | 说明 |
|------|------|
| `src/maxtext/layers/attention_mla.py` (766 行) | 上游 fork 点已有，ant-pretrain 零修改 |

## 实施计划

### 移植策略

原计划（参考 replan.md Section 7.5）：
1. 基于 fork 基点 `d4ed2261` 创建分支
2. 从 ant-pretrain 检出 MLA 文件
3. `git rebase origin/main`（MLA 是新文件，大概率零冲突）

**实际调整**：与 PR2 (MoE) 遇到相同情况 — 上游 `origin/main` 做了大规模目录重组（`src/MaxText/` → `src/maxtext/`，详见 PR2 RFC）。虽然 MLA 改动仅涉及 `attentions.py`（已有文件修改）和一个新增测试文件，rebase 冲突风险较低，但为保持与 PR2 一致的工作流，**统一改为直接基于 `origin/main` 手动移植**：

1. 从 `origin/main`（`4d5f0bdd`）创建 `feature/mla-attention` 分支
2. 读取 ant-pretrain 中 `attentions.py` 的 diff（`git diff d4ed2261..HEAD -- src/MaxText/layers/attentions.py`）
3. 将 diff 对应的两处修改应用到上游新路径 `src/maxtext/layers/attentions.py`
4. 将测试文件写入上游新路径 `tests/unit/test_mla_interleaved_rope.py`
5. 添加 `mla_interleaved_rope` 配置字段到 `types.py` 和 `base.yml`

MLA 改动极小（attentions.py +13/-1），无 Argus dump hooks，移植过程无复杂冲突。

### 执行步骤

1. 基于 `origin/main` (`4d5f0bdd`) 创建 `feature/mla-attention` 分支 ✅
2. 读取 ant-pretrain diff，移植 `attentions.py` 两处修改（注释修复 + MLA RoPE 分支） ✅
3. 添加 `mla_interleaved_rope` 配置字段（types.py + base.yml） ✅
4. 写入测试文件 `test_mla_interleaved_rope.py`（209 行，6 个测试场景） ✅
5. 运行测试：6/6 通过 ✅
6. 提交 commit (`6e7c6f4f`) 并 push ✅
7. 提交 RFC PR（本文档）
8. 创建代码 PR → `primatrix/maxtext:main`
9. 数值对比验证

**代码分支**：[`feature/mla-attention`](https://github.com/primatrix/maxtext/tree/feature/mla-attention)

## 测试计划

### 单元测试（6 个场景）

| 用例 | 覆盖分支 |
|------|---------|
| `test_yarn_interleave_true_skips_external_deinterleave` | YaRN interleave=True → 跳过 |
| `test_mla_interleaved_rope_false_no_deinterleave` | mla_interleaved_rope=False → 不做 |
| `test_non_yarn_mla_interleaved_deinterleaves` | 非 YaRN + True → 外部 de-interleave |
| `test_partial_rotary_non_yarn_deinterleaves_rot_portion` | partial_rotary → 仅 rotary 部分 |
| `test_yarn_interleave_false_deinterleaves` | YaRN interleave=False → 外部 de-interleave |
| `test_non_mla_no_deinterleave` | 非 MLA type → 不进入 |

### 数值对比

| 对比项 | 容差 |
|--------|------|
| interleaved RoPE (MaxText de-interleave vs Megatron) | BF16 精确一致 |
| MLA forward output (DeepSeek-V2/V3 权重) | BF16 < 1e-5 |

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| `apply_rotary_embedding` 修改影响其他 attention 类型 | 高 | `AttentionType.MLA` 守卫 + 6 个测试 |
| 上游 `attentions.py` 有冲突 | 中 | 变更仅 13 行，冲突可手动解决 |

## References

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA 原始论文
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE 维度排列约定
- [YaRN](https://arxiv.org/abs/2309.00071) — YaRN interleave 模式
- Megatron-LM `megatron/core/models/deepseek/deepseek_attention.py`
