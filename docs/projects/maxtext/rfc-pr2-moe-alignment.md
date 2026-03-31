---
title: "RFC-PR2: MoE Alignment & Fixes"
status: draft
author: fdz-1999
date: 2026-03-31
reviewers: []
---

# RFC-PR2: MoE Alignment & Fixes

## 概述

将 ant-pretrain 中对齐 Megatron-LM 的 MoE 修改推回上游 MaxText，包括 z-loss、fp32 routing、expert counts 分离、GA aux loss 聚合修复和 router 监控指标。

## 背景

ant-pretrain 在 MaxText 基础上对齐 Megatron-LM 的 MoE 实现，核心目标是保证 **相同权重 + 相同输入 → 相同 loss**。在对齐过程中发现上游 MaxText 的 MoE 实现在以下方面与 Megatron-LM 存在差异或缺失：

1. **z-loss 缺失**：Megatron-LM 支持 router z-loss（ST-MoE 论文），MaxText 无此功能
2. **Router 数值精度**：Megatron-LM 的 router gate matmul 在 fp32 下执行，MaxText 跟随模型 dtype（通常 bf16），导致 top-k 路由决策不同
3. **Expert counts / bias 更新**：原有 `calculate_load_balance_updates` 在 GA（Gradient Accumulation）场景下无法正确累积跨微批次的 expert counts
4. **GA aux loss 聚合 bug**：aux loss（lb_loss, z_loss, mtp_loss）在 GA scan 后未除以 GA steps，导致有效 loss 值被 GA 步数倍放大
5. **Router stats 缺失**：缺乏 router 行为的监控指标（bias 均值/标准差、topk 权重、路由概率分布）

### Megatron-LM 参考

- **z-loss**: `megatron/core/transformer/moe/moe_utils.py:z_loss_func` — `mean(logsumexp(logits)^2) * coeff`
- **fp32 routing**: `megatron/core/transformer/moe/router.py` — gate matmul 和 top-k 在 fp32 下执行
- **Expert bias**: `megatron/core/transformer/moe/router.py:TopKRouter` — loss-free load balancing via learnable bias

## 方案

### Z-loss 实现

**公式（ST-MoE paper）：**
```
z_loss = mean(logsumexp(logits, axis=-1)^2) * coeff / tp_size
```

**实现位置**：`GateLogit.__call__()` — 在 raw logits 上（score_func 之前）计算。

**传播路径**：
```
moe.py GateLogit (计算 z_loss)
  → moe.py RoutedMoE (5-tuple 返回)
    → deepseek.py / gpt_oss.py / mixtral.py / llama4.py / qwen3.py (sow z_loss)
      → train.py loss_fn (_collect_moe_intermediate_sum 收集跨层 z_loss)
        → train.py (加入 total loss)
```

**配置**：`moe_z_loss_coeff`（默认 0，关闭）。有效系数 = `moe_z_loss_coeff / tp_size`。

### FP32 Router

- `GateLogit.__call__()`: input 和 kernel 强制 cast 到 `jnp.float32`
- `RoutedMoE`: `matmul_precision` 强制为 `"highest"`
- `get_topk()`: gate_logits 和 pre_bias_logits cast 到 `jnp.float32`

### Expert Counts 分离

**原有**：`calculate_load_balance_updates()` 同时计算 expert counts 和 bias update。

**重构**：拆分为两个函数：
- `calculate_expert_counts(top_k_indices, num_experts)` — 返回原始 bincount
- `expert_counts_to_bias_update(expert_counts, num_experts, rate)` — 计算 `sign(avg_load - counts) * rate`

GA 场景下，expert counts 需要跨微批次累积后再计算 bias update。

### RoutedMoE 返回值扩展

**原有**：3-tuple `(output, lb_loss, bias_updates)`
**新**：5-tuple `(output, lb_loss, z_loss, expert_counts, router_stats)`

### GA Aux Loss 聚合修复 (Bug Fix)

scan 后对所有 aux loss 除以 `gradient_accumulation_steps`：
```python
aux["moe_lb_loss"] = aux["moe_lb_loss"] / config.gradient_accumulation_steps
aux["moe_z_loss"] = aux["moe_z_loss"] / config.gradient_accumulation_steps
aux["mtp_loss"] = aux["mtp_loss"] / config.gradient_accumulation_steps
```

### Router Stats

新增 `router_stats` dict：`router_bias_mean/std`、`router_topk_weight_mean`、`router_probs_std`。

### 备选方案

1. **z-loss 放在 softmax 后计算** — 未采用，ST-MoE 论文和 Megatron 均在 raw logits 上计算
2. **不做 fp32 routing** — 未采用，bf16 精度会导致 top-k 路由决策不一致

## 影响范围

| 方面 | 影响 |
|------|------|
| `moe_z_loss_coeff=0`（默认） | 不计算 z-loss，零开销 |
| FP32 routing | **行为变更**：router 精度提升，可能导致不同的 top-k 路由决策 |
| GA bug fix | **行为变更（修正）**：修复前 aux loss 被 GA steps 倍放大 |
| 5-tuple 返回 | decoder layer 文件需同步更新 |
| 非 MoE 模型 | 无影响（MoE 代码路径仅在 `num_experts > 1` 时激活） |

### 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/maxtext/layers/moe.py` | 修改 | z-loss, fp32 routing, expert counts 分离, 5-tuple, router stats |
| `src/maxtext/trainers/pre_train/train.py` | 修改 | z-loss 收集, expert counts, bias update, metrics |
| `src/maxtext/utils/gradient_accumulation.py` | 修改 | GA bug fix, z-loss/expert_counts 累积 |
| `src/maxtext/utils/maxtext_utils.py` | 修改 | bias utility 函数 |
| `src/maxtext/common/metric_logger.py` | 修改 | 新增日志字段 |
| `src/maxtext/models/{deepseek,gpt_oss,mixtral,llama4,qwen3}.py` | 修改 | 5-tuple 解包 |
| `src/maxtext/configs/{types.py,base.yml}` | 修改 | 新增配置字段 |
| `src/maxtext/common/common_types.py` | 修改 | AL_MODEL 枚举 |
| `tests/unit/moe_test.py` | 修改 | 测试更新 + 新增测试类 |

**总计**：14 files, +992/-136 lines

## 实施计划

### 移植策略

原计划（参考 replan.md Section 7.4）：
1. 基于 fork 基点 `d4ed2261` 创建分支
2. 从 ant-pretrain 零冲突 apply patch
3. `git rebase origin/main` 统一解决冲突

**实际调整**：上游 `origin/main` 在 fork 后的 634 个 commit 中做了**大规模目录重组**：
- `src/MaxText/` → `src/maxtext/`（全小写）
- `src/MaxText/layers/deepseek.py` → `src/maxtext/models/deepseek.py`
- `src/MaxText/train.py` → `src/maxtext/trainers/pre_train/train.py`（旧路径变为 shim）
- `src/MaxText/gradient_accumulation.py` → `src/maxtext/utils/gradient_accumulation.py`
- `src/MaxText/metric_logger.py` → `src/maxtext/common/metric_logger.py`
- `src/MaxText/maxtext_utils.py` → `src/maxtext/utils/maxtext_utils.py`
- `src/MaxText/optimizers.py` → `src/maxtext/optimizers/optimizers.py`
- `tests/moe_test.py` → `tests/unit/moe_test.py`

执行 `git rebase origin/main` 时产生 **10 个文件冲突**，其中 `train.py` 冲突尤为严重（上游将其替换为 20 行 shim，实际逻辑移至新路径），手动解决容易出错。

**改为直接基于 `origin/main` 手动移植**：
1. 从 `origin/main`（`4d5f0bdd`）创建 `feature/moe-alignment` 分支
2. 逐文件读取 ant-pretrain 源码和上游目标文件，对比差异
3. 将 MoE 修改应用到上游新路径，同步更新 import 路径（`MaxText.xxx` → `maxtext.xxx`）
4. 剥离所有 Argus dump hooks

**效果一致**：最终分支基于 `origin/main`，diff 仅包含 MoE 相关修改，与 rebase 方案结果相同。

### 执行步骤

1. 基于 `origin/main` (`4d5f0bdd`) 创建 `feature/moe-alignment` 分支 ✅
2. 逐文件对比 ant-pretrain 和上游，移植 MoE 修改到新目录结构 ✅
3. 剥离 Argus dump hooks（moe.py 10 处、train.py 5 处） ✅
4. 更新 import 路径适配上游代码风格 ✅
5. 语法验证通过 ✅
6. 提交 commit (`52e85564`) 并 push ✅
7. 提交 RFC PR（本文档）
8. 创建代码 PR → `primatrix/maxtext:main`
9. 数值对比验证（ant-pretrain vs upstream）

**代码分支**：[`feature/moe-alignment`](https://github.com/primatrix/maxtext/tree/feature/moe-alignment)

## 测试计划

### 单元测试
- `moe_test.py::RoutedMoeTest` — 5-tuple, z-loss, fp32 routing
- `moe_test.py::AlModelRoutingTest` — AL_MODEL 路由
- `moe_test.py::ZeroMeanStateParamTest` — bias 零均值更新

### 数值对比
| 对比项 | 容差 |
|--------|------|
| load balance loss 逐层值 | BF16 < 1e-5 |
| z-loss (moe_z_loss_coeff > 0) | BF16 < 1e-5 |
| z-loss 零开销 (coeff=0) | 无 tensor |
| router bias update | BF16 < 1e-5 |
| GA aux metrics 聚合 | 精确一致 |
| 单步 forward loss | BF16 < 1e-5 |

## 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| FP32 routing 改变所有 MoE 用户的路由行为 | 高 | PR 描述中说明精度提升的理由和数值对比 |
| 5-tuple 返回影响所有 decoder layer 文件 | 中 | 统一更新 5 个 decoder 文件 |
| GA bug fix 改变某些用户的训练曲线 | 低 | 明确标注为 bug fix |

## References

- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- Megatron-LM `megatron/core/transformer/moe/router.py`
