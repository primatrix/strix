---
title: "RFC-PR2: MoE Alignment & Fixes"
status: draft
author: fdz-1999
date: 2026-03-31
reviewers: []
---

# RFC-PR2: MoE Alignment & Fixes

**Depends on**: PR1 (Config Extensions)
**代码分支**: [`feature/moe-alignment`](https://github.com/primatrix/maxtext/tree/feature/moe-alignment)

---

## 1. 动机与目标

### 1.1 背景

ant-pretrain 在 MaxText 基础上对齐 Megatron-LM 的 MoE 实现，核心目标是保证 **相同权重 + 相同输入 → 相同 loss**。在对齐过程中发现上游 MaxText 的 MoE 实现在以下方面与 Megatron-LM 存在差异或缺失：

1. **z-loss 缺失**：Megatron-LM 支持 router z-loss（ST-MoE 论文），MaxText 无此功能
2. **Router 数值精度**：Megatron-LM 的 router gate matmul 在 fp32 下执行，MaxText 跟随模型 dtype（通常 bf16），导致 top-k 路由决策不同
3. **Expert counts / bias 更新**：原有 `calculate_load_balance_updates` 在 GA（Gradient Accumulation）场景下无法正确累积跨微批次的 expert counts
4. **GA aux loss 聚合 bug**：aux loss（lb_loss, z_loss, mtp_loss）在 GA scan 后未除以 GA steps，导致有效 loss 值被 GA 步数倍放大
5. **Router stats 缺失**：缺乏 router 行为的监控指标（bias 均值/标准差、topk 权重、路由概率分布）
6. **MlpBlock activation 精度**：SwiGLU 激活（`silu(gate) * up`）的 fp32 计算由 `cfg.activations_in_float32` 条件控制，而 Megatron-LM / GPU `torch.compile` 无条件在 fp32 下执行

### 1.2 Megatron-LM 参考

- **z-loss**: `megatron/core/transformer/moe/moe_utils.py:z_loss_func` — `mean(logsumexp(logits)^2) * coeff`
- **fp32 routing**: `megatron/core/transformer/moe/router.py` — gate matmul 和 top-k 在 fp32 下执行
- **Expert bias**: `megatron/core/transformer/moe/router.py:TopKRouter` — loss-free load balancing via learnable bias

### 1.3 目标

将以上对齐修复推回上游 MaxText，保证：

- 所有新功能可选，默认关闭时零开销
- 向后兼容，不改变现有用户行为
- GA bug fix 正确修复聚合逻辑

---

## 2. 设计方案

### 2.1 Z-loss 实现

**公式（ST-MoE paper）：**

```text
z_loss = mean(logsumexp(logits, axis=-1)^2) * coeff / tp_size
```

**实现位置**：`GateLogit.__call__()` — 在 raw logits 上（score_func 之前）计算。

**传播路径**：

```text
moe.py GateLogit (计算 z_loss)
  → moe.py RoutedMoE (5-tuple 返回)
    → deepseek.py / gpt_oss.py / mixtral.py / llama4.py / qwen3.py (sow z_loss)
      → train.py loss_fn (_collect_moe_intermediate_sum 收集跨层 z_loss)
        → train.py (加入 total loss)
```

**配置**：`moe_z_loss_weight`（默认 0，关闭）。有效系数 = `moe_z_loss_weight / tp_size`。

### 2.2 FP32 Router

**变更**：

- `GateLogit.__call__()`: input 和 kernel 强制 cast 到 `jnp.float32`
- `RoutedMoE`: `matmul_precision` 强制为 `"highest"`
- `get_topk()`: gate_logits 和 pre_bias_logits cast 到 `jnp.float32`
- `deepseek_scale_weights()`: weights cast 到 `jnp.float32`
- Softmax 输出不再 cast 回 `self.dtype`

**理由**：保证 top-k 路由决策在 fp32 下执行，与 Megatron 一致，避免 bf16 精度导致的路由差异。

### 2.3 Expert Counts 分离

**原有**：`calculate_load_balance_updates()` 同时计算 expert counts 和 bias update。

**重构**：拆分为两个函数：

- `calculate_expert_counts(top_k_indices, num_experts)` — 返回原始 bincount
- `expert_counts_to_bias_update(expert_counts, num_experts, rate)` — 计算 `sign(avg_load - counts) * rate`

**理由**：GA 场景下，expert counts 需要跨微批次累积后再计算 bias update。原有函数无法支持。

### 2.4 RoutedMoE 返回值扩展

**原有**：3-tuple `(output, lb_loss, bias_updates)`

**新**：5-tuple `(output, lb_loss, z_loss, expert_counts, router_stats)`

| 位置 | 含义 | 类型 |
|------|------|------|
| 0 | MoE output | Array |
| 1 | load balance loss | Optional[Array] |
| 2 | z-loss | Optional[Array] |
| 3 | expert counts (raw bincount) | Optional[Array] |
| 4 | router stats dict | Optional[dict] |

### 2.5 GA Aux Loss 聚合修复 (Bug Fix)

**问题**：`gradient_accumulation.py` scan 后 `moe_lb_loss`、`moe_z_loss`、`mtp_loss` 累加了 GA steps 次但未除以 GA steps，导致有效 loss 被放大。

**修复**：scan 后对所有 aux loss 除以 `gradient_accumulation_steps`：

```python
aux["moe_lb_loss"] = aux["moe_lb_loss"] / config.gradient_accumulation_steps
aux["moe_z_loss"] = aux["moe_z_loss"] / config.gradient_accumulation_steps
aux["mtp_loss"] = aux["mtp_loss"] / config.gradient_accumulation_steps
```

同时 `moe_expert_counts` 也需除以 GA steps，否则 bias update rate 被隐式放大。

### 2.6 Router Stats

新增 `router_stats` dict（仅训练时计算）：

- `router_bias_mean` / `router_bias_std` — bias 参数的均值/标准差
- `router_topk_weight_mean` — top-k softmax 概率的均值
- `router_probs_std` — 每 token 路由概率分布的标准差均值

通过 sow 机制传播到 train.py，记录到 metric_logger。

### 2.7 Metric Logger 增强

新增日志字段：

- `learning/lm_loss` — 纯 LM loss（不含 aux loss）
- `learning/moe_z_loss` — raw z-loss 值
- `learning/moe_lb_loss` — load balance loss
- `learning/router_*` — router stats
- `learning/grad_norm` / `learning/raw_grad_norm` — 梯度范数
- `learning/num_zeros` / `learning/is_nan` / `learning/is_inf` — 梯度健康指标
- Loss 精度从 `.3f` 提升到 `.6f`

### 2.8 MlpBlock FP32 Activation 修复

**问题**：`MlpBlock.__call__()` 中 SwiGLU 激活（`silu(gate) * up`）的 fp32 计算此前由 `cfg.activations_in_float32` 配置控制。而 Megatron-LM / GPU `torch.compile` 会将 `silu(gate) * up` 融合为一个 kernel，所有中间结果保持在 fp32 寄存器中。MaxText 在 bf16 下执行激活函数会导致数值差异。

**修复**：将 activation 的 fp32 cast 改为**无条件执行**，不再依赖 `activations_in_float32` 配置：

```python
# 修改前（条件性 fp32）：
if cfg.activations_in_float32:
    x = x.astype(jnp.float32)
x = _convert_to_activation_function(act_fn)(x)

# 修改后（无条件 fp32）：
x = _convert_to_activation_function(act_fn)(x.astype(jnp.float32))
```

elementwise product（`silu(gate) * up`）在 fp32 下完成后再 cast 回 `self.dtype`（通常 bf16）。

### 2.9 备选方案

| 方案 | 不采用的原因 |
|------|------------|
| z-loss 放在 softmax 后计算 | ST-MoE 论文和 Megatron 均在 raw logits 上计算 |
| 不做 fp32 routing | bf16 精度会导致 top-k 路由决策不一致 |
| MlpBlock fp32 保持条件控制 | Megatron 无条件 fp32，条件控制会导致数值不一致 |

---

## 3. 涉及文件

### 修改文件

| 文件 | 变更说明 |
|------|---------|
| `src/maxtext/layers/moe.py` | z-loss, fp32 routing, expert counts 分离, 5-tuple 返回, router stats |
| `src/maxtext/layers/linears.py` | MlpBlock activation 无条件 fp32 |
| `src/maxtext/trainers/pre_train/train.py` | z-loss 收集, expert counts 收集, bias update 逻辑, metrics |
| `src/maxtext/utils/gradient_accumulation.py` | GA bug fix, z-loss/expert_counts 累积 |
| `src/maxtext/utils/maxtext_utils.py` | bias utility 函数 |
| `src/maxtext/common/metric_logger.py` | 新增日志字段, 精度提升 |
| `src/maxtext/models/deepseek.py` | 5-tuple 解包, z-loss/expert_counts sow |
| `src/maxtext/models/gpt_oss.py` | 5-tuple 解包, z-loss sow |
| `src/maxtext/models/mixtral.py` | 5-tuple 解包, z-loss sow |
| `src/maxtext/models/llama4.py` | 5-tuple 解包, z-loss sow |
| `src/maxtext/models/qwen3.py` | 5-tuple 解包, z-loss sow |
| `src/maxtext/configs/types.py` | 新增配置字段 |
| `src/maxtext/configs/base.yml` | 新增默认值 |
| `src/maxtext/common/common_types.py` | LING2 枚举 |
| `tests/unit/moe_test.py` | API 更新, Ling2RoutingTest, ZeroMeanStateParamTest |
| `tools/moe_cross_compare.py` | MoE 跨框架精度对比工具 |

### 依赖的配置字段 (via PR1)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `moe_z_loss_weight` | 0.0 | z-loss 权重 |
| `moe_shared_expert_intermediate_size` | 0 | shared expert 中间层大小 |
| `routed_bias_dtype` | "" | bias 参数 dtype |
| `enable_routed_bias_grad` | True | bias 是否接收优化器梯度 |
| `routed_bias_zero_mean_update` | False | bias 零均值更新 |
| `calculate_per_token_loss` | True | GA per-token loss 模式 |

---

## 4. Argus Dump 剥离清单

以下 Argus dump 调用在提交上游前需删除：

| 文件 | 类型 | 内容 |
|------|------|------|
| `moe.py` | import | `from MaxText.utils.argus_utils import dump_tensor as _dump_tensor` |
| `moe.py` | dump 调用 ×6 | sparse_matmul input/output, dense_matmul weights, RoutedMoE input/router output, RoutedAndSharedMoE input/output |
| `moe.py` | 变量 ×3 | `layer_name_experts` / `layer_name` 赋值（仅用于 dump） |
| `train.py` | import | `from MaxText.utils import argus_utils` |
| `train.py` | 参数 | `dump_enabled=False` in train_step |
| `train.py` | 返回值 | `raw_grads if dump_enabled else None` |
| `train.py` | 实例化 | `dumper = argus_utils.Dumper(config)` |
| `train.py` | 调用 ×3 | `dumper.save()` / `dumper.is_dump_step()` |

---

## 5. 向后兼容性

| 方面 | 影响 |
|------|------|
| `moe_z_loss_weight=0`（默认） | 不计算 z-loss，零开销 |
| `routed_bias=False`（默认） | 不影响 router bias 逻辑 |
| FP32 routing | **行为变更**：router 精度提升。可能导致不同的 top-k 路由决策，但结果更准确 |
| MlpBlock fp32 activation | **行为变更**：activation 无条件 fp32，不再依赖 `activations_in_float32` 配置 |
| 5-tuple 返回 | decoder layer 文件需同步更新解包逻辑 |
| GA bug fix | **行为变更（修正）**：修复前 aux loss 被 GA steps 倍放大，修复后正确 |
| `calculate_per_token_loss=True`（默认） | 与原有行为一致 |
| 所有 non-MoE 模型 | 无影响（MoE 代码路径仅在 `num_experts > 1` 时激活） |

---

## 6. 测试计划

### 6.1 单元测试

```bash
python tools/moe_cross_compare.py \
  --megatron-dump-dir <dump_dir>/step_1 \
  --orbax-ckpt-path <ckpt_path> \
  --ling2-profile \
  --layers 0,1 \
  --ci
```
