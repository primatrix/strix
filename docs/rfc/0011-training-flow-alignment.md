# RFC: Refactor PR10 - 训练主流程对齐

| 字段     | 值                                    |
| -------- | ------------------------------------- |
| **作者** | @Garrybest                            |
| **日期** | 2026-03-30                            |
| **状态** | Draft                                 |
| **PR**   | PR10（PR2、PR7 的前置依赖，依赖 PR1） |

## 动机

ant-pretrain 对 MaxText 的训练主流程（`train.py`、`gradient_accumulation.py`）做了一系列修改，核心目标是让 MaxText 的 loss 计算、梯度归一化和训练指标报告与 Megatron-LM 对齐。这些修改是通用的训练基础设施改动，不绑定任何特定模型架构（MoE、MTP 等），但为后续 MoE 对齐（PR2）和 MTP 对齐（PR7）提供了必要的基础。

PR10 是从原 upstream-refactoring RFC 的 9-PR 计划中新拆分出来的 PR。原计划中 GA 和 train.py 的基础设施改动分散在 PR2（MoE 对齐）中，但这些改动实际上是通用训练基础设施，不应与 MoE 具体逻辑绑定，因此独立为 PR10。

**目标：**

1. 支持 `calculate_per_token_loss` 双模式 GA 梯度归一化，使 MaxText 能够复现 Megatron 的 loss 数值
1. 修复 GA 场景下 aux metrics 日志指标被 `gradient_accumulation_steps` 倍数放大的 bug
1. 将纯 LM loss 与辅助损失分离，使日志中的 loss 指标可与 Megatron 直接对比
1. 添加训练健康监控指标（梯度零值计数、NaN/Inf 检测）

## PR 依赖关系

```text
PR1: 配置扩展 (base.yml + types.py + common_types.py)
 ├── PR10: 训练主流程对齐 (GA + Loss + 监控)     ← 本 PR
 │    ├── PR2: MoE 对齐与修复 (依赖 PR10)
 │    └── PR7: Multi-Token Prediction (依赖 PR6 + PR10)
 ├── PR3: Megatron MMap 数据管道
 ├── PR4: GLA 线性注意力
 ├── PR5: MLA 注意力
 │    └── PR6: AL Model Decoder 集成 (依赖 PR4+PR5)
 │         └── PR8: 检查点转换扩展 (依赖 PR6)
 └── PR9: FP8 GEMM/Group GEMM
```

- PR10 依赖 PR1（`calculate_per_token_loss` 配置字段）
- PR2 依赖 PR10（GA 基础设施和 loss 分离）
- PR7 依赖 PR6 + PR10

## 涉及文件

| 文件                                         | 变更类型 | 说明                                                                    |
| -------------------------------------------- | -------- | ----------------------------------------------------------------------- |
| `src/maxtext/utils/gradient_accumulation.py` | 修改     | GA 双模式归一化、aux metrics 平均化 bug fix、`lm_loss_per_token` 累加器 |
| `src/maxtext/trainers/pre_train/train.py`    | 修改     | `loss_fn` 纯 LM loss 分离、`train_step` 监控指标、eval batch placement  |
| `scripts/pretrain_ling2.sh`                  | 新增     | Ling2 一键训练脚本，从 ant-pretrain 迁移并适配 maxtext 模块路径         |

## 核心修改逻辑：完整代码示例

以下通过三个关键代码流的 upstream vs 变更后对比，展示 PR10 的完整修改逻辑。

### A. `loss_fn` 返回逻辑对比

**upstream（`train.py`）：** aux losses 直接加到 loss 上，GA 条件使用 `use_tunix_gradient_accumulation`：

```python
# === upstream loss_fn ===

# 1. 计算 cross-entropy（含 z_loss）
xent, z_loss = max_utils.cross_entropy_with_logits(
    logits, one_hot_targets, z_loss=config.z_loss_multiplier
)
xent = xent * (data["targets_segmentation"] != 0)  # mask padding
total_loss = jnp.sum(xent)

# 2. GA 条件分支
total_weights = jnp.sum(data["targets_segmentation"] != 0)
if (
    config.gradient_accumulation_steps > 1
    and not config.use_tunix_gradient_accumulation
):
    loss = total_loss  # 返回 raw sum，GA 后续除以 total_weights
else:
    loss = total_loss / (total_weights + EPS)  # 直接 per-token 平均

# 3. z_loss 归一化后加入
total_z_loss = total_z_loss / (total_weights + EPS)

# 4. 所有 aux losses 直接加到 loss 上（不区分 GA 模式）
loss += mtp_loss
loss += moe_lb_loss
loss += indexer_loss

# 5. 返回
aux = {
    "total_loss": total_loss,
    "z_loss": total_z_loss,
    "total_weights": total_weights,
    "moe_lb_loss": moe_lb_loss,
    "indexer_loss": indexer_loss,
    "mtp_loss": mtp_loss,
    "moe_bias_updates": moe_bias_updates,
}
return loss, aux
```

**变更后：** GA 条件新增 `calculate_per_token_loss`，同时保留 `use_tunix_gradient_accumulation` 兼容性。aux losses 在 GA per-token 模式下乘以 `total_weights` 补偿：

```python
# === 变更后 loss_fn ===

# 1. 计算 cross-entropy（保留 z_loss 传递，不做修改）
xent, z_loss = max_utils.cross_entropy_with_logits(
    logits, one_hot_targets, z_loss=config.z_loss_multiplier
)
xent = xent * (data["targets_segmentation"] != 0)
total_loss = jnp.sum(xent)

# 2. GA 条件分支（保留 use_tunix_gradient_accumulation，新增 calculate_per_token_loss）
#    返回 raw sum 的条件：GA>1 且 per-token 模式 且非 Tunix GA
total_weights = jnp.sum(data["targets_segmentation"] != 0)
use_ga_raw_sum = (
    config.gradient_accumulation_steps > 1
    and config.calculate_per_token_loss
    and not config.use_tunix_gradient_accumulation
)
if use_ga_raw_sum:
    loss = total_loss  # raw sum，GA 后续除以 total_weights
else:
    loss = total_loss / (total_weights + EPS)  # per-token 平均

# 3. aux losses 区分 GA 模式处理
#    GA per-token 模式下，loss 会被 GA 除以 total_weights，
#    所以 aux losses 需先乘以 total_weights 抵消。
if use_ga_raw_sum:
    loss += mtp_loss * total_weights  # 乘以 total_weights 补偿
    loss += moe_lb_loss * total_weights
else:
    loss += mtp_loss  # 非 GA / Megatron 模式 / Tunix GA，直接加
    loss += moe_lb_loss

# 4. 返回——aux dict 独立传递用于日志
aux = {
    "total_loss": total_loss,
    "total_weights": total_weights,
    "moe_lb_loss": moe_lb_loss,
    "mtp_loss": mtp_loss,
}
return loss, aux
```

### B. `gradient_accumulation.py` 累加与归一化逻辑对比

**upstream：** 累加器只累加 raw 值，scan 后统一归一化并混合所有 losses：

```python
# === upstream gradient_accumulation.py ===

# 1. 初始化累加器
init_grad_and_loss = {
    "loss": 0.0,
    "grad": init_grad,
    "total_weights": 0,
    "moe_lb_loss": 0.0,
    "indexer_loss": 0.0,
    "mtp_loss": 0.0,
    "ga_params": ga_params,
}


# 2. scan 内部：逐 micro-batch 累加
def accumulate_gradient(acc, data):
    (_, aux), grads = grad_func(...)
    acc["loss"] += aux["total_loss"]  # 累加 CE raw sum
    acc["moe_lb_loss"] += aux["moe_lb_loss"]  # 累加 aux losses
    acc["indexer_loss"] += aux["indexer_loss"]
    acc["mtp_loss"] += aux["mtp_loss"]
    acc["grad"] = tree_map(add, grads, acc["grad"])
    acc["total_weights"] += aux["total_weights"]
    return acc, aux


# 3. scan 后归一化——返回混合 loss
grad_and_loss, aux = jax.lax.scan(accumulate_gradient, init, data)
loss = (
    grad_and_loss["loss"] / grad_and_loss["total_weights"]  # LM: 除以全局 token 数
    + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps  # aux: 除以 K
    + grad_and_loss["indexer_loss"] / config.gradient_accumulation_steps
    + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
)
raw_grads = tree_map(
    lambda x: x / grad_and_loss["total_weights"], raw_grads
)  # 梯度除以全局 token 数

return loss, aux, raw_grads  # loss = LM + aux 混合值
```

**变更后：** 新增 `lm_loss_per_token` 累加器，支持双模式，返回纯 LM loss，aux metrics 做平均：

```python
# === 变更后 gradient_accumulation.py ===

# 1. 初始化累加器（新增 lm_loss_per_token）
init_grad_and_loss = {
    "loss": 0.0,
    "grad": init_grad,
    "total_weights": 0,
    "moe_lb_loss": 0.0,
    "mtp_loss": 0.0,
    "lm_loss_per_token": 0.0,  # 新增：Megatron 模式用
    "ga_params": ga_params,
}


# 2. scan 内部：逐 micro-batch 累加
def accumulate_gradient(acc, data):
    (_, aux), grads = grad_func(...)
    acc["loss"] += aux["total_loss"]
    acc["moe_lb_loss"] += aux["moe_lb_loss"]
    acc["mtp_loss"] += aux["mtp_loss"]
    acc["grad"] = tree_map(add, grads, acc["grad"])
    acc["total_weights"] += aux["total_weights"]
    # 新增：Megatron 模式下，每个 micro-batch 独立算 per-token 平均后累加
    if not config.calculate_per_token_loss:
        acc["lm_loss_per_token"] += aux["total_loss"] / (aux["total_weights"] + EPS)
    return acc, aux


# 3. scan 后归一化——返回纯 LM loss
grad_and_loss, aux = jax.lax.scan(accumulate_gradient, init, data)

# 双模式梯度归一化
grad_divisor = (
    grad_and_loss["total_weights"]  # MaxText 默认：全局 per-token
    if config.calculate_per_token_loss
    else config.gradient_accumulation_steps  # Megatron 模式：等权平均
)

# 纯 LM loss（不含 aux losses）
if config.calculate_per_token_loss:
    loss = grad_and_loss["loss"] / grad_and_loss["total_weights"]
else:
    loss = grad_and_loss["lm_loss_per_token"] / config.gradient_accumulation_steps

raw_grads = tree_map(lambda x: x / grad_divisor, raw_grads)

# 新增：aux metrics 平均化（修复 K 倍放大 bug）
aux = tree_map(lambda x: jnp.sum(x, axis=0), aux)
aux["mtp_loss"] = aux["mtp_loss"] / config.gradient_accumulation_steps
aux["moe_lb_loss"] = aux["moe_lb_loss"] / config.gradient_accumulation_steps

return loss, aux, raw_grads  # loss = 纯 LM loss
```

### C. `train_step` 指标计算逻辑对比

**upstream：** `loss` 是混合值（含 aux），直接上报：

```python
# === upstream train_step ===
if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(...)
else:
    (loss, aux), raw_grads = grad_func(...)

# loss 是混合值（LM + moe_lb_loss + mtp_loss + indexer_loss）
scalar_metrics = {
    "learning/loss": loss,  # 混合 loss
    "learning/z_loss": aux["z_loss"],
    "learning/moe_lb_loss": aux["moe_lb_loss"],
    "learning/indexer_loss": aux["indexer_loss"],
    "learning/mtp_loss": aux["mtp_loss"],
    "learning/total_weights": aux["total_weights"],
}
```

**变更后：** 新增 `lm_loss` 纯 LM loss 指标，`learning/loss` 保持混合 loss 语义不变：

```python
# === 变更后 train_step ===
if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(...)
else:
    (loss, aux), raw_grads = grad_func(...)

total_loss = aux["total_loss"]
total_weights = aux["total_weights"]

# 计算纯 LM loss（两条路径）
if config.gradient_accumulation_steps > 1:
    lm_loss = loss  # GA 已返回纯 LM loss
else:
    lm_loss = total_loss / (total_weights + EPS)  # 从 aux 中计算

scalar_metrics = {
    # learning/loss 保持 upstream 混合 loss 语义（LM + aux losses）
    "learning/loss": lm_loss + aux["moe_lb_loss"] + aux["mtp_loss"],
    # 新增：纯 LM loss 和各 aux loss 独立上报
    "learning/lm_loss": lm_loss,  # 对标 Megatron "lm loss"
    "learning/moe_lb_loss": aux["moe_lb_loss"],
    "learning/mtp_loss": aux["mtp_loss"],
    "learning/total_weights": total_weights,
}

# 新增：训练健康监控
if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/num_zeros"] = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(x == 0),
        raw_grads,
        initializer=jnp.array(0, dtype=jnp.int64),
    )
scalar_metrics["learning/is_nan"] = jnp.any(jnp.isnan(lm_loss)).astype(jnp.int32)
scalar_metrics["learning/is_inf"] = jnp.any(jnp.isinf(lm_loss)).astype(jnp.int32)
```

## 详细变更

### 1. `gradient_accumulation.py` — GA 双模式归一化

**现状（upstream）：** GA 结束后，LM loss 除以 `total_weights`（全局 token 总数），aux losses（`moe_lb_loss`、`mtp_loss`、`indexer_loss`）各自除以 `gradient_accumulation_steps`，四者相加作为返回的 loss。梯度始终除以 `total_weights`：

```python
# upstream 现状
loss = (
    grad_and_loss["loss"] / grad_and_loss["total_weights"]
    + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
    + grad_and_loss["indexer_loss"] / config.gradient_accumulation_steps
    + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
)
raw_grads = jax.tree_util.tree_map(
    lambda arr: arr / grad_and_loss["total_weights"], raw_grads
)
```

**变更：** 引入 `calculate_per_token_loss`（PR1 新增配置字段）切换两种模式：

| 模式          | `calculate_per_token_loss` | `grad_divisor`                | Loss 返回值                                                                                |
| ------------- | -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| MaxText 默认  | `True`                     | `total_weights`               | `loss / total_weights`（全局 per-token 平均）                                              |
| Megatron 对齐 | `False`                    | `gradient_accumulation_steps` | `lm_loss_per_token / gradient_accumulation_steps`（微批次独立 per-token 平均后取等权均值） |

关键变更：返回的 loss **只包含纯 LM loss**，不再混入 aux losses。Aux losses 通过 `aux` dict 独立传递，由 `train.py` 在 `loss_fn` 中统一处理。

```python
# 变更后
grad_divisor = (
    grad_and_loss["total_weights"]
    if config.calculate_per_token_loss
    else config.gradient_accumulation_steps
)
if config.calculate_per_token_loss:
    loss = grad_and_loss["loss"] / grad_and_loss["total_weights"]
else:
    loss = grad_and_loss["lm_loss_per_token"] / config.gradient_accumulation_steps
raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_divisor, raw_grads)
```

**关于 `use_tunix_gradient_accumulation` 的处理：** upstream 的 `loss_fn` 中使用 `not config.use_tunix_gradient_accumulation` 作为 GA 条件（控制是否返回 raw sum 而非 per-token average）。本 PR **保留该条件**，同时新增 `calculate_per_token_loss` 判断。GA 返回 raw sum 的完整条件为：

```python
use_ga_raw_sum = (
    config.gradient_accumulation_steps > 1
    and config.calculate_per_token_loss
    and not config.use_tunix_gradient_accumulation
)
```

语义：`use_tunix_gradient_accumulation=True`（Tunix GA）和 `calculate_per_token_loss=False`（Megatron 模式）的行为一致——都在 `loss_fn` 中直接返回 per-token 平均值。两者任一满足即走 `else` 分支。这保证了 Tunix SFT 路径（`use_tunix_gradient_accumulation=True`）不受影响。

**向后兼容性：** `calculate_per_token_loss=True` 为默认值，此时 `grad_divisor = total_weights`，与 upstream 现有梯度归一化行为一致。

### 2. `gradient_accumulation.py` — aux metrics 日志平均化（Bug Fix）

**现状（upstream）：** `jax.lax.scan` 在每个 micro-batch 中累加 aux metrics（`moe_lb_loss`、`mtp_loss` 等），scan 结束后对 aux dict 做 `jnp.sum(x, axis=0)` 合并，但不做平均。返回给 `train.py` 的 aux dict 中的值是 K 个 micro-batch 的**总和**。

**注意：** upstream 的 **loss 和梯度计算不受影响**——loss 公式中 aux losses 已除以 `gradient_accumulation_steps`（见第 1 节）。此 bug 仅影响 **aux dict 中传回 `train.py` 用于日志上报的指标值**。当 GA=4 时，日志中的 `moe_lb_loss` 等指标是 GA=1 时的 4 倍。

**修复：** scan 结束后对 aux dict 中的 metrics 除以 `gradient_accumulation_steps`：

```python
# Fix: aux metrics are sums across K microbatches,
# but should be averages for consistent reporting with GA=1 case.
aux["mtp_loss"] = aux["mtp_loss"] / config.gradient_accumulation_steps
aux["moe_lb_loss"] = aux["moe_lb_loss"] / config.gradient_accumulation_steps
```

**影响范围：**

- GA=1 用户：无影响（除以 1）
- GA>1 用户：日志指标报告更准确（修复放大 bug）

### 3. `gradient_accumulation.py` — `lm_loss_per_token` 累加器

**新增：** 在 `calculate_per_token_loss=False` 模式下，每个 micro-batch 独立计算 `total_loss / (total_weights + EPS)`，累加到 `lm_loss_per_token`。最终 loss = `lm_loss_per_token / gradient_accumulation_steps`。

此计算方式与 Megatron 报告的 `lm loss` 数值一致——每个 micro-batch 独立做 per-token 平均后取等权均值，而非全局 token 平均。

**为什么需要这个累加器——两种 per-token 平均的数学差异：**

MaxText 默认和 Megatron 模式的区别在于 per-token 平均的**计算顺序**不同。以 GA=2、两个 micro-batch 为例：

|                               | micro-batch 0 | micro-batch 1 |
| ----------------------------- | ------------- | ------------- |
| total_loss (CE raw sum)       | 100           | 80            |
| total_weights (有效 token 数) | 50            | 20            |

- **MaxText 默认（先累加，再除）**：`(100 + 80) / (50 + 20) = 180 / 70 ≈ 2.571`——全局 per-token 平均，token 多的 micro-batch 贡献更大权重。
- **Megatron 模式（先除，再平均）**：`(100/50 + 80/20) / 2 = (2.0 + 4.0) / 2 = 3.0`——每个 micro-batch 等权贡献，token 少的 micro-batch 不会因 token 少而被"稀释"。

当 micro-batch 间有效 token 数量不均匀时，两种方式会产生数值差异。

`jax.lax.scan` 逐个处理 micro-batch，scan 结束后只有 `sum(loss_i)` 和 `sum(weights_i)`，只能算出 MaxText 方式的结果（`sum(loss_i) / sum(weights_i)`），无法还原 Megatron 方式（`mean(loss_i / weights_i)`）。因此需要在 scan **过程中**就把每个 micro-batch 的 `loss_i / weights_i` 累加到 `lm_loss_per_token`。

**涉及改动：**

- `init_grad_and_loss` 新增 `"lm_loss_per_token": 0.0`
- `accumulate_gradient` 中每个 micro-batch 累加：`acc["lm_loss_per_token"] += aux["total_loss"] / (aux["total_weights"] + EPS)`（`EPS` 来自 `MaxText.globals`，值为 `1e-6`，防止除零）
- 最终 loss 计算分支（见第 1 节代码）

### 4. `train.py` — `loss_fn` GA 模式 loss 构建

**现状（upstream）：** `loss_fn` 中 GA 条件为 `gradient_accumulation_steps > 1 and not use_tunix_gradient_accumulation`，满足时返回 raw sum（`loss = total_loss`），否则返回 per-token average（`loss = total_loss / (total_weights + EPS)`）。所有 aux losses 直接加到 loss 上。

Upstream 的 aux losses 处理逻辑**分散在两个地方**：

- **非 GA 路径**（在 `loss_fn` 中）：`loss = total_loss / total_weights + moe_lb_loss + ...`，aux losses 直接加到 per-token 平均后的 loss 上
- **GA 路径**（在 `gradient_accumulation.py` 中）：`loss_fn` 只返回 raw `total_loss`，aux losses 在 GA 函数中单独除以 `gradient_accumulation_steps` 再加回（见第 1 节 upstream 现状代码）

Upstream 这种做法的计算结果是正确的，但 aux losses 的归一化逻辑耦合在 `gradient_accumulation.py` 中。

**变更动机：** 第 1 节要求 GA 函数只返回**纯 LM loss**（不混入 aux losses），因此需要将 aux losses 的处理**统一移到 `loss_fn`**。

**问题：** GA 路径下，`loss_fn` 返回的 loss 最终会被 GA 除以 `total_weights`。如果在 `loss_fn` 中直接加 aux loss：

```python
# 错误做法——aux loss 会被 GA 的 total_weights 除法错误缩小
loss = total_loss + moe_lb_loss
# GA 结束后: final = (total_loss + moe_lb_loss) / total_weights
#                   = total_loss/total_weights + moe_lb_loss/total_weights
#                                                ^^^^^^^^^^^^^^^^^^^^^^^^^ 被错误缩小！
# moe_lb_loss 本身已是 per-token 量级（如 0.01），
# 再除以 total_weights（如 2048）就变成 ~0.000005，完全失去作用。
```

**解决方案：** 先乘以 `total_weights` 抵消 GA 的除法：

```python
# 正确做法——乘以 total_weights 抵消 GA 的除法
loss = total_loss + moe_lb_loss * total_weights
# GA 结束后: final = (total_loss + moe_lb_loss * total_weights) / total_weights
#                   = total_loss/total_weights + moe_lb_loss
#                                                ^^^^^^^^^^^ 保持原始量级，正确！
```

**变更：** GA 条件新增 `calculate_per_token_loss`，同时保留 `use_tunix_gradient_accumulation`：

```python
# 变更后
use_ga_raw_sum = (
    config.gradient_accumulation_steps > 1
    and config.calculate_per_token_loss
    and not config.use_tunix_gradient_accumulation
)
if use_ga_raw_sum:
    # GA per-token 模式：loss 返回 raw sum，GA 中会除以 total_weights。
    # aux losses 已是 per-token 的，需乘以 total_weights 抵消 GA 的除法。
    loss = total_loss
    loss += moe_lb_loss * total_weights
    loss += mtp_loss * total_weights  # MTP loss 同理
else:
    # 非 GA / Megatron 模式 / Tunix GA：直接 per-token 平均
    loss = total_loss / (total_weights + EPS)
    loss += moe_lb_loss
    loss += mtp_loss
# 注：所有 aux losses（moe_lb_loss、mtp_loss、moe_z_loss 等）均使用此模式。
# moe_z_loss 由 PR2 引入，mtp_loss 由 PR7 引入，但 GA 缩放模式由本 PR 建立。
```

**关于 `indexer_loss` 的处理：** upstream 的 `loss_fn` 和 `gradient_accumulation.py` 均包含 `indexer_loss`（`use_indexer` 功能的辅助损失），在 GA 累加器中初始化、累加并除以 `gradient_accumulation_steps`。本 PR 不引入 `indexer_loss` 相关逻辑（Ling2 不使用 indexer）。推送到 primatrix/maxtext 时需确保 **保留** upstream 已有的 `indexer_loss` 代码路径——本 PR 的改动应与 `indexer_loss` 代码并存，不应删除它。

### 5. `train.py` — CE z_loss 保留（不做修改）

**背景：** ant-pretrain 中移除了 CE z_loss 的传递（Ling2 不使用 CE z-loss，`z_loss_multiplier=0`）。但为了保持框架通用性，**本 PR 保留 upstream 的 CE z_loss 逻辑不做修改**：

```python
# 保持 upstream 原样，不做改动
xent, z_loss = max_utils.cross_entropy_with_logits(
    logits, targets, z_loss=config.z_loss_multiplier
)
```

**原因：** CE z_loss 是一种对 logits 大小的正则化（来自 PaLM 论文），部分模型的训练依赖此功能（`z_loss_multiplier > 0`）。如果移除传递逻辑，这些模型的 z_loss 将静默失效，丧失框架通用性。Ling2 使用 `z_loss_multiplier=0`（默认值），保留此逻辑对 Ling2 训练无任何影响。

**注意：** CE z_loss 与 MoE router z-loss 是不同的机制。MoE router z-loss 在 PR2 中处理。

同理，`vocabulary_tiling.py` 的 `vocab_tiling_linen_loss` 返回值保持 `(total_loss, total_z_loss)` 不变。

### 6. `train.py` — `train_step` 监控指标

**新增指标：**

| 指标                 | 类型  | 说明                                                                                                                             |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| `learning/lm_loss`   | float | **新增**。纯 LM 交叉熵 per-token loss（不含 aux losses），与 Megatron 的 `lm loss` 直接可比                                     |
| `learning/num_zeros` | int   | **新增**。梯度中零值参数数量，用于检测梯度稀疏化或死神经元。**仅在 `optimizer_memory_host_offload=False` 时可用**                 |
| `learning/is_nan`    | 0/1   | **新增**。loss 是否为 NaN                                                                                                        |
| `learning/is_inf`    | 0/1   | **新增**。loss 是否为 Inf                                                                                                        |

**保持不变的指标：**

| 指标               | 说明                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `learning/loss`    | **语义不变**，仍为混合 loss（LM loss + aux losses），与 upstream 行为一致。值 = `lm_loss + moe_lb_loss + mtp_loss` |

`lm_loss` 的计算在 `train_step` 中有两条路径：

- **GA 模式**（`gradient_accumulation_steps > 1`）：`lm_loss = loss`，直接使用 GA 函数返回的纯 LM loss（第 1 节中已保证返回值不含 aux losses）
- **非 GA 模式**：`lm_loss = aux["total_loss"] / (aux["total_weights"] + EPS)`，从 aux dict 中的交叉熵原始值计算

`learning/loss` 由 `lm_loss` 加上 aux dict 中的各辅助损失组合而成，保持 upstream 的混合 loss 语义。

### 7. `train.py` — eval batch 显式 device placement

**变更：** 在 eval 循环中添加：

```python
eval_batch = jax.device_put(eval_batch, data_loader.input_data_shardings)
```

确保 eval 数据正确分片到设备上。`jax.device_put` 对已正确放置的数据是幂等操作（no-op），不会引起额外开销或改变已正确分片的数据。

### 8. `scripts/pretrain_ling2.sh` — 训练启动脚本迁移

**来源：** ant-pretrain 的 `scripts/pretrain_al_model.sh`，迁移到 primatrix/maxtext 并适配新的模块路径和配置结构。

**迁移改动：**

| 项目            | ant-pretrain                   | primatrix/maxtext                             |
| --------------- | ------------------------------ | --------------------------------------------- |
| 脚本路径        | `scripts/pretrain_al_model.sh` | `scripts/pretrain_ling2.sh`                   |
| 训练入口        | `python3 -m MaxText.train`     | `python3 -m maxtext.trainers.pre_train.train` |
| 配置文件路径    | `src/MaxText/configs/base.yml` | `src/maxtext/configs/base.yml`                |
| 模型名称        | `model_name=al_model`          | `model_name=ling2`                            |
| 模型配置        | `configs/models/al_model.yml`  | `configs/models/ling2.yml`（PR1 已迁移）      |
| Argus dump 参数 | 包含 `argus_dump_*` 系列参数   | 移除（Argus 独立推送）                        |

**脚本结构：** 保持 ant-pretrain 的分段式结构，分为以下配置区块：

```bash
#!/bin/bash
# Ling2 Pretraining Script
# Architecture: Ling2 (Hybrid MLA/GLA + MoE)
# Dataset: Megatron MMap indexed datasets (.bin/.idx)

set -e

# ============================================================================
# 1. Tokenizer Config
# ============================================================================
VOCAB_SIZE=157184
PAD_ID=156892   # <|endoftext|>
BOS_ID=156891   # <|startoftext|>

# ============================================================================
# 2. Environment Config (GCS Bucket & Run Name)
# ============================================================================
BASE_OUTPUT_DIR=${GCS_BUCKET:-"gs://ant-pretrain/pretrain/dev"}
RUN_NAME=${RUN_NAME:-"ling2-pretrain-$(date +%Y%m%d_%H%M)"}

# ============================================================================
# 3. JAX Multi-node Config (Auto-detect)
# ============================================================================
if [[ -n "$TPU_PROCESS_ADDRESSES" ]]; then
    export JAX_COORDINATOR_ADDRESS=$(echo $TPU_PROCESS_ADDRESSES | cut -d',' -f1)
fi

# ============================================================================
# 4. Dataset Config (Megatron MMap Indexed)
# ============================================================================
DATASET_TYPE="grain"
GRAIN_FILE_TYPE="mmap_npy"
# ... 数据集路径配置（环境相关，用户按需修改）

# ============================================================================
# 5. Training Hyperparameters
# ============================================================================
MODEL_NAME="ling2"
CONFIG_FILE="src/maxtext/configs/base.yml"  # 适配 maxtext 路径
STEPS=${STEPS:-100000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
# ... 优化器、学习率、MoE 参数

# ============================================================================
# 6. Start Training
# ============================================================================
python3 -m maxtext.trainers.pre_train.train "$CONFIG_FILE" \
    model_name=$MODEL_NAME \
    override_model_config=true \
    run_name=$RUN_NAME \
    base_output_directory=$BASE_OUTPUT_DIR \
    \
    dataset_type=$DATASET_TYPE \
    grain_file_type=$GRAIN_FILE_TYPE \
    # ... 完整参数列表
    "$@"
```

**关键设计原则：**

- **环境变量覆盖**：所有关键参数支持 `${VAR:-default}` 模式，用户可通过环境变量覆盖而无需修改脚本
- **CLI 透传**：脚本末尾 `"$@"` 允许在命令行追加任意参数覆盖脚本默认值
- **数据集路径外置**：数据集路径是环境相关的（TPU Pod 本地路径），脚本提供默认值但使用者需按环境修改
- **不含 Argus dump**：Argus dump 参数从脚本中移除，待 Argus 代码独立推送后再添加

**与 ant-pretrain 的差异（不迁移的部分）：**

- `enable_routed_bias_grad=false`、`bias_zero_mean_update=true`：MoE routing bias 参数，由 PR2 引入后再加入脚本
- `mtp_loss_scaling_factor=0.1`：MTP 参数，由 PR7 引入后再加入脚本
- `moe_z_loss_coeff`：MoE z-loss 参数，由 PR2 引入后再加入脚本
- 上述参数在 PR2/PR7 合入后需同步更新脚本

## 不在本 PR 范围内的 train.py 改动

以下改动出现在 ant-pretrain 的 `train.py` 中，但**不属于 PR10 的范围**，由对应 PR 负责：

| 改动                                                               | 归属         | 说明                                                                            |
| ------------------------------------------------------------------ | ------------ | ------------------------------------------------------------------------------- |
| `_count_moe_layers`、`_collect_moe_intermediate_sum/mean` 辅助函数 | PR2          | MoE loss 跨层收集，替代 upstream 的 `possible_keys` 搜索                        |
| MoE z-loss 收集与加入 loss                                         | PR2          | `moe_z_loss` 在 `loss_fn` 中的计算和累加                                        |
| MoE expert counts 收集与 bias 更新重构                             | PR2          | `expert_counts_to_bias_update()`、`enable_routed_bias_grad`、`zero_mean_update` |
| Router 监控指标（`router_bias_mean`、`router_probs_std` 等）       | PR2          | MoE router 统计指标                                                             |
| `raw_mtp_loss` 追踪（scaled + unscaled 分离）                      | PR7          | MTP loss 的双指标上报                                                           |
| MTP expert counts 收集与 bias 更新                                 | PR7          | MTP block 中的 MoE 层 bias 更新                                                 |
| Argus dump 集成（`Dumper`、`dump_enabled`、`raw_grads` 返回）      | 上游独立目录 | 调试工具链，随 Argus dump 代码统一推回                                          |

## 后续 PR 需补充的 GA 改动（备忘）

以下改动已在 ant-pretrain 的 `gradient_accumulation.py` 中实现，但与具体功能耦合，将随对应 PR 一起推送到 primatrix/maxtext。**这些改动不包含在 PR10 中**，后续 PR 提交时必须同步包含，否则 GA>1 场景下对应功能的指标和 expert counts 将不正确：

| 改动                                | 归属 PR | 文件                       | 说明                                                            |
| ----------------------------------- | ------- | -------------------------- | --------------------------------------------------------------- |
| `moe_z_loss` init + 累加 + 平均化   | PR2     | `gradient_accumulation.py` | MoE z-loss 在 GA 中的累加与归一化                               |
| `moe_expert_counts` 平均化          | PR2     | `gradient_accumulation.py` | expert counts 在 GA 中归一化，防止 bias update rate 被放大 K 倍 |
| `raw_mtp_loss` init + 累加 + 平均化 | PR7     | `gradient_accumulation.py` | 未缩放 MTP loss 的独立上报                                      |
| `mtp_expert_counts` 平均化          | PR7     | `gradient_accumulation.py` | MTP block 的 expert counts 在 GA 中归一化                       |

## 向后兼容性

| 保证                                                 | 机制                                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `calculate_per_token_loss=True` 保留现有 GA 行为     | 默认值为 `True`，梯度归一化方式不变                                                             |
| `use_tunix_gradient_accumulation` 路径完整保留        | GA 条件使用三条件合取，Tunix GA 路径行为不变                                                    |
| CE z_loss 逻辑完整保留                               | `loss_fn` 中保持 `z_loss=config.z_loss_multiplier` 传递，`vocabulary_tiling.py` 返回值不变      |
| `indexer_loss` 不误删                                | 本 PR 不引入 indexer 逻辑，但推送时需确保与 upstream 已有的 indexer 代码并存                    |
| `learning/loss` 语义不变                             | 仍为混合 loss（LM + aux），与 upstream 保持一致                                                 |
| 新增监控指标不影响性能                               | `num_zeros`、`is_nan`、`is_inf` 仅在 metrics 汇报时计算                                         |
| GA aux metrics 平均化是 bug fix                      | GA>1 用户受益（日志指标更准确），GA=1 用户无影响                                                |
| 默认配置下梯度不变                                   | 当 `calculate_per_token_loss=True` 且 `z_loss_multiplier=0`（默认值）时，梯度值与 upstream 一致 |

## 测试方案

### 1. 现有测试套件通过

```bash
python3 -m pytest -m "cpu_only"
```

确保 `calculate_per_token_loss=True` 默认行为不变，所有现有测试不受影响。

### 2. GA 双模式数值验证（loss + 梯度）

新增测试用例，构造一个小模型 + 合成数据，分别验证两种模式下的 loss **和梯度**一致性。

**关键：测试数据需构造 micro-batch 间有效 token 数不均匀的场景**（如 micro-batch 0 有 50 个有效 token，micro-batch 1 有 20 个），否则两种模式计算结果相同，测试无法验证差异。

```python
# 构造不均匀 padding 数据：不同 micro-batch 的有效 token 数差异显著
batch_4x = make_uneven_batch(
    microbatch_sizes=[50, 20, 35, 15],  # 每个 micro-batch 的有效 token 数
    seq_len=64,
    vocab_size=1024,
)

# 验证 1：calculate_per_token_loss=True（默认模式）
# GA=1 的 loss 和梯度应与 GA=4 数值一致（全局 per-token 平均）
config_ga1 = make_config(gradient_accumulation_steps=1, calculate_per_token_loss=True)
config_ga4 = make_config(gradient_accumulation_steps=4, calculate_per_token_loss=True)
loss_ga1, _, grads_ga1 = train_step(config_ga1, batch_4x)
loss_ga4, _, grads_ga4 = train_step(config_ga4, batch_4x)
assert jnp.allclose(loss_ga1, loss_ga4, atol=1e-5)
chex.assert_trees_all_close(grads_ga1, grads_ga4, atol=1e-5)

# 验证 2：calculate_per_token_loss=False（Megatron 模式）
# loss 应等于各 micro-batch 独立 per-token 平均后的等权均值
config_ga4_meg = make_config(
    gradient_accumulation_steps=4, calculate_per_token_loss=False
)
loss_ga4_meg, _, grads_ga4_meg = train_step(config_ga4_meg, batch_4x)
expected_loss = mean([total_loss_i / total_weights_i for i in range(4)])
assert jnp.allclose(loss_ga4_meg, expected_loss, atol=1e-5)

# 验证 3：两种模式在不均匀 padding 下 loss 确实不同
# 确认测试数据有效——如果两个值相等，说明测试数据没有区分度
assert not jnp.allclose(loss_ga4, loss_ga4_meg, atol=1e-3), (
    "测试数据需保证两种模式产生不同的 loss 值"
)
```

### 3. 向后兼容性：默认配置梯度不变

验证默认配置（`calculate_per_token_loss=True`、`z_loss_multiplier=0`）下，变更前后梯度值完全一致：

```python
# 使用默认配置（upstream 行为），对比变更前后
config_default = make_config(
    gradient_accumulation_steps=4,
    calculate_per_token_loss=True,
    z_loss_multiplier=0,
)
# upstream_grads: 用 upstream gradient_accumulation.py 计算（或预存 reference）
# new_grads: 用变更后代码计算
loss_new, _, grads_new = train_step_new(config_default, batch)
loss_ref, _, grads_ref = train_step_upstream(config_default, batch)
assert jnp.allclose(loss_new, loss_ref, atol=1e-6)
chex.assert_trees_all_close(grads_new, grads_ref, atol=1e-6)
```

### 4. aux losses × total_weights 补偿验证

验证 GA 路径下 aux losses（`moe_lb_loss`、`mtp_loss`）对梯度的贡献量级与非 GA 路径一致。**此测试覆盖变更 4 的核心逻辑——`loss_fn` 中 `moe_lb_loss * total_weights` 补偿的正确性**。

```python
# GA=1（非 GA 路径）和 GA=4（GA 路径）应产生量级一致的 aux loss 梯度贡献
config_ga1 = make_config(gradient_accumulation_steps=1, calculate_per_token_loss=True)
config_ga4 = make_config(gradient_accumulation_steps=4, calculate_per_token_loss=True)

# 方法：构造一个 moe_lb_loss 较大的场景（如 moe_lb_coeff=1.0），
# 对比 GA=1 和 GA=4 的梯度差异
loss_ga1, aux_ga1, grads_ga1 = train_step(config_ga1, batch_4x)
loss_ga4, aux_ga4, grads_ga4 = train_step(config_ga4, batch_4x)
# aux losses 对梯度的贡献应一致（atol 允许浮点累加误差）
chex.assert_trees_all_close(grads_ga1, grads_ga4, atol=1e-4)

# 负面测试：如果去掉 * total_weights 补偿，GA=4 的 aux loss 梯度贡献
# 会被缩小 ~total_weights 倍，梯度差异将非常大
# （此验证在代码 review 中确认，不需要运行时测试）
```

### 5. aux metrics 平均化验证

验证 GA>1 时 aux dict 中的 `moe_lb_loss` 与 GA=1 时的值一致（而非被放大 `gradient_accumulation_steps` 倍）：

```python
config_ga1 = make_config(gradient_accumulation_steps=1)
config_ga4 = make_config(gradient_accumulation_steps=4)
_, aux_ga1, _ = train_step(config_ga1, batch)
_, aux_ga4, _ = train_step(config_ga4, batch_4x)
# Bug fix 验证：GA=4 的 moe_lb_loss 应与 GA=1 的 moe_lb_loss 量级一致
assert jnp.allclose(aux_ga1["moe_lb_loss"], aux_ga4["moe_lb_loss"], rtol=0.01)
# 同理验证 mtp_loss
assert jnp.allclose(aux_ga1["mtp_loss"], aux_ga4["mtp_loss"], rtol=0.01)
```

### 6. 监控指标验证

```python
# === is_nan / is_inf ===
metrics_normal = train_step(config, normal_batch)
assert metrics_normal["is_nan"] == 0
assert metrics_normal["is_inf"] == 0

# 构造 NaN 场景验证检测生效（可选：用极端权重触发溢出）
# metrics_nan = train_step(config, nan_inducing_batch)
# assert metrics_nan["is_nan"] == 1

# === num_zeros ===
# 验证与手动计算一致
num_zeros_manual = sum(jnp.sum(p == 0) for p in jax.tree.leaves(grads))
assert metrics["num_zeros"] == num_zeros_manual

# === lm_loss 纯 LM loss 验证 ===
# GA 路径：lm_loss 直接等于 GA 返回值
config_ga = make_config(gradient_accumulation_steps=4)
metrics_ga = train_step(config_ga, batch_4x)
# lm_loss 不应包含 moe_lb_loss、mtp_loss 等辅助损失
# 验证方法：lm_loss 应与手动从 aux 中计算的纯 CE loss 一致
lm_loss_manual = aux["total_loss"] / aux["total_weights"]
assert jnp.allclose(metrics_ga["lm_loss"], lm_loss_manual, atol=1e-5)

# 非 GA 路径：lm_loss 从 aux 中计算
config_no_ga = make_config(gradient_accumulation_steps=1)
metrics_no_ga = train_step(config_no_ga, batch)
lm_loss_no_ga = aux["total_loss"] / aux["total_weights"]
assert jnp.allclose(metrics_no_ga["lm_loss"], lm_loss_no_ga, atol=1e-5)

# === learning/loss 保持混合 loss 语义 ===
# learning/loss = lm_loss + moe_lb_loss + mtp_loss（与 upstream 一致）
expected_mixed_loss = (
    metrics_ga["lm_loss"] + aux["moe_lb_loss"] + aux["mtp_loss"]
)
assert jnp.allclose(metrics_ga["loss"], expected_mixed_loss, atol=1e-5)
```

### 7. 非 GA 路径覆盖（GA=1）

验证 GA=1 时变更 4 和变更 6 的 `else` 分支行为正确：

```python
config_no_ga = make_config(gradient_accumulation_steps=1, calculate_per_token_loss=True)
config_no_ga_meg = make_config(
    gradient_accumulation_steps=1, calculate_per_token_loss=False
)

# GA=1 时 calculate_per_token_loss 不影响结果——
# 两种模式都走 else 分支: loss = total_loss / (total_weights + EPS)
loss_default, _, _ = train_step(config_no_ga, batch)
loss_meg, _, _ = train_step(config_no_ga_meg, batch)
assert jnp.allclose(loss_default, loss_meg, atol=1e-6), (
    "GA=1 时两种模式的 loss 应相同（均为 per-token 平均）"
)

# aux losses 在 GA=1 的 else 分支中直接相加（无 * total_weights）
# 验证 moe_lb_loss 和 mtp_loss 对梯度的贡献正确
# （与 GA>1 + calculate_per_token_loss=True 的结果一致即可）
```

### 8. eval batch device placement 验证

验证变更 7（eval batch 显式 device placement）不影响 eval 结果：

```python
# 验证 jax.device_put 是幂等的——已正确分片的数据 put 后值不变
eval_batch = make_eval_batch(config)
placed_batch = jax.device_put(eval_batch, data_loader.input_data_shardings)
chex.assert_trees_all_equal(eval_batch, placed_batch)

# 验证 eval loss 在添加 device_put 前后一致
eval_loss_before = eval_step(config, eval_batch)
eval_loss_after = eval_step(config, placed_batch)
assert jnp.allclose(eval_loss_before, eval_loss_after, atol=1e-6)
```

**注意：** 此测试主要验证幂等性。`device_put` 的实际作用在多 host 场景下才体现（eval 数据可能未正确分片到设备），单机 CPU 测试中为 no-op。完整验证需在 TPU Pod 上运行 eval。

### 9. 训练脚本验证

验证变更 8（`pretrain_ling2.sh`）的正确性：

```bash
# 9a. 脚本语法检查
bash -n scripts/pretrain_ling2.sh

# 9b. 引用路径存在性验证
test -f src/maxtext/configs/base.yml
test -f src/maxtext/configs/models/ling2.yml
python3 -c "import maxtext.trainers.pre_train.train"  # 模块可导入

# 9c. dry-run 验证（可选，需 TPU 环境）
# 用极小的 steps 和 batch 验证脚本能成功启动训练
STEPS=2 PER_DEVICE_BATCH_SIZE=1 bash scripts/pretrain_ling2.sh
```

### 10. CE z_loss 保持不变

无需额外测试——保留 upstream 原有逻辑，现有测试套件已覆盖。

### 测试覆盖矩阵

| 变更                           | 测试编号 | 验证内容                                     |
| ------------------------------ | -------- | -------------------------------------------- |
| 1. GA 双模式归一化             | 2, 3     | loss 双模式 + 梯度一致性 + 不均匀 padding    |
| 2. aux metrics 平均化 bug fix  | 5        | GA>1 vs GA=1 aux 值一致                      |
| 3. lm_loss_per_token 累加器    | 2        | Megatron 模式 loss 数值正确                  |
| 4. loss_fn aux losses 补偿     | 4        | GA 路径下 aux loss 梯度量级正确              |
| 5. CE z_loss 保留              | 10       | upstream 测试套件已覆盖                      |
| 6. train_step 监控指标         | 6, 7     | is_nan/is_inf/num_zeros/lm_loss + 非 GA 路径 |
| 7. eval batch device placement | 8        | 幂等性 + eval loss 不变                      |
| 8. pretrain_ling2.sh 脚本      | 9        | 语法 + 路径 + 可选 dry-run                   |
| 向后兼容性                     | 1, 3     | 默认配置梯度不变 + 现有测试通过              |

## Megatron 参数映射

| MaxText 字段/指标                | Megatron 对应                   | 说明                                        |
| -------------------------------- | ------------------------------- | ------------------------------------------- |
| `calculate_per_token_loss=False` | `--no-calculate-per-token-loss` | 布尔语义相反                                |
| `learning/lm_loss`               | `lm loss`                       | 纯 LM 交叉熵 per-token loss                 |
| GA aux metrics 平均化            | Megatron 无此 bug               | Megatron 的 GA 实现天然正确处理 aux metrics |

## 后续 PR 需补充的 MTP Loss 计算改动（备忘）

以下改动已在 ant-pretrain 的 `multi_token_prediction.py` 和 `train.py` 中实现，与 MTP 功能耦合，将随 PR7 一起推送。**这些改动不包含在 PR10 中**，但直接影响 MTP loss 数值，是精度对齐的关键点：

### 1. MTP per-layer 独立归一化

与 PR10 变更 3（`lm_loss_per_token` 累加器）本质相同——"先除再平均" vs "先累加再除"的问题，但发生在 MTP 层维度。

**upstream（`calculate_mtp_loss`）：** 所有 MTP 层的 losses 和 weights 先池化，再统一除一次：

```python
avg_mtp_loss = jnp.sum(mtp_losses_array) / (jnp.sum(mtp_weights_array) + EPS)
```

**ant-pretrain：** 每个 MTP 层独立算 per-token 平均，再取等权均值：

```python
per_layer_avg_losses = mtp_losses_array / (mtp_weights_array + EPS)
avg_mtp_loss = jnp.mean(per_layer_avg_losses)
```

当各 MTP 层有效 token 数不同时（packed sequence 中不同层 mask 不同），两种方式会产生数值差异。ant-pretrain 的做法与 Megatron 一致——每层等权贡献，不因有效 token 数少而被"稀释"。

### 2. `calculate_mtp_loss` 返回值元组化

**upstream：** 返回单个标量 `scaled_mtp_loss`。

**ant-pretrain：** 返回 `(scaled_mtp_loss, raw_mtp_loss)` 元组，其中 `raw_mtp_loss` 是未乘 `mtp_loss_scaling_factor` 的原始值。`loss_fn` 中解包为 `mtp_loss, raw_mtp_loss = calculate_mtp_loss(...)`，`raw_mtp_loss` 通过 aux dict 传递并在日志中上报，用于与 Megatron 的 `mtp_1 loss` 直接对比。

### 3. Segment-aware MTP rolling

**upstream：** `roll_and_mask(rolled_input_ids)` 不考虑文档边界，在 packed sequence 中会将一个文档的 token 滚动到相邻文档中。

**ant-pretrain：** `roll_and_mask_by_segment(rolled_input_ids, rolled_segment_ids)` 在滚动时检查 segment 边界，跨文档边界的位置会被 mask 为 0，避免无意义的跨文档 token 预测。

此改动影响 MTP loss 的正确性——跨文档边界的 token 预测是无意义的噪声，会干扰 MTP loss 的精度对齐。
