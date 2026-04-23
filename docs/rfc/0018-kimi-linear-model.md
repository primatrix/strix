---
title: "RFC-0018: Kimi-Linear 模型框架实现"
status: draft
author: zhengkezhou1
date: 2026-04-22
reviewers: []
---

# RFC-0018: Kimi-Linear 模型框架实现

## 概述

在 SGL-JAX 中实现 `kimi_linear.py` 模型框架文件，组装现有和新建组件，使引擎能加载并推理 `moonshotai/Kimi-Linear-48B-A3B-Instruct` 模型。

## 背景

Kimi-Linear-48B-A3B-Instruct 是 Moonshot AI 发布的 hybrid 架构模型（48B 总参数 / ~3B 活跃参数），结合了 MLA (Multi-Latent Attention) 和 KDA (Key-Delta Attention) 两种注意力机制，以及 MoE (Mixture of Experts) 稀疏 FFN。

SGL-JAX 仓库已有的可复用组件：

- `MLAAttention`（`layers/attention/mla.py`）— 支持 `q_lora_rank=None`
- `GateLogit` + `TopK` + `EPMoE`/`FusedEPMoE`（`layers/moe.py`）
- `RMSNorm`（`layers/layernorm.py`）

参考实现：<https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py>

## 方案

### 架构

```text
KimiLinearForCausalLM (EntryClass, 匹配 HF config architectures 字段)
├── KimiModel
│   ├── Embed (embed_tokens)
│   ├── KimiDecoderLayer[] × 27  (单类 if/else 模式)
│   │   ├── RMSNorm (input_layernorm)
│   │   ├── Attention:
│   │   │   if config.is_kda_layer(layer_idx):
│   │   │       KDAAttention (import from layers/attention/kda.py)
│   │   │   else:
│   │   │       MLAAttention (复用现有, q_lora_rank=None)
│   │   ├── RMSNorm (post_attention_layernorm)
│   │   └── FFN:
│   │       if layer_idx < first_k_dense_replace:
│   │           KimiMLP (dense, 仅 layer 0)
│   │       else:
│   │           GateLogit + TopK + EPMoE + shared_experts (MoE)
│   └── RMSNorm (final norm)
```

### 层分派模式

采用**单类 if/else** 模式，与仓库现有 `BailingMoEDecoderLayer`、vLLM 和 HuggingFace 官方 Kimi 实现一致。MLA 层和 KDA 层共享相同的 scaffolding（input_layernorm → attention → post_attention_layernorm → FFN → residual），仅 `self_attn` 组件不同，无需拆分为多个类。

### 组件复用与差异分析

对比 HuggingFace 官方实现（`KimiMoEGate`、`KimiSparseMoeBlock`）、SGLang（`sglang/srt/models/kimi_linear.py`）、vLLM（`vllm/model_executor/models/kimi_linear.py`）三方实现，确认 SGL-JAX 现有 MoE 组件的可复用性及差异。

**MoE 数据流逐步对比：**

1\. **Router 打分（`GateLogit`）**：HF 在 matmul 前将输入和权重 cast 到 FP32（`F.linear(x.float32, weight.float32)`），SGL-JAX 用原始 dtype 做 matmul，FP32 cast 在 `TopK` 中才发生（`gate.py:89`）。数值上有微小差异，功能无影响。

2\. **Expert 选择（`TopK._biased_grouped_topk`）**：代码路径完全一致——加 `e_score_correction_bias`（仅用于选择）→ 分组取 top-2 求和 → 选组 → mask 未选中组 → 从剩余 expert 选 topk → 用原始分数（不含 bias）作为权重。但 Kimi 的 `num_expert_group=1`、`topk_group=1`，grouped topk 实际退化为直接 top-8，分组逻辑不生效。唯一区别是 mask 值：HF 用 `0.0`，SGL-JAX 用 `-inf`。因为 sigmoid 输出范围为 `(0,1)`，两种 mask 值都低于任何实际分数，效果相同。

3\. **权重归一化与缩放**：HF 无条件应用 `routed_scaling_factor`；SGL-JAX 仅在 `renormalize=True` 时应用（`gate.py:105-108`）。但 Kimi 的 `routed_scaling_factor=1.0`（乘 1），无实际影响。HF 归一化分母加 `1e-20` 防除零，SGL-JAX 不加，但 Kimi `norm_topk_prob=false`（不归一化），该代码不执行。

4\. **Expert 计算（`EPMoE`）**：结构完全相同（SwiGLU：`act(w1(x)) * w3(x) → w2(...)`），仅命名不同（HF `w1/w3/w2` vs SGL-JAX `wi_0/wi_1/wo`），通过 `create_moe_weights_mapping` 的 `expert_type_names=("w1", "w3", "w2")` 处理。

5\. **Shared Expert**：三方一致，均为 SwiGLU MLP + 加法组合（`routed_output + shared_output`）。Dense MLP 和 shared experts 使用自定义的 `KimiMLP`。

**差异汇总：**

| 步骤 | 差异 | 对 Kimi 有无实际影响 |
|---|---|---|
| Router FP32 cast 时机 | HF matmul 前 cast；SGL-JAX matmul 后 cast | 无（微小数值差异） |
| Mask 值 | HF `0.0`；SGL-JAX `-inf` | 无（sigmoid 输出范围为 `(0,1)`，`0.0` 和 `-inf` 都低于任何实际分数，效果相同） |
| `routed_scaling_factor` 条件 | HF 无条件；SGL-JAX 仅 `renormalize=True` 时 | 无（Kimi 值为 `1.0`） |
| 归一化 epsilon | HF `+1e-20`；SGL-JAX 无 | 无（Kimi `norm_topk_prob=false`，不归一化） |
| `moe_layer_freq` 检查 | HF 检查 `layer_idx % freq == 0`；SGL-JAX 未检查 | 无（Kimi `freq=1`，全部命中） |
| Expert 权重命名 | `w1/w2/w3` vs `wi_0/wi_1/wo` | 无（WeightMapping 参数处理） |

结论：所有差异对 Kimi 模型均无实际影响。`GateLogit`、`TopK`、`EPMoE` 可直接复用。Dense MLP 和 shared experts 使用自定义的 `KimiMLP`（与 `BailingMoEMLP` 结构相同，均为 SwiGLU，但避免跨模型模块依赖）。

**整体结构对比：**

三方实现在 decoder 层结构上完全一致：pre-norm residual、`is_kda_layer` 分派 attention、`first_k_dense_replace` 分派 FFN。差异仅在基础设施层面：SGLang 复用 `DeepseekV2AttentionMLA` 并使用 `RadixLinearAttention` 管理 KDA 状态；vLLM 将 MLA 包装在 `MultiHeadLatentAttentionWrapper` 中，KDA 使用 Mamba 状态管理基础设施。SGL-JAX 的单类 if/else 模式与三方一致。

### WeightMapping

基于 safetensors index（20493 个 key，34 种 pattern）的完整映射：

**全局权重：**

| HF Key | SGL-JAX Target | Sharding |
|---|---|---|
| `model.embed_tokens.weight` | `model.embed_tokens.embedding` | `("tensor", None)` |
| `model.norm.weight` | `model.norm.scale` | `(None,)` |
| `lm_head.weight` | `lm_head.embedding` | `("tensor", None)` |

**每层公共权重（27 层）：**

| HF Key Pattern | SGL-JAX Target | Sharding |
|---|---|---|
| `layers.{L}.input_layernorm.weight` | `.input_layernorm.scale` | `(None,)` |
| `layers.{L}.post_attention_layernorm.weight` | `.post_attention_layernorm.scale` | `(None,)` |
| `layers.{L}.self_attn.q_proj.weight` | `.self_attn.q_proj.weight` | `(None, "tensor")` |
| `layers.{L}.self_attn.o_proj.weight` | `.self_attn.o_proj.weight` | `("tensor", None)` |

**MLA 层专属（7 层）：**

| HF Key Pattern | SGL-JAX Target |
|---|---|
| `layers.{L}.self_attn.kv_a_proj_with_mqa.weight` | `.self_attn.kv_a_proj.weight` |
| `layers.{L}.self_attn.kv_a_layernorm.weight` | `.self_attn.kv_a_layernorm.scale` |
| `layers.{L}.self_attn.kv_b_proj.weight` | `.self_attn.kv_b_proj.weight` |

注意：`q_lora_rank=null`，MLA 的 Q 路径仅有 `q_proj`（无 `q_a_proj`/`q_b_proj`）。`mla_use_nope=true` 与现有 `MLAAttention` 默认行为（nope+rope 分离）一致，无需额外处理。

**KDA 层专属（20 层，13 种参数）：**

| HF Key Pattern | 说明 |
|---|---|
| `layers.{L}.self_attn.{k,v}_proj.weight` | K/V 投影 |
| `layers.{L}.self_attn.{q,k,v}_conv1d.weight` | 短卷积权重 |
| `layers.{L}.self_attn.A_log` | 衰减系数（1D） |
| `layers.{L}.self_attn.f_{a,b}_proj.weight` | 衰减 gate 低秩投影 |
| `layers.{L}.self_attn.dt_bias` | gate 偏置（1D） |
| `layers.{L}.self_attn.b_proj.weight` | Beta gate |
| `layers.{L}.self_attn.g_{a,b}_proj.weight` | 输出 gate 低秩投影 |
| `layers.{L}.self_attn.o_norm.weight` | FusedRMSNormGated |

KDA 层的 SGL-JAX target path 取决于 KDA 模块实现方的属性命名，在 KDA 模块实现后确认。

**Dense MLP（仅 Layer 0）：**

| HF Key Pattern | SGL-JAX Target | Sharding |
|---|---|---|
| `layers.0.mlp.gate_proj.weight` | `.mlp.gate_proj.weight` | `(None, "tensor")` |
| `layers.0.mlp.up_proj.weight` | `.mlp.up_proj.weight` | `(None, "tensor")` |
| `layers.0.mlp.down_proj.weight` | `.mlp.down_proj.weight` | `("tensor", None)` |

**MoE（Layer 1-26，256 experts）：**

| HF Key Pattern | SGL-JAX Target | Sharding |
|---|---|---|
| `layers.{L}.block_sparse_moe.gate.weight` | `.moe_gate.kernel` | `(None, None)` |
| `layers.{L}.block_sparse_moe.gate.e_score_correction_bias` | `.moe_gate.bias` | `(None,)` |
| `layers.{L}.block_sparse_moe.experts.{E}.w1.weight` | `.block_sparse_moe.wi_0` | `("expert", None, "tensor")` |
| `layers.{L}.block_sparse_moe.experts.{E}.w3.weight` | `.block_sparse_moe.wi_1` | `("expert", None, "tensor")` |
| `layers.{L}.block_sparse_moe.experts.{E}.w2.weight` | `.block_sparse_moe.wo` | `("expert", "tensor", None)` |
| `layers.{L}.block_sparse_moe.shared_experts.gate_proj.weight` | `.shared_experts.gate_proj.weight` | `(None, "tensor")` |
| `layers.{L}.block_sparse_moe.shared_experts.up_proj.weight` | `.shared_experts.up_proj.weight` | `(None, "tensor")` |
| `layers.{L}.block_sparse_moe.shared_experts.down_proj.weight` | `.shared_experts.down_proj.weight` | `("tensor", None)` |

注意：256 个 expert 的权重经转置后按 expert 维度堆叠为 3D 张量，由 `create_moe_weights_mapping` 生成映射。

### 测试策略

- 模型实例化测试：传入 HF config，验证 27 层的层类型（MLA/KDA）和 FFN 类型（dense/MoE）
- WeightMapping 完整性测试：加载 safetensors index，断言每个 HF key 都有对应映射

### 备选方案

1. **继承 BailingMoEForCausalLM** — 拒绝：hybrid 架构（MLA+KDA 交替）与 BailingMoE 的单一 attention 类型有根本差异，继承关系带来不必要耦合
2. **先实现 KDA 再做模型框架** — 拒绝：可并行开发，框架定义集成点，KDA 模块独立实现
3. **多类字典分派** — 拒绝：MLA 和 KDA 层共享相同 scaffolding（norm、residual、FFN 分派），拆分为多类会导致代码重复；当前 hybrid 仅两种 attention 类型，单类 if/else 足够

## 影响范围

- 新增文件：`python/sgl_jax/srt/models/kimi_linear.py`
- 依赖文件：`layers/attention/mla.py`、`layers/moe.py`、`layers/layernorm.py`、`layers/embeddings.py`、`layers/attention/kda.py`
- 不修改现有文件

## 实施计划

1. 实现 `KimiDecoderLayer`、`KimiModel`、`KimiForCausalLM` 框架
2. 实现完整 WeightMapping（MLA 层 + dense MLP + MoE，KDA 层）
3. 注册 `EntryClass`
4. 编写实例化测试和 WeightMapping 完整性测试

## 风险

- KDA,MLA 模块实现进度可能影响端到端验证时间
- KDA 模块属性命名若与 HF 不一致，WeightMapping 需要额外调整

<!-- provenance
- "num_hidden_layers: 27, hidden_size: 2304" ← HF config.json
- "kda_layers 1-indexed, is_kda_layer uses (layer_idx + 1)" ← HF configuration_kimi.py is_kda_layer method
- "MLA layers [3,7,11,15,19,23,26]" ← derived from config full_attn_layers [4,8,12,16,20,24,27] minus 1
- "q_lora_rank=null" ← HF config.json
- "kv_a_proj_with_mqa naming" ← safetensors index weight_map
- "block_sparse_moe prefix" ← safetensors index weight_map
- "w1/w2/w3 expert naming" ← safetensors index weight_map
- "e_score_correction_bias" ← safetensors index weight_map
- "20493 total keys, 34 patterns" ← safetensors index analysis
- "BailingMoEDecoderLayer single class if/else pattern" ← models/bailing_moe.py line 206
- "MLAAttention supports q_lora_rank=None" ← layers/attention/mla.py lines 60-68
- "single class if/else used by vLLM Kimi, HF Kimi, vLLM DeepSeek-V2" ← QA round 3 research
- "SGLang Kimi MLA reuses DeepseekV2AttentionMLA" ← sglang/srt/models/kimi_linear.py
- "vLLM Kimi MLA uses MultiHeadLatentAttentionWrapper" ← vllm/model_executor/models/kimi_linear.py
- "SGLang/vLLM store compressed KV, SGL-JAX stores decompressed" ← cross-implementation comparison
- "routed_scaling_factor=1.0, moe_layer_freq=1, norm_topk_prob=false" ← HF config.json
- "score_func=sigmoid for Kimi MoE gate" ← HF modeling_kimi.py KimiMoEGate uses sigmoid
- "GateLogit FP32 cast in TopK not GateLogit" ← layers/gate.py line 89
- "routed_scaling_factor only applied when renormalize=True" ← layers/gate.py lines 105-108
- "LinearBase uses self.weight not self.kernel" ← layers/linear.py line 50
- "create_moe_weights_mapping moe_path used in both source and target" ← layers/moe.py lines 733-737
-->
