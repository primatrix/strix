---
title: 实现差距：MLA 与 DeepSeek V3
---

# 实现差距：MLA 与 DeepSeek V3

> tpu-inference 已完整实现 MLA 和 DeepSeek V3 模型，sglang-jax 仅有基础设施脚手架。本文详细分析两者差距并给出实现路线。

---

## 一、差距总览

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| `DeepseekV3ForCausalLM` 模型 | ✅ 完整实现 | ❌ 仅 `model_config.py` 中有架构名注册 |
| `DeepseekV3MLA` 注意力层 | ✅ Absorbed MLA + Weight Absorption | ❌ 无 |
| `MLAEinsum` 权重拆分 | ✅ 加载时拆分 `kv_b_proj` → `k_up_proj` + `v_up_proj` | ❌ 无 |
| MLA Pallas Kernel (V2) | ✅ 融合 KV Cache 更新 + FP8 Packing | ❌ 无 MLA kernel |
| MLA KV Cache 池 | ✅ 压缩存储 640 维/token | ✅ `MLATokenToKVPool` 已实现 |
| `AttentionArch.MLA` 枚举 | — | ✅ 已定义但未使用 |
| YARN RoPE + mscale | ✅ `DeepseekScalingRotaryEmbedding` | ❌ 仅有 YaRN 辅助函数 |
| MoE 256 专家 + Shared Expert | ✅ `SharedFusedMoe` | ❌ 有 MoE 但无 DeepSeek 专用配置 |
| Kimi K2.5 模型 | ✅ 通过 DeepSeek V3 路径运行 | ❌ 仅 multimodal config 注册 |

---

## 二、MLA 实现原理（tpu-inference 方案）

### 2.1 Absorbed MLA 投影流程

MLA 的核心思想是**低秩压缩 KV Cache**，通过 Weight Absorption 避免运行时解压缩：

```text
Query 侧:
  x (D=7168)
  → q_a_proj (7168 → 1536)              # 低秩压缩
  → RMSNorm
  → q_b_proj (1536 → 128×192)           # 展开到 per-head
  → split: q_nope (128d) + q_rope (64d)
  → RoPE(q_rope)
  → k_up_proj(q_nope): TNH → TNA        # Weight Absorption

KV 侧:
  x (D=7168)
  → kv_a_proj_with_mqa (7168 → 576)     # MQA 式单投影
  → split: kv_latent (512d) + k_rope (64d)
  → RMSNorm(kv_latent) + RoPE(k_rope)
  → 直接缓存压缩表示 (~640 维)

注意力计算:
  score = einsum(q_TNA, k_SA) + einsum(q_rope, k_rope)  # 双路径
  output = softmax(score) @ kv_latent                     # Value 复用 latent

输出:
  v_up_proj(output): TNA → TNH           # 从 latent 投影回 head 维度
```

### 2.2 Weight Absorption 机制 (MLAEinsum)

`MLAEinsum` 在权重加载时执行：

1. 加载融合的 `kv_b_proj` 权重 `(512, 128×256)`
2. Reshape 为 `(A=512, N=128, 256)`
3. 拆分为 `k_ANH`（nope 维度 128）和 `v_ANH`（v 维度 128）
4. 创建两个 Einsum 层：
   - **`k_up_proj`**: `TNH,ANH→TNA` — 吸收 nope 权重到 query（注意力前）
   - **`v_up_proj`**: `TNA,ANH→TNH` — 投影注意力输出回 head 维度（注意力后）

**关键文件**: `tpu_inference/models/jax/deepseek_v3.py:478` (`MLAEinsum`)

### 2.3 KV Cache 格式

| 项目 | 标准 MHA | MLA |
|---|---|---|
| 每 token 存储 | 128 heads × 128 dim × 2 = 32768 | 512 + 128 = 640 |
| 压缩率 | 1× | ~51× |
| KV Head 数 | 128 | 1（在 runner 中设为 1） |
| 分片策略 | 按页 + KV Head 分片 | 仅按页维度分片 |

Cache 形状：

```python
(total_num_pages,
 align_to(page_size, kv_packing) // kv_packing,
 kv_packing,
 align_to(640, 128))
```

### 2.4 MLA Pallas Kernel (V2 生产版本)

V2 Kernel 的核心优势：

- **融合 KV Cache 更新到 Kernel 内部**：新 KV token 通过 DMA 拉取并在 VMEM 中 pack，避免额外 HBM 读写
- **分离 Kernel Launch**：Decode (`static_q_len=1`) 和 Mixed 模式各自独立调优 block size
- **FP8 Packing** (`_pack_new_kv()`): 处理 sub-word 类型的位对齐
- **双缓冲 DMA Pipeline**: BKV、BQ、BO 均使用双缓冲，4 类 Semaphore × 2 缓冲区

核心计算逻辑：

```python
# Kernel 内部
s_nope = einsum("nd,md->nm", ql_nope, kv_c)    # Nope 注意力分数
s_pe   = einsum("nd,md->nm", q_pe, k_pe)       # RoPE 注意力分数
s = (s_nope + s_pe) * sm_scale
# FlashAttention-2 式 online softmax
pv = einsum("nm,md->nd", p, kv_c)              # kv_c 同时作为 value
```

**关键文件**: `tpu_inference/kernels/mla/v2/kernel.py`

---

## 三、sglang-jax 现有基础设施

### 3.1 已有组件

1. **`AttentionArch.MLA` 枚举** (`model_config.py:23-25`)：已定义但 `attention_arch` 始终设为 `MHA`

2. **`MLATokenToKVPool`** (`memory_pool.py:799+`)：
   - 参数：`kv_lora_rank`, `qk_rope_head_dim`
   - 存储形状：`[cache_size, 1, kv_lora_rank + qk_rope_head_dim]`
   - 无分片（`P(None, None, None)`）— MLA latent 足够小
   - 提供 `set_mla_kv_buffer()` 方法分别写入 nope/rope 组件

3. **`use_mla_backend` 标志** (`model_runner.py:92`)：检查 `attention_arch == AttentionArch.MLA`

4. **YaRN 辅助函数** (`embeddings.py`)：`_yarn_get_mscale()`, `_yarn_find_correction_dim()`, `_yarn_find_correction_range()` — 但无完整的 `DeepseekScalingRotaryEmbedding` 类

### 3.2 缺失组件

1. **DeepSeek V3 模型类**：无 `DeepseekV3ForCausalLM` 前向传播实现
2. **MLA 注意力层**：无 `DeepseekV3MLA` 或等价类
3. **MLAEinsum / Weight Absorption**：无权重拆分和吸收逻辑
4. **MLA Pallas Kernel**：无 MLA 专用注意力 kernel
5. **DeepseekScalingRotaryEmbedding**：无完整 YARN RoPE 类
6. **MLA 注意力后端集成**：`FlashAttention` 后端无 MLA 路径
7. **DeepSeek V3 Router / Shared Expert**：无 `DeepSeekV3Router`, `SharedFusedMoe`

---

## 四、实现路线

### Phase 1: YARN RoPE + mscale

**工作量**: ~1 天

在 `srt/layers/embeddings.py` 中实现 `DeepseekScalingRotaryEmbedding`：

```python
class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """YaRN with mscale for DeepSeek V3"""
    # 参数: scaling_factor, beta_fast, beta_slow, mscale, mscale_all_dim
    # 1. _yarn_find_correction_range() 计算校正范围
    # 2. 线性混合插值频率和外推频率
    # 3. mscale 校正 sin/cos
    # 4. 使用 interleaved (even/odd) 配对而非 split-half
```

**注意**：需在 `get_rope()` 工厂函数中添加 `scaling_type == "deepseek"` 分支。

已有辅助函数 `_yarn_get_mscale()`, `_yarn_find_correction_range()` 可直接复用。

### Phase 2: MLA 注意力层

**工作量**: ~3-5 天

1. 创建 `srt/models/deepseek_v3.py`，实现 `DeepseekV3MLA` 注意力模块：

   - Query 侧：`q_a_proj` → `q_a_layernorm` → `q_b_proj` → split → RoPE → `k_up_proj`
   - KV 侧：`kv_a_proj_with_mqa` → split → `kv_a_layernorm` → RoPE
   - 实现 `MLAEinsum` 权重吸收（加载时拆分 `kv_b_proj`）

2. softmax scale 需包含 mscale²：

   ```python
   yarn_mscale = 0.1 * mscale_all_dim * log(scaling_factor) + 1.0
   scale = qk_head_dim ** (-0.5) * yarn_mscale ** 2
   ```

### Phase 3: MLA Pallas Kernel

**工作量**: ~5-7 天

可分两步：

**Step 1 — 纯 JAX 参考实现**：

- 在 `NativeAttention` 后端中添加 MLA 路径
- 双路径分数计算（nope + rope）
- 验证数值正确性

**Step 2 — Pallas Kernel**：

- 参考 tpu-inference V2 kernel，在 `srt/kernels/` 下创建 `mla_attention/` 目录
- 关键设计决策：
  - 融合 KV Cache 更新 vs 外部更新（V2 融合方案更优但实现更复杂）
  - Decode 和 Mixed 模式分离 launch
  - FP8 Packing 支持
  - 双缓冲 DMA pipeline

**与现有 ragged_paged_attention 的差异**：

| 维度 | 现有 RPA | MLA RPA |
|---|---|---|
| 输入 | Q, K, V 分离 | q_TNA, q_rope_TNH, k_SA, k_rope_SH |
| KV heads | 多个 | 1（压缩 latent） |
| 分数计算 | `QK^T` | `s_nope + s_pe`（双路径） |
| Value | 独立 V tensor | 复用 kv_c (latent) |
| 分片 | 沿 KV head 分片 | 沿 token 维度分片 |

### Phase 4: DeepSeek V3 完整模型

**工作量**: ~3-5 天（依赖 Phase 1-3）

1. **模型结构**：
   - 61 个 Transformer 层（层 0-2 Dense，层 3-60 MoE）
   - MLA 注意力（所有层）
   - 256 专家 MoE + 1 Shared Expert（层 3+）
   - 第 62 层 MTP（可跳过）

2. **DeepSeek V3 Router**：
   - `sigmoid` scoring（非 softmax）
   - `use_grouped_topk=True`
   - `routed_scaling_factor` 缩放

3. **Shared Expert**：
   - `SharedFusedMoe` 包装 MoE + 独立 Shared Expert
   - Shared Expert 与 Routed Experts 的输出相加

4. **权重加载**：
   - HuggingFace 权重映射
   - MLAEinsum 的 `kv_b_proj` 拆分
   - FP8 量化权重支持

5. **注册**：
   - `EntryClass` 注册 `DeepseekV3ForCausalLM`
   - `model_config.py` 中 `attention_arch = AttentionArch.MLA`

### Phase 5: Kimi K2.5

**工作量**: ~1 天（依赖 Phase 4）

Kimi K2.5 架构基于 DeepSeek V3，可通过同一 `DeepseekV3ForCausalLM` 路径运行，仅需调整配置参数。

---

## 五、关键设计决策

### 5.1 Weight Absorption 时机

**tpu-inference 方案**：权重加载时一次性完成 → 零运行时开销

**建议**：在 `weight_utils.py` 的权重加载流程中集成 MLAEinsum 逻辑，确保加载完成后 `k_up_proj` 和 `v_up_proj` 已分别创建。

### 5.2 MLA 注意力后端集成

**两种方案**：

- **方案 A**: 在 `FlashAttention` 后端内部添加 MLA 分支（检查 `use_mla`）
- **方案 B**: 创建独立的 `MLAAttention` 后端类

**建议**：方案 B，因为 MLA 的输入格式、分片策略、kernel 调用都与标准 RPA 差异很大。

### 5.3 KV Cache 分片

sglang-jax 的 `MLATokenToKVPool` 已使用 `P(None, None, None)` 无分片策略，这与 tpu-inference 的 `P(MLP_TENSOR)` 不同。

**建议**：保持无分片可作为初始实现，后续在 DP Attention 实现后再调整为按 token/page 维度分片。

---

## 六、参考文件

### tpu-inference 关键代码

| 文件 | 内容 |
|---|---|
| `models/jax/deepseek_v3.py` | 完整模型、MLA 层、MLAEinsum、Router |
| `kernels/mla/v2/kernel.py` | 生产 MLA Pallas Kernel |
| `layers/common/attention_interface.py` | `mla_attention()` 入口 |
| `runner/kv_cache.py` | MLA KV Cache 形状与分片 |
| `layers/jax/rope.py` | `DeepseekScalingRotaryEmbedding` |

### sglang-jax 待修改文件

| 文件 | 修改内容 |
|---|---|
| `srt/layers/embeddings.py` | 添加 `DeepseekScalingRotaryEmbedding` |
| `srt/models/deepseek_v3.py` | 新建：完整模型实现 |
| `srt/kernels/mla_attention/` | 新建：MLA Pallas Kernel |
| `srt/layers/attention/mla_backend.py` | 新建：MLA 注意力后端 |
| `srt/models/registry.py` | 自动发现新模型 |
| `srt/configs/model_config.py` | MLA 配置路径激活 |
| `srt/utils/weight_utils.py` | 权重加载时 MLAEinsum 拆分 |
