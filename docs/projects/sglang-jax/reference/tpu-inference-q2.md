---
title: 参考：tpu-inference Q2 对照
---

# 参考：tpu-inference 对 sglang-jax Q2 规划的支持情况

> 本文基于 [tpu-inference](https://github.com/google/tpu-inference) 仓库 `main` 分支（commit: `2f764005`）的源码分析，对照 sglang-jax Q2 规划逐项给出「已实现 / 未实现」判定及实现原理摘要，供团队开发参考。

---

## 一、总览

### 已实现

| 类别 | 项目 |
|---|---|
| 模型 | DeepSeek V3/R1/V3.2、GPT-OSS、Qwen 3.5、Kimi K2.5 |
| 目标 | DP Attention、PD Disaggregation、高性能投机采样 |
| 组件 | Hybrid Layer-Type System、MLA、YARN RoPE with mscale、Partial RoPE |
| 优化 | MoE + GMM Kernel、量化（FP8/Int8/MXFP4/AWQ）、通信-计算重叠、N:M 结构化稀疏 |

### 未实现

| 类别 | 项目 |
|---|---|
| 模型 | Ling 2.5、MiniMax M2.5、GLM 模型族、DeepSeek OCR-2 |
| 目标 | Multi-level KV Cache Buffer、KV Cache Unloading/Uploading、MTP |
| 组件 | Linear Attention Variants、Native Sparse Attention (NSA) |

---

## 二、模型支持

| Q2 计划模型 | 状态 | tpu-inference 实现方式 |
|---|---|---|
| **DeepSeek V3/R1/V3.2** | ✅ | `DeepseekV3ForCausalLM`，256 专家 MoE + MLA/MHA 双路径 + YaRN RoPE + FP8 量化 |
| **DeepSeek OCR-2** | ❌ | 文本模型已有，但 OCR-2 需要额外的视觉编码器，尚未实现 |
| **GPT-OSS** | ✅ | `GptOss`，全层 MoE + Sliding Window/Full Attention 交替 + Attention Sinks + MXFP4 量化 |
| **Qwen 3.5** | ✅ | Dense 变体走 `Qwen3ForCausalLM`（QK-Norm），MoE 变体走 `Qwen3MoeForCausalLM` |
| **Kimi K2.5** | ✅ | 架构基于 DeepSeek V3，通过 `DeepseekV3ForCausalLM` 路径运行，CI 中已有 pipeline |
| **Ling 2.5** | ❌ | 仓库中无任何相关代码 |
| **MiniMax M2.5** | ❌ | 仓库中无任何相关代码 |
| **GLM 模型族** | ❌ | 仓库中无任何相关代码 |

### 参考价值

- DeepSeek V3 是最复杂的模型实现（MLA + MoE + YaRN + FP8），可作为我们实现 DeepSeek 系列的完整参考
- GPT-OSS 的 Sliding Window + Attention Sinks + MXFP4 量化值得参考
- Qwen3 的 QK-Norm 实现简洁，是在 Qwen2 基础上的增量改动

---

## 三、核心目标

### 3.1 DP Attention ✅

> 针对 MoE 模型中 KV Head 数远小于 TP 度导致 KV Cache 复制浪费的问题

**实现原理**：

tpu-inference 设计了 **5D Mesh**，将 Attention 和 MLP/MoE 使用不同的分片策略：

```plaintext
Mesh 轴: ("data", "attn_dp", "attn_dp_expert", "expert", "model")

Attention 阶段:
  Batch 跨 (data × attn_dp × attn_dp_expert) 分片 → 每设备 batch 小，独立 KV heads
  KV heads 跨 (model × expert) 分片

MLP/MoE 阶段:
  Batch 仅跨 data 分片
  权重跨 (attn_dp × attn_dp_expert × model × expert) 分片 → 完整模型并行容量
```

自动计算 DP Degree：当 `TP > num_kv_heads_per_device` 时自动启用 DP Attention，将部分 TP 度转为 `attn_dp`。

多进程 DP Scheduler：每个 DP rank 一个独立 Scheduler 进程（避免 GIL），通过 Pipe 通信，支持 Prefix Cache 亲和性路由。

**关键文件**：

- `tpu_inference/layers/common/sharding.py` — 5D Mesh 定义、分片策略
- `tpu_inference/core/sched/dp_scheduler.py` — DP 调度器

**对 sglang-jax 参考**：5D Mesh 设计、Attention/MLP 分片解耦、自动 DP Degree 计算的逻辑可直接参考。多进程 Scheduler 架构与 sglang-jax 当前的单进程 Scheduler 差异较大，需评估是否引入。

---

### 3.2 PD Disaggregation ✅

> Prefill-Decode 分离部署

**实现原理**：

支持两种模式：

**Mode A — 进程内分离 (Local Disagg)**：

- 通过 `PREFILL_SLICES` / `DECODE_SLICES` 环境变量将 TPU 芯片分为两组
- 每组运行独立 vLLM Engine，通过 `jax.device_put` / `experimental_reshard` 跨 mesh 转移 KV Cache
- `_DisaggOrchestrator` 编排三类线程：Prefill → Transfer → Decode

**Mode B — 跨进程分离 (Multi-host Disagg)**：

- 独立 vLLM Server 进程 + Proxy Server
- **JAX P2P Transfer** (`jax.experimental.transfer.start_transfer_server`) 作为数据平面（Pull-based）
- **ZMQ Side Channel** 用于缓冲区释放通知
- Pallas DMA Kernel (`multi_layer_copy`) 实现异步 HBM-to-HBM 复制

**关键文件**：

- `tpu_inference/core/core_tpu.py` — `_DisaggOrchestrator`
- `tpu_inference/core/disagg_executor.py` — 设备分配
- `tpu_inference/distributed/tpu_connector.py` — P2P 传输 connector
- `tpu_inference/distributed/kv_transfer.py` — Pallas DMA 传输 Kernel

**对 sglang-jax 参考**：Local Disagg 模式适合单机多 chip 场景，实现较轻。Multi-host 模式的 JAX P2P Transfer + ZMQ 架构值得参考。`multi_layer_copy` 的层间管线化 DMA 是性能关键。

---

### 3.3 Multi-level KV Cache Buffer ❌

tpu-inference 仅使用 **单级 HBM Paged KV Cache**。无多级/分层/L2 KV Cache 缓冲。

---

### 3.4 KV Cache Unloading/Uploading ❌

无 HBM ↔ Host Memory 的 KV Cache 卸载/加载机制。`cpu_offload_gb` 配置仅用于模型权重卸载。

> 注：PD Disaggregation 中的 D2H 转移（`TPU_ENABLE_D2H_TRANSFER`）是为了跨机 KV 传输而非 offloading。

---

### 3.5 高性能投机采样 ✅

**实现原理**：

支持两种方法：

**EAGLE3**（模型驱动）：

- Target Model 在层 `(2, N//2, N-3)` 收集辅助隐藏状态
- 三个隐藏状态拼接 → `fc(3×H → H)` → 单层 Draft Model 自回归生成
- Draft Model 使用 Greedy Argmax（非采样），共享 Target 的 Embedding
- Rejection Sampler 支持 Greedy / Random 两种模式（含 Gumbel-Max trick）

**N-gram**（无模型）：

- 基于请求 token 历史的 N-gram 模式匹配

当前限制：

- 无 Tree Attention（平坦序列验证）
- 仅支持同步调度
- 仅 Llama3 系列支持 EAGLE3

**关键文件**：

- `tpu_inference/spec_decode/jax/eagle3.py` — EAGLE3 Proposer
- `tpu_inference/models/jax/llama_eagle3.py` — Draft Model 架构
- `tpu_inference/runner/speculative_decoding_manager.py` — 管理器

**对 sglang-jax 参考**：EAGLE3 的辅助隐藏状态收集 + 单层 Draft 架构与 sglang-jax 现有 Eagle 实现相似，可对比 Rejection Sampler 的实现。无 Tree Attention 是一个差距。

---

### 3.6 MTP (Multi-Token Prediction) ❌

DeepSeek V3 的第 62 层（MTP 层）在权重加载时被显式跳过：

```python
end_ignore_layer_num = 62  # last layer is MTP, we ignore it for now
```

无 MTP 专用 Proposer 或解码路径。

---

## 四、组件支持

### 4.1 Hybrid Layer-Type System ✅

**实现原理**：

通过多态 `TransformerBlock` 设计实现：

| 层类型 | 实现类 | 使用模型 |
|---|---|---|
| Dense FFW | `TransformerBlock(custom_module=DenseFFW)` | Qwen, Llama3 |
| MoE | `TransformerBlock(custom_module=JaxMoE)` | 全 MoE 模型 |
| MoE + Shared Expert | `SharedExpertsTransformerBlock` | DeepSeek V3, Llama4 |
| Dense/MoE 混合 | 按层索引切换 `custom_module` | DeepSeek V3 (前3层Dense), Llama4 Maverick |
| RoPE/NoPE 交替 | `use_attention_rope` per-layer 配置 | Llama4 (每4层一个 NoPE) |
| Sliding Window 交替 | `sliding_window` per-layer 配置 | GPT-OSS (奇偶交替) |

**关键文件**：

- `tpu_inference/layers/jax/transformer_block.py` — `TransformerBlock`, `SharedExpertsTransformerBlock`

**对 sglang-jax 参考**：`custom_module` 多态 + `SharedExpertsTransformerBlock` 是简洁的层级混合方案。

---

### 4.2 Multi-Head Latent Attention (MLA) ✅

**实现原理**：

Absorbed MLA 变体——通过 Weight Absorption 避免运行时解压缩，KV Cache ~51× 压缩：

```plaintext
Query 侧:
  x → q_a_proj(D→1536) → LayerNorm → q_b_proj(1536→N×192) → split(q_nope + q_rope)
  → RoPE(q_rope)
  → k_up_proj(q_nope): TNH→TNA    ← Weight Absorption

KV 侧:
  x → kv_a_proj_with_mqa(D→576) → split(kv_latent[512] + k_rope[64])
  → LayerNorm(kv_latent) + RoPE(k_rope)
  → 直接缓存压缩表示 (~640 维)    ← 无 per-head 展开

Kernel:
  score = einsum(q_absorbed, kv_latent) + einsum(q_rope, k_rope)
  output = softmax(score) @ kv_latent    ← Value 复用 latent

Output:
  v_up_proj(output): TNA→TNH    ← 从 latent 投影回 head 维度
```

**Weight Absorption 机制 (MLAEinsum)**：权重加载时将 `kv_b_proj` 拆分为 `k_up_proj` 和 `v_up_proj`，分别用于 query 侧吸收和 output 侧投影。

**Pallas Kernel**：V2 Kernel 为生产版本，融合 KV Cache 更新 + 分离 Decode/Mixed Launch + FP8 Packing。

**关键文件**：

- `tpu_inference/models/jax/deepseek_v3.py` — `DeepseekV3MLA`, `MLAEinsum`
- `tpu_inference/kernels/mla/v2/kernel.py` — 生产 MLA Kernel
- `tpu_inference/layers/common/attention_interface.py` — `mla_attention()` 入口

**对 sglang-jax 参考**：MLA 是 DeepSeek V3 性能的核心。Weight Absorption 方案、压缩 KV Cache 格式、V2 Kernel 的融合设计均为高优先级参考项。

---

### 4.3 YARN RoPE with mscale ✅

**实现原理**：

三个独立实现覆盖不同模型需求：

| 变体 | 使用模型 | 特点 |
|---|---|---|
| `DeepseekScalingRotaryEmbedding` | DeepSeek V3 | YaRN correction range + mscale + interleaved pairing |
| `GptOssRotaryEmbedding` | GPT-OSS | NTK-by-parts + concentration factor（不缓存 sin/cos） |
| `apply_rope_scaling` | Llama3/4 | Llama 3.1 式平滑频率混合 |

DeepSeek V3 额外在 softmax scale 上叠加 mscale 校正：`scale = qk_head_dim^{-0.5} × yarn_mscale²`

**关键文件**：

- `tpu_inference/layers/jax/rope.py` — 所有 RoPE 实现
- `tpu_inference/layers/jax/rope_interface.py` — 函数式 RoPE 接口

---

### 4.4 Partial RoPE ✅

三种机制：

1. **Per-layer NoPE** (Llama4): 每 4 层一个 NoPE 层（无位置编码 + 分块注意力 + Temperature Tuning）
2. **rope_proportion** 参数 (rope_interface.py): sub-head-dim 级别部分旋转（已定义但未使用）
3. **MLA 结构级 Partial RoPE** (DeepSeek V3): `q_nope(128d)` + `q_rope(64d)` 拆分

---

### 4.5 Native Sparse Attention (NSA) ❌

无任何 NSA 实现。DeepSeek V3 使用标准 MHA 或 MLA，不包含 NSA 的分层 token 压缩/选择/滑动窗口注意力分支。

---

### 4.6 Linear Attention Variants ❌

无线性注意力机制（Linear Transformer、RetNet、RWKV 等）。所有注意力均基于标准 softmax。

---

## 五、优化能力

### 5.1 MoE 实现与 GMM Kernel ✅

**分层架构**：

```plaintext
分发层 (moe.py)          → 5 种后端选择 (FUSED_MOE, GMM_EP, GMM_TP, DENSE_MAT, MEGABLX_GMM)
编排层 (fused_moe_gmm.py) → Token 路由(scoring→top-k→argsort) + shard_map 分片(EP/TP) + 归约
计算核 (gmm_v2.py)        → Pallas GMM, emit_pipeline 三重缓冲, 动态 LHS 量化, 激活融合
加速器 (gather_reduce.py)  → TPU v7 SparseCore gather-reduce (batch > threshold 时启用)
通信层 (all_gather_matmul.py) → AllGather+MatMul 融合, 双向环通信, DMA-计算重叠
```

**GMM V2 Kernel 关键特性**：

- `pltpu.emit_pipeline` 自动 DMA-计算重叠，RHS 三重缓冲
- VMEM 感知 tiling（目标 1/3 VMEM 容量）
- Gate+Up 投影融合到单次 MatMul + 激活函数融合
- 动态 LHS 量化 + 子通道量化
- `BoundedSlice` 动态 DMA 大小

**SparseCore Gather-Reduce** (TPU v7)：

- Mosaic MLIR Kernel，使用 `tpu.enqueue_indirect_dma` 硬件间接 gather
- 双缓冲管线，多 SparseCore 并行

**关键文件**：

- `tpu_inference/layers/common/moe.py` — 分发
- `tpu_inference/layers/common/fused_moe_gmm.py` — 编排
- `tpu_inference/kernels/megablox/gmm_v2.py` — 生产 GMM Kernel
- `tpu_inference/kernels/gather/gather_reduce.py` — SparseCore 加速
- `tpu_inference/kernels/collectives/all_gather_matmul.py` — 通信融合

**对 sglang-jax 参考**：GMM V2 的 `emit_pipeline` + VMEM 感知 tiling 是 TPU 上 MoE 性能的关键。AllGather+MatMul 融合可用于 TP 场景优化。SparseCore gather-reduce 是 TPU v7 独有优化。

---

### 5.2 量化 ✅

| 方法 | 位宽 | 使用模型 | 说明 |
|---|---|---|---|
| FP8 (`float8_e4m3fn`) | 8-bit | DeepSeek V3, Qwen3, Llama4 | 权重 + KV Cache 量化 |
| Int8 | 8-bit | 通用 | 标准整型量化 |
| MXFP4 (`float4_e2m1fn`) | 4-bit | GPT-OSS | 两个 e2m1 打包到 uint8 + e8m0 scale |
| AWQ | 4-bit | 通用 | 特殊打包顺序 `(0,2,4,6,1,3,5,7)` |
| Blockwise | 可变 | MoE 专家权重 | 子通道 scale，每 block 独立量化 |

**量化 MatMul Kernel**：Per-channel 和 Blockwise 两种 Pallas Kernel，支持动态 LHS 量化。

**关键文件**：

- `tpu_inference/layers/common/quantization/__init__.py` — 量化/反量化函数
- `tpu_inference/kernels/quantized_matmul/kernel.py` — Per-channel Kernel
- `tpu_inference/kernels/quantized_matmul/blockwise_kernel.py` — Blockwise Kernel

---

### 5.3 通信-计算重叠 ✅

AllGather+MatMul 融合 Kernel：

```plaintext
Phase 0 (prologue):  启动远程 DMA + 本地 HBM→VMEM 复制
Phase 1..N (steady): DMA 下一 shard ∥ MatMul 当前 shard ∥ 写回上一结果
Phase N+1 (epilogue): 写回最终结果
```

双向环通信（左半左传、右半右传），有效带宽 ×2。

---

### 5.4 N:M 结构化稀疏矩阵乘 ✅

软件模拟 N:M 稀疏 MatMul（`Sparsifier` 编码/解码 + Pallas Kernel）。支持任意 N:M（M≤16），f32/bf16/int8 dtype。当前为独立 Kernel，需显式集成到 MoE pipeline。

---

### 5.5 KV Cache 管理 ✅

| 特性 | 实现 |
|---|---|
| 分页存储 | K/V 交错打包到固定大小页中 |
| 连续块分配器 | Best-fit 搜索优化 `dynamic_update_slice`（TPU 上 slice ≫ scatter） |
| MLA 压缩 Cache | ~640 维/token vs MHA 的 32768 维/token |
| DP 分片 | 页维度跨 ATTN_DATA 分片，KV heads 跨 ATTN_HEAD 分片 |
| 量化 Cache | FP8 KV Cache + kv_packing 支持 |

**关键文件**：

- `tpu_inference/runner/kv_cache.py` — Cache 创建
- `tpu_inference/runner/continuous_block_pool.py` — 连续块分配器

---

## 六、附加发现

### 多模态支持

tpu-inference 在以下模型中实现了视觉编码器：

| 模型 | 视觉编码器 |
|---|---|
| Qwen2.5-VL | 3D Conv Patch Embed + ViT + Window Attention + MRoPE |
| Llama4 | ViT + Pixel Shuffle + Multi-Modal Projector |
| Llama Guard 4 | 复用 Llama4 Vision Encoder |

### 补充注意力 Kernel

| Kernel | 用途 |
|---|---|
| 自定义 Flash Attention (Pallas) | Prefill + 视觉编码器 |
| Ragged Paged Attention (RPA) | Decode，支持 sliding window + attention sinks |
| RPA hd64 | GPT-OSS 专用 (head_dim=64) |
| MLA RPA (V2) | DeepSeek V3 MLA 专用 |
| Splash Attention | JAX 内置，支持 soft cap + MQA |
| Paged Attention | JAX 内置，支持 megacore 模式 |
