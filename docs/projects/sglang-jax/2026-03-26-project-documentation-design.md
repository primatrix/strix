# sglang-jax 项目文档设计

**日期**：2026-03-26
**目标受众**：项目开发者/贡献者
**语言**：中文
**策略**：全新重写，替换现有 docs/ 内容
**粒度**：架构导览级（每模块 1-2 页，重点是职责、接口、数据流）
**组织方式**：按系统层次（请求从入口流经系统到底层）

---

## 文档总览

| # | 文件名 | 标题 | 篇幅 |
|---|--------|------|------|
| 1 | `00-overview.md` | 项目概览 | 1-2 页 |
| 2 | `01-quickstart.md` | 快速上手 | 1-2 页 |
| 3 | `02-architecture.md` | 系统架构总览 | 2-3 页 |
| 4 | `03-entrypoints.md` | 入口层 | 1-2 页 |
| 5 | `04-scheduler.md` | 调度层 | 2 页 |
| 6 | `05-model-executor.md` | 模型执行层 | 2 页 |
| 7 | `06-models.md` | 模型实现 | 2 页 |
| 8 | `07-attention-kv-cache.md` | Attention & KV Cache | 2-3 页 |
| 9 | `08-core-layers.md` | 核心层组件 | 2 页 |
| 10 | `09-pallas-kernels.md` | Pallas 内核 | 2-3 页 |
| 11 | `10-speculative-decoding.md` | 投机解码 | 1-2 页 |
| 12 | `11-lora.md` | LoRA 动态加载 | 1-2 页 |
| 13 | `12-quantization.md` | 量化 | 1-2 页 |
| 14 | `13-structured-output.md` | 结构化输出 & Function Calling | 1 页 |
| 15 | `14-multimodal.md` | 多模态子系统 | 2 页 |
| 16 | `15-developer-guide.md` | 开发者指南 | 2 页 |

目前总计：16 篇文档，约 25-35 页

---

## 各文档详细设计

### 1. 项目概览 (`00-overview.md`)

**覆盖内容**：

- sglang-jax 是什么：SGLang 推理引擎的 JAX/TPU 移植版，核心价值和定位

- 支持的模型列表：
  - Dense: Qwen 1/2/3, Llama, Gemma2, Grok-2
  - MoE: Qwen-MoE, Bailing-MoE, Grok-2

  - 多模态: Qwen2.5-VL, Qwen3-Omni, Wan DiT/VAE, MiMo Audio

- 支持的硬件（TPU v5e/v6e/v7x，实验性 GPU）

- 核心特性一览（RadixCache、ChunkedPrefill、投机解码、LoRA、量化、MoE、多模态等）

- 与原版 SGLang (PyTorch/CUDA) 的异同

**涉及文件**：`README.md`, `python/pyproject.toml`, `srt/models/registry.py`

### 2. 快速上手 (`01-quickstart.md`)

**覆盖内容**：

- 环境要求（Python 3.12+, JAX 0.8.1, TPU VM / GPU）

- 安装方式（pip, Docker, 源码）

- 启动一个 Qwen 模型的完整流程（launch_server → curl 请求 → 看到输出）

- 常见问题 FAQ

**涉及文件**：`Dockerfile`, `Makefile`, `launch_server.py`, `__main__.py`, `server_args.py`

### 3. 系统架构总览 (`02-architecture.md`)

**覆盖内容**：

- 三进程模型全景图：TokenizerManager（主进程） → Scheduler（子进程） → DetokenizerManager（子进程）

- 请求生命周期：HTTP 请求 → 分词 → 调度 → Prefill/Decode → 采样 → 增量解码 → SSE 响应

- 进程间通信机制（ZMQ）

- 关键数据结构流转：Req → ScheduleBatch → ModelWorkerBatch → ForwardBatch

- 并行策略概览：Tensor Parallel、DP-Attention、Expert Parallel，JAX Device Mesh

- JAX 特有适配：NNX graph-state 分离、形状桶化与预编译、Pallas in-place 更新

**涉及文件**：

- `srt/entrypoints/engine.py`

- `srt/managers/scheduler.py`

- `srt/managers/tp_worker.py`

- `srt/model_executor/model_runner.py`

- `srt/managers/schedule_batch.py`

- `srt/utils/mesh_utils.py`

### 4. 入口层 (`03-entrypoints.md`)

**覆盖内容**：

- HTTP Server：FastAPI 路由结构，OpenAI 兼容 API（/v1/chat/completions, /v1/completions, /v1/embeddings）

- Engine 类：如何组装 TokenizerManager、Scheduler、DetokenizerManager

- OpenAI 协议实现：ChatCompletionRequest/Response，SSE streaming 机制

- TokenizerManager：请求接收、分词、会话模板、Jinja 模板

- DetokenizerManager：增量解码、token 到文本的流式转换

**涉及文件**：

- `srt/entrypoints/http_server.py`

- `srt/entrypoints/engine.py`, `EngineBase.py`

- `srt/entrypoints/openai/` 所有文件

- `srt/managers/tokenizer_manager.py`, `detokenizer_manager.py`

- `srt/managers/io_struct.py`

### 5. 调度层 (`04-scheduler.md`)

**覆盖内容**：

- Scheduler 主循环：event_loop_normal() 的工作流

- 请求状态机：WAITING → RUNNING → FINISHED（含 rewind/pause/continue）

- ScheduleBatch：批次管理，Prefill vs Decode 分离

- 调度策略：FCFS、Cache-Aware，PrefillAdder 的 token 预算控制

- Continuous Batching：动态加入/移除请求

- Chunked Prefill：长序列分块处理

- 内存管理交互：与 RadixCache、KV Pool 的协作

**涉及文件**：

- `srt/managers/scheduler.py`

- `srt/managers/schedule_batch.py`

- `srt/managers/schedule_policy.py`

- `srt/managers/scheduler_output_processor_mixin.py`

### 6. 模型执行层 (`05-model-executor.md`)

**覆盖内容**：

- ModelRunner：JIT 编译的 forward 函数（forward_prefill, forward_decode, forward_extend）

- ForwardBatch 与 ForwardMode：不同前向模式的数据准备

- 形状桶化（Shape Bucketing）：为什么需要、如何工作、桶的选择策略

- 预编译（Precompilation）：预热各种形状的 JIT trace

- KV Cache Pool 初始化：内存分配与 Pool 类型选择

- NNX Graph-State 分离：如何让 JAX JIT 与 stateful model 兼容

**涉及文件**：

- `srt/model_executor/model_runner.py`

- `srt/model_executor/forward_batch_info.py`

- `srt/managers/tp_worker.py`

### 7. 模型实现 (`06-models.md`)

**覆盖内容**：

- 模型注册机制：registry.py 的 EntryClass 发现模式

- 模型结构模式：以 Qwen2 为例讲解 Decoder-only Transformer 的实现

- 各模型架构概览：Dense（Qwen/Llama/Gemma/Grok）、MoE（Qwen-MoE/Bailing）

- 特殊模型：Eagle3 投机解码模型、MiMo MTP、UMT5 Encoder-Decoder

- 权重加载：HuggingFace → JAX 的转换与分片逻辑（weight_utils.py）

- 如何添加一个新模型（简要步骤指引）

**涉及文件**：

- `srt/models/registry.py`

- `srt/models/qwen2.py`（作为典型示例）

- `srt/utils/weight_utils.py`

- `srt/configs/model_config.py`

### 8. Attention & KV Cache (`07-attention-kv-cache.md`)

**覆盖内容**：

- Attention 后端抽象：BaseAttnBackend 接口设计

- FlashAttention 后端：Pallas Ragged/Paged Attention 内核的调用方式

- Native 后端：纯 JAX 实现，作为参考/回退

- RadixAttention 层：如何包装后端

- KV Cache 架构：
  - ReqToTokenPool：请求到 token slot 的映射
  - MHATokenToKVPool / SplitMHATokenToKVPool：KV 存储池（支持 K/V 不同 head 维度）
  - SWAKVPool：滑动窗口注意力的 KV 池

  - Allocator：Token 分配器、Paged 分配器

- RadixCache：前缀共享的 Radix Tree 实现

- Pallas KV Update 内核：如何突破 JAX 不可变性限制

**涉及文件**：

- `srt/layers/attention/`

- `srt/layers/radix_attention.py`

- `srt/mem_cache/memory_pool.py`, `radix_cache.py`, `allocator.py`

- `srt/kernels/ragged_paged_attention/`, `paged_attention/`, `update_kv_cache/`

### 9. 核心层组件 (`08-core-layers.md`)

**覆盖内容**：

- Linear 层：Tensor Parallel 的列/行分片策略

- MoE 层：路由（Gate）→ 专家计算 → 结果合并，Expert Parallel 支持

- FusedMoE：融合内核的使用与优势

- Embedding 层：Token/Position Embedding，RoPE（含 YaRN、3D-RoPE）

- LayerNorm：RMSNorm 实现

- Sampler：贪心/Top-K/Top-P/Temperature 采样

- Logits Processor：logits 计算与后处理

**涉及文件**：

- `srt/layers/linear.py`, `moe.py`, `fused_moe.py`, `gate.py`

- `srt/layers/embeddings.py`, `layernorm.py`

- `srt/layers/sampler.py`, `logits_processor.py`

### 10. Pallas 内核 (`09-pallas-kernels.md`)

**覆盖内容**：

- Pallas 简介：JAX 的低级内核编程框架，为什么 sglang-jax 需要它

- Ragged Paged Attention 内核：核心算法、split 版本、tuned block sizes

- KV Cache Update 内核：如何实现 in-place mutation

- Fused MoE 内核（v1）：专家计算融合，tuned block configs

- GMM 内核：MegaBLoX 风格的 Grouped Matrix Multiply，v1 vs v2

- Quantized MatMul 内核：INT8/FP8/Block-wise 量化矩阵乘

- 投机解码内核：Eagle tree 构建、tree sampling、greedy verification

- 内核调优流程：如何生成 tuned block sizes（benchmark → YAML configs）

**涉及文件**：

- `srt/kernels/ragged_paged_attention/`

- `srt/kernels/update_kv_cache/`

- `srt/kernels/fused_moe/v1/`

- `srt/kernels/gmm/`

- `srt/kernels/quantized_matmul/`

- `srt/kernels/speculative/`

- `benchmark/kernels/`

### 11. 投机解码 (`10-speculative-decoding.md`)

**覆盖内容**：

- Eagle 算法原理：draft model → tree construction → verification

- Eagle Worker 架构：与主 ModelWorker 的协作

- Tree 构建与验证：eagle_util.py 的核心逻辑

- Pallas 内核加速：tree 结构操作的内核化

**涉及文件**：`srt/speculative/`, `srt/kernels/speculative/`, `srt/models/llama_eagle3.py`

### 12. LoRA 动态加载 (`11-lora.md`)

**覆盖内容**：

- LoRA 架构：Manager → Registry → Memory Pool → BGMV Backend

- 动态加载机制：运行时切换 LoRA adapter

- 内存池管理：LoRA 权重的分配与复用

- 与 RadixCache 的交互：LoRA namespace 隔离

**涉及文件**：`srt/lora/`

### 13. 量化 (`12-quantization.md`)

**覆盖内容**：

- 量化配置体系：quantization_config.py + YAML presets

- 支持的量化方式：FP8 per-tensor、INT8、Block Dynamic、Channel-wise

- 量化内核：quantized_matmul/ 内核调用

- 各模型的量化适配

**涉及文件**：`srt/configs/quantization_config.py`, `srt/utils/quantization/`, `srt/kernels/quantized_matmul/`

### 14. 结构化输出 & Function Calling (`13-structured-output.md`)

**覆盖内容**：

- llguidance 集成：grammar-guided generation

- Bitmask 操作：constrained decoding 的 token mask

- Function Call Parser：工具调用解析

- EBNF Composer：语法规则组合

**涉及文件**：`srt/constrained/`, `srt/function_call/`

### 15. 多模态子系统 (`14-multimodal.md`)

**覆盖内容**：

- 多模态架构概览：独立的子系统设计（独立 HTTP Server、GlobalScheduler）

- 子调度器体系：vit_scheduler, diffusion_scheduler, vae_scheduler, audio_scheduler, embed_scheduler

- 模型支持：
  - Qwen2.5-VL：ViT + Language Model 的视觉理解流水线
  - Qwen3-Omni-MoE：视觉 + 音频 + 语言的多模态 Thinker
  - Wan 2.1/2.2 DiT + VAE：文本到视频生成流水线

  - MiMo Audio：音频理解 backbone

- 流水线配置：static_configs/ 中的 stage YAML

- 专用 Pallas 内核：DiT Flash Attention

**涉及文件**：`srt/multimodal/` 整个子目录

### 16. 开发者指南 (`15-developer-guide.md`)

**覆盖内容**：

- 项目结构速查：目录 → 职责对照表

- 本地开发环境搭建：依赖、pre-commit hooks、代码风格

- 测试体系：
  - 单元测试（python/sgl_jax/test/）：内核、层、缓存
  - 集成测试（test/srt/）：引擎、serving、模型

  - 测试套件运行器：run_suite.py

- CI/CD 流程：PR 测试、Nightly 测试、发布流水线

- 基准测试与 Profiling：
  - 内核基准（benchmark/kernels/）
  - 服务基准（bench_serving.py）

  - Profiling 工具（profiler.py, memory_profiler.py, precision_tracer.py）

- 贡献代码流程：PR 规范、Review 指南

**涉及文件**：

- `test/`, `python/sgl_jax/test/`

- `benchmark/`

- `.github/workflows/`

- `.pre-commit-config.yaml`
