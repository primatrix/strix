# KDA (Kimi Delta Attention) sgl-jax 推理组件

| 字段     | 值                                 |
| -------- | ---------------------------------- |
| **作者** | @MokusMokun |
| **日期** | 2026-04-23 |
| **状态** | Draft (to be reviewed) |
| **issue** | [sgl-project/sglang-jax#948](https://github.com/sgl-project/sglang-jax/issues/948) |

---

## 1. 背景与动机

### 1.1 最终目标 — SGLangJax 支持 Kimi-Linear-48B-A3B-Instruct

sgl-jax 的目标是支持运行 [`moonshotai/Kimi-Linear-48B-A3B-Instruct`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)，并在公开测试集上对齐论文精度：

- **MMLU-Pro(EM)** 72.7（[Kimi-Linear 论文](https://arxiv.org/pdf/2510.26692) Table 9）
- Sampling: Temperature = 1.0（[论文 §Evaluation Configurations](https://arxiv.org/pdf/2510.26692)）
- Context length: 4k（[HuggingFace 模型卡](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)）

Kimi-Linear 关键模型属性：

- **KDA** — 一种带细粒度门控的 gated delta rule 线性注意力
- **3:1 KDA-to-MLA hybrid 架构** — 每 4 层中 3 层 KDA + 1 层 global MLA，以减小内存占用
- **无 MTP** — `num_nextn_predict_layers = 0`

本 RFC 仅交付 KDA 推理组件本身；模型组装、MLA 集成、3:1 hybrid 调度、HF 权重加载、MMLU-Pro 评测均由后续 issue / RFC 处理。

### 1.2 什么是 KDA

KDA (Kimi Delta Attention) 是 Kimi 团队提出的一种改进的线性注意力机制，结合了：

- **短卷积 (Short Convolution)**：通过 1D 因果卷积建模局部依赖
- **可学习 Decay (A_log)**：每头独立的衰减参数
- **门控机制 (Gate & Beta)**：全秩投影 g_proj / b_proj
- **线性注意力**：O(T) 复杂度，支持长序列

### 1.3 为什么需要在 sgl-jax 端单独实现

| 方面 | sglang (GPU 推理) | sgl-jax (本 RFC) |
|---|---|---|
| 框架 | PyTorch + Triton | JAX 0.8.1 + Flax NNX 0.12.4 |
| 硬件 | GPU/CUDA | TPU (Pallas) |
| Kernel | `chunk_kda` / `fused_recurrent_kda` (Triton) | jax-naive (本 RFC v1) → pallas (后续 RFC) |
| Cache | `MambaAttnBackendBase` + 上游 SSM/conv state pool | 消费 RFC-0015 `RecurrentStatePool`（recurrent + conv 双 buffer，本 RFC 不自建） |
| Layer 集成 | sglang `KimiDeltaAttention` | sgl-jax `KimiDeltaAttention` (nnx.Module) |

### 1.4 为什么 v1 = jax-naive，而不直接接 pallas kernel

上游 `pallas-kernel/tops/ops/kda` 的 forward kernel 计划下周交付。但 sgl-jax 端的集成工作（state pool 接驳、backend 串联、layer 约定、权重加载、`ForwardBatch` 适配）跟用哪个 kernel 实现无关。先用 jax-naive 把这些都跑通、对齐数值（性能不是 v1 的目标）；backend 默认使用 pallas，pallas import 失败时自动 fallback 到 jax-naive，切换点收敛在 `kernels/kda/__init__.py`，layer / pool / model 不感知。

### 1.5 参考实现

- **HF 上游模型**：[`moonshotai/Kimi-Linear-48B-A3B-Instruct/modeling_kimi.py`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py) 中的 `KimiDeltaAttention`（含已知 bug，详见 §6.1.5）
- **sglang GPU 实现**：[`sglang/srt/models/kimi_linear.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_linear.py) `KimiDeltaAttention`（L166-408）+ [`kda_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py)
- **姊妹 RFC (训练侧)**：[RFC-0008 KDA MaxText 实现](https://github.com/primatrix/wiki/blob/main/docs/rfc/0008-kda-maxtext-implementation.md)
- **配套 RFC (公共 state pool)**：[RFC-0015 Hybrid Architecture Recurrent State 管理](https://github.com/primatrix/wiki/pull/99)（@Rodrian7）— 本 RFC 消费其 `RecurrentStatePool` / `MemoryPools` / `HybridReqToTokenPool` / `forward_batch.recurrent_pool_indices`

---

## 2. 设计目标

### 2.1 功能目标

- [ ] `KimiDeltaAttention` nnx.Module — 完整 KDA 前向（projections / **q/k/v_conv1d + silu** / l2norm / gate / kernel / output gated norm / o_proj）
- [ ] `RadixLinearAttention` — layer→backend 的 dispatch 入口（对齐 sgl-jax `RadixAttention` + sglang `RadixLinearAttention`）
- [ ] `LinearRecurrentAttnBackend`（基类，未来 GDN / Mamba2 共用）+ `KDAAttnBackend`（KDA 子类）
- [ ] jax-naive kernel — `chunk_kda_jax_naive` (prefill) + `fused_recurrent_kda_jax_naive` (decode)
- [ ] kernel 切换：默认 pallas，import 失败时自动 fallback jax-naive（`kernels/kda/__init__.py`）
- [ ] 对接 RFC-0015 `RecurrentStatePool` / `MemoryPools` / `HybridReqToTokenPool`（pool 由 RFC-0015 提供，本 RFC 仅消费）

### 2.2 非目标

- 完整 Kimi-Linear 模型组装（`KimiLinearModel` / 3:1 hybrid 调度 / MLA layer / 权重加载）— 后续 issue
- pallas kernel vendor / port — 后续 RFC
- Speculative decoding (`DRAFT_EXTEND` / `TARGET_VERIFY`)
- KDA layer 量化（FP8 / INT8）
- `HybridLinearAttnBackend` wrapper（layer-id 级 KDA + MLA + FA dispatcher）— 不属于本 RFC 范围；由 PR #961 落地，@aolemila 正在写设计文档
- Kimi-Linear 端到端 MMLU-Pro 评测 — 模型组装完成后另立 issue
- `--enable-mixed-chunk=True`（一个 batch 里同时塞 prefill + decode）— chunk kernel 处理 "prefill segment + 1-token decode segment" 的 cu_seqlens 切片是否正确未验证；默认关，开了由用户自负风险。**Guard**：当模型使用 KDA backend 时，`ServerArgs.check_server_args` 检测到 `--enable-mixed-chunk=True` 直接报错。

> **`--chunked-prefill-size` 与 `--enable-mixed-chunk` 的区别**：
>
> - **chunked prefill**（默认 4096，**支持**）：把一个长 prefill 拆成多个 chunk 跨 step 完成；每个 chunk 进 backend 时 forward_mode = EXTEND，走 `chunk_kda`。
> - **mixed chunk**（默认关，**未验证**）：在同一个 batch 里同时塞 prefill 和 decode；scheduler 会把 forward_mode 强制写成 EXTEND（`schedule_batch.py:725-727 mix_with_running`），所以 backend 看到的还是 EXTEND，但 cu_seqlens 里会出现"长 prefill segment + 多个 1-token decode segment" 的混合形态。

**In-goal**：single-host TP、multi-host TP、DP-attn 必须工作或不被破坏。

### 2.3 验收方式

| ID | 验收对象 | 通过条件 | Tolerance |
|---|---|---|---|
| M1 | attention module 正确性 | layer 在等价 weight / 等价输入下输出与 HF 一致；prefill→decode invariance：`prefill(seq[:T])` ≈ `prefill(seq[:T-k]) → decode k 次`（k ≥ 8） | fp32 `atol=1e-5, rtol=1e-5`；bf16 `atol=1e-3, rtol=1e-3` |
| M2 | Lint clean | `pre-commit run --all-files` exit 0 | — |

> **Tolerance**：fp32 `atol=1e-5, rtol=1e-5`；bf16 `atol=1e-3, rtol=1e-3`。
> **Ground truth**：`kda_gpu/dumps`（H100 + Triton + fla-core 的 production kernel 输出，详见 §6.1）。

---

## 3. 架构设计

### 3.1 整体架构

```text
KimiLinearModel  (后续 issue)
    │
    ├─ KimiMLAAttention   (后续 issue)
    │
    └─ KimiDeltaAttention   ─────────  models/kimi_linear.py（对齐 sglang GPU）
        │  • q/k/v_proj
        │  • q/k/v_conv1d + silu      ◄── conv 算子 + 权重在 layer
        │  • l2norm(q,k)              ◄── conv 之后 (与 sglang 一致)
        │  • forget gate, beta, output gate
        │  • o_norm + o_proj
        │
        ├─ self.attn = RadixLinearAttention   ────  layers/radix_linear_attention.py
        │       │   dispatch 入口；持 layer 元数据
        │       ▼
        │   forward_batch.attn_backend(q,k,v,g,beta, layer=self, …)
        │       │
        │       ▼
        │   KDAAttnBackend  ──────────────  layers/attention/linear/kda_backend.py
        │       └─ 继承 LinearRecurrentAttnBackend  (KDA / GDN / Mamba2 共用基类)
        │              read prev recurrent → mode 派发 → write new recurrent
        │              kernel = pallas (default) | jax-naive (fallback)
        │
        └─ backend.get_conv_state / set_conv_state    ◄── conv buffer thin pass-through
                │
                ▼
            RecurrentStatePool   (RFC-0015 提供，本 RFC 不实现)
                • recurrent_buffer  fp32  [L, N+1, H, K, V]
                • conv_buffer       bf16  [L, N+1, W-1, q+k+v dim]
```

### 3.2 核心组件

| 组件 | 文件路径 | 功能描述 |
|---|---|---|
| `KimiDeltaAttention` | `python/sgl_jax/srt/models/kimi_linear.py` | 主 KDA 模块 (nnx.Module): projections / **q/k/v_conv1d + silu** / l2norm / gate / o_norm / o_proj；对齐 [sglang GPU `KimiDeltaAttention`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_linear.py#L166) |
| `RadixLinearAttention` | `python/sgl_jax/srt/layers/radix_linear_attention.py` | layer→backend dispatch 入口 + 元数据容器；对齐 sgl-jax `RadixAttention` 与 sglang `RadixLinearAttention` |
| `HybridLinearAttnBackend` | `python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py` | 不属于本 RFC 范围；由 PR #961 落地 |
| `LinearRecurrentAttnBackend` | `python/sgl_jax/srt/layers/attention/linear/kda_backend.py` | KDA / 未来 GDN / Mamba2 共用基类；对齐 sglang `MambaAttnBackendBase`；管 metadata、conv state I/O、forward mode 派发框架 |
| `LinearRecurrentAttnBackendMetadata` | 同上 | `LinearRecurrentAttnBackend` 的 metadata pytree（注册为 JAX pytree） |
| `KDAAttnBackend` | 同上 | KDA 子类：仅填 `_dispatch_chunk` / `_dispatch_recurrent` 绑定到 pallas（fallback jax-naive）kernel |
| `KDAAttnBackendMetadata` | 同上 | `KDAAttnBackend` 的 metadata pytree（注册为 JAX pytree） |
| `RecurrentStatePool` / `MemoryPools` | (RFC-0015 提供，本 RFC 不实现) | recurrent + conv state 双 buffer + slot 分配 + JIT donate 生命周期；详见 RFC-0015 |
| `chunk_kda_jax_naive` / `fused_recurrent_kda_jax_naive` | `python/sgl_jax/srt/kernels/kda/naive.py` | 纯 JAX kernel 实现（v1） |
| Kernel dispatcher | `python/sgl_jax/srt/kernels/kda/__init__.py` | 统一导出 `chunk_kda` / `fused_recurrent_kda`；默认 pallas，import 失败时 fallback jax-naive |

### 3.3 文件布局

布局对齐 PR #961 落地的目录结构：backend 在 `srt/layers/attention/linear/`，kernel 在 `srt/kernels/kda/`，model（含 `KimiDeltaAttention`）在 `srt/models/`。

```text
python/sgl_jax/srt/
├── models/
│   └── kimi_linear.py                    # 部分NEW — KimiDeltaAttention (KimiDecoderLayer + KimiLinearModel 取决于 RFC0018)
├── kernels/kda/                          # NEW — 对齐 kernels/simple_gla/
│   ├── __init__.py                       #   kernel dispatcher (pallas default, fallback naive)
│   ├── naive.py                          #   chunk_kda_jax_naive + fused_recurrent_kda_jax_naive
│   └── kda.py                            #   pallas kernel impl
└── layers/
    ├── radix_linear_attention.py         # NEW — RadixLinearAttention（与既有 radix_attention.py 同级）
    └── attention/
        ├── hybrid_linear_attn_backend.py # (PR #961，不属于本 RFC)
        └── linear/                       # NEW (PR #961) — linear attention backends
            └── kda_backend.py            #     LinearRecurrentAttnBackend + LinearRecurrentAttnBackendMetadata
                                          #     KDAAttnBackend + KDAAttnBackendMetadata
# RecurrentStatePool / HybridReqToTokenPool / MemoryPools 由 RFC-0015 在 mem_cache/memory_pool.py 落地

python/sgl_jax/test/
└── layers/test_kda_backend.py            # NEW — M1（attention module 对齐 HF + prefill→decode invariance）
```

### 3.4 Q/K L2 Normalization

L2-norm 在 layer 内、conv1d 之后、SSM kernel 之前完成（`KimiDeltaAttention.__call__` step 3，详见 §4.6）。这跟 sglang 把 L2 fuse 进 Triton kernel（`use_qk_l2norm_in_kernel=True`）效果等价；jax-naive kernel 不 fuse，所以放外部小算子做。

### 3.5 设计决策

| 项 | 决策 | 备注 |
|---|---|---|
| Backend 基类 | 新建 `LinearRecurrentAttnBackend`，对齐 sglang `MambaAttnBackendBase` | KDA / 未来 GDN / Mamba2 共用；详见 §4.3.1 |
| State pool | 不自建，消费 RFC-0015 `RecurrentStatePool` | 详见 §4.5 |
| conv1d 算子 | 在 `KimiDeltaAttention` layer | 权重在 layer |
| Conv state 读写路径 | backend 通过 `recurrent_state_pool.get/set_linear_recurrent_layer_cache` 直接读写 | 不经过 backend 转发接口 |
| Sharding | 由 RFC-0015 `RecurrentStatePool.__init__` 声明 | 本 RFC 不重复 |
| Pallas swap | 默认走 pallas；pallas import 失败时自动 fallback jax-naive | layer / pool / model 不感知 |
| K = V 假设 | `head_dim == v_head_dim == 128`（sglang `kimi_linear.py:182-187` 强制 $d_k = d_v$）| 本 RFC 沿用；K≠V 时需改 `o_norm` / `g_b_proj` |

---

## 4. 详细实现方案

### 4.1 Kernel 接口

**文件**：`python/sgl_jax/srt/kernels/kda/naive.py`（jax-naive）、`python/sgl_jax/srt/kernels/kda/kda.py`（pallas）

Kernel 的具体实现委托给 [pathfinder-pf](https://github.com/pathfinder-pf)，本 RFC 只提供文件和接口约定。两个 kernel 文件各导出同名函数，签名一致：

**Prefill — `chunk_kda` Pallas 实现**：

```python
o, recurrent_state = chunk_kda(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    initial_state=recurrent_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=True,
    cu_seqlens=cu_seqlens,
)
```

**Decode — `fused_recurrent_kda` Naive 实现**：

```python
o, recurrent_state = fused_recurrent_kda(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    initial_state=recurrent_state,
    output_final_state=True,
    use_gate_in_kernel=False,
)
```

### 4.2 Kernel Dispatcher

**文件**：`python/sgl_jax/srt/kernels/kda/__init__.py`

```python
from .naive import chunk_kda as chunk_kda_jax_naive
from .naive import fused_recurrent_kda as fused_recurrent_kda_jax_naive

# pallas 版本由 pathfinder-pf 提供；import 失败时 fallback jax-naive
try:
    from .kda import chunk_kda as chunk_kda_pallas
    _HAS_PALLAS_CHUNK = True
except ImportError:
    _HAS_PALLAS_CHUNK = False

try:
    from .kda import fused_recurrent_kda as fused_recurrent_kda_pallas
    _HAS_PALLAS_RECURRENT = True
except ImportError:
    _HAS_PALLAS_RECURRENT = False

# prefill / decode 独立 fallback
# v1: prefill pallas + decode naive
# v2: 全部 pallas
chunk_kda = chunk_kda_pallas if _HAS_PALLAS_CHUNK else chunk_kda_jax_naive
fused_recurrent_kda = fused_recurrent_kda_pallas if _HAS_PALLAS_RECURRENT else fused_recurrent_kda_jax_naive
```

### 4.3 Backend：dispatch 入口 + base class + KDA 子类

#### 4.3.0 `RadixLinearAttention`（layer→backend dispatch 入口）

**文件**：`python/sgl_jax/srt/layers/radix_linear_attention.py`

为 linear-recurrent 路径提供 dispatch 入口，对照 sgl-jax `RadixAttention`（`layers/radix_attention.py:23`）和 sglang `RadixLinearAttention`（`radix_linear_attention.py:31`）。Layer 在 `__init__` 内 `self.attn = RadixLinearAttention(...)`，forward 时调 `self.attn(...)` 把 q/k/v + 元数据送到 backend。

**为什么不复用 `RadixAttention` 而是另起一个**：KDA 的输入跟普通 attention 不一样（多了 `g`、`beta`，state 走 `recurrent_state_pool` 而非 `token_to_kv_pool`），如果改 `RadixAttention` 兼容 KDA 会让既有的 `RadixAttention` / `FlashAttentionBackend` 接口被 KDA 专用参数撑大。sglang 也是这么处理的——`radix_linear_attention.py` 跟 `radix_attention.py` 并存，本 RFC 1:1 沿用。

```python
class RadixLinearAttention(nnx.Module):
    """KDA / 未来 GDN / Mamba2 layer→backend dispatch 入口；元数据容器。
    对齐 sglang RadixLinearAttention（radix_linear_attention.py:31）。"""

    def __init__(
        self,
        layer_id: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
        # GDN / KDA shared weights
        conv_weights=None,
        bias=None,
        activation: str = "silu",
        A_log=None,
        dt_bias=None,
    ):
        ...

    def __call__(
        self,
        forward_batch,
        mixed_qkv: jax.Array,
        a: jax.Array,           # forget gate
        b: jax.Array,           # beta
        recurrent_state_pool,   # 
    ) -> jax.Array:
        return forward_batch.attn_backend.forward(
            layer=self,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            recurrent_state_pool=recurrent_state_pool, # 
        )
```

#### 4.3.1 `LinearRecurrentAttnBackend` + `LinearRecurrentAttnBackendMetadata`

**文件**：`python/sgl_jax/srt/layers/attention/linear/kda_backend.py`

KDA / 未来 GDN / Mamba2 共用基类，对齐 sglang `MambaAttnBackendBase`。基类不持权重也不持 pool 引用——pool 由 layer 在调用 forward 时显式传入。

**Metadata**：每个 forward batch 的动态 array 容器，用于穿越 JIT 边界（参考 `FlashAttentionMetadata`）。在 JIT 外计算，存入 backend 实例的 `self.forward_metadata`，作为 pytree child 穿越 JIT 边界，在 forward 时被 kernel 消费。

**Metadata 生命周期**：

```text
TpModelWorker.forward_batch_generation()                              ← JIT 外
  │
  ├─ forward_metadata = attn_backend.get_forward_metadata(model_worker_batch)
  │     → HybridLinearAttnBackend.get_forward_metadata():              (PR #961，不在本 RFC 范围)
  │       ├─ full_attn_backend.get_forward_metadata(...)
  │       │     → FlashAttentionMetadata（cu_q_lens, page_indices, ...）
  │       │     → full_attn_backend.forward_metadata = result
  │       │
  │       └─ linear_attn_backend.get_forward_metadata(...)
  │             → LinearRecurrentAttnBackendMetadata（cu_q_lens, recurrent_indices）
  │             → linear_attn_backend.forward_metadata = result        ← 子 backend 各自存自己的
  │             注：recurrent_indices 由 HybridLinearAttnBackend 算好后传入
  │                 (recurrent_state_pool.get_recurrent_indices(req_pool_indices))
  │
  ├─ attn_backend.forward_metadata = forward_metadata                  ← 顶层也存一份
  │     forward_batch.attn_backend 是同一个对象引用
  │
  └─ jitted_run_model(forward_batch, memory_pool, logits_metadata)     ← JIT 入口
       forward_batch.attn_backend（pytree child）携带 forward_metadata 穿越 JIT 边界
```

> **注意**：`HybridLinearAttnBackend` 的 `get_forward_metadata` 实现不在本 RFC 范围（PR #961）。本 RFC 只负责 `LinearRecurrentAttnBackend.get_forward_metadata` 的 linear 部分。`recurrent_indices` 由外部（`HybridLinearAttnBackend`）算好后传入，`LinearRecurrentAttnBackend` 自身不持 pool 引用。

| 类 | 字段 | pytree 注册 |
|---|---|---|
| `LinearRecurrentAttnBackendMetadata` | `cu_q_lens: jax.Array`（`[N+1]` int32, chunk_kda varlen）、`recurrent_indices: jax.Array`（`[B]` int32, pool slot 索引，来自 `recurrent_state_pool.get_recurrent_indices`） | `@register_pytree_node_class`；children = `(cu_q_lens, recurrent_indices)`，aux_data = `{}` |
| `LinearRecurrentAttnBackend` | `mesh`（仅 JIT 外用）, `forward_metadata` | `@register_pytree_node_class`；children = `(forward_metadata,)`，aux_data = 仅存 int/float 常量（`mesh` 不序列化，参考 `FlashAttention` 的做法） |

关键方法：

- `get_forward_metadata(model_worker_batch, recurrent_indices)` — 从 batch 计算 `cu_q_lens`，`recurrent_indices` 由外部传入；返回 `LinearRecurrentAttnBackendMetadata`
- `forward` — 按 `forward_mode` 分支到 `_dispatch_chunk`（传 `cu_seqlens`）或 `_dispatch_recurrent`；`recurrent_state_pool` 由 `RadixLinearAttention` 显式传入，通过 `get/set_linear_recurrent_layer_cache(layer.layer_id)` 读写单层 state
- `_dispatch_chunk` / `_dispatch_recurrent` — abstract，子类填

#### 4.3.2 `KDAAttnBackend` + `KDAAttnBackendMetadata`

**文件**：同上

继承基类，填两个 dispatch 方法绑定到 kernel（pallas default, fallback naive）。`KDAAttnBackendMetadata` 当前无额外字段（复用基类），预留扩展位。

| 类 | pytree 注册 |
|---|---|
| `KDAAttnBackendMetadata(LinearRecurrentAttnBackendMetadata)` | `@register_pytree_node_class`；继承基类 `tree_flatten` / `tree_unflatten` |
| `KDAAttnBackend(LinearRecurrentAttnBackend)` | `@register_pytree_node_class`；children = `(forward_metadata,)`，aux_data = 仅存 int/float 常量；`tree_unflatten` 中需重新绑定 kernel fns |

> **Per-runner 单例**：`KDAAttnBackend` 不持权重也不持 pool，可由 `ModelRunner` 创建一个实例注入到所有 KDA layer，与 `LinearAttentionBackend`（`linear_attention_backend.py:43`）一致。

### 4.4 ForwardMode → kernel dispatch

| `ForwardMode` | Kernel call | 行为 |
|---|---|---|
| `EXTEND` | `_chunk_fn(q, k, v, g, beta, initial_state, cu_seqlens, output_final_state=True)` | normal prefill |
| `DECODE` | `_recurrent_fn(q, k, v, g, beta, initial_state, output_final_state=True)` (T=1) | normal decode |
| `MIXED` | (不会出现) | 当前 sgl-jax 没有任何地方把 forward_mode 设成 `MIXED`：唯一的混合 batch 路径（`--enable-mixed-chunk`，默认关；§2.2 列为非目标）也会被 `mix_with_running` 强制写成 `EXTEND`。详见 §2.2 注释中两个 chunked flag 的区别 |
| `IDLE` / `DUMMY_FIRST` | (none) | backend 短路返回 zeros（与现有 backends 的 no-op 行为一致） |
| `DRAFT_EXTEND` / `TARGET_VERIFY` | (none) | 非目标；backend 抛 `NotImplementedError` |

### 4.5 State Pool（消费 RFC-0015）

本 RFC **不自建 state pool**。recurrent + conv state 的双 buffer、per-request slot、clear-on-alloc、JIT donate 生命周期统一由 [RFC-0015](https://github.com/primatrix/wiki/pull/99) 的 `RecurrentStatePool` 提供。

本 RFC 对 RFC-0015 的依赖项：

#### Buffer

- `recurrent_buffer`：`[L, N+1, H_local, K, V] fp32`，SSM 累积 buffer；TP 沿 H 轴切分
- `conv_buffer`：`[L, N+1, W-1, q_dim_local + k_dim_local + v_dim_local] bf16`，conv 滑窗 tail；TP 沿合并投影维度切分
- 不接入 RadixCache（`has_recurrent_state` 模型在 `server_args.__post_init__` 强制 `disable_radix_cache = True`）

#### 索引

- Layer 维度：pool 是全 layer 的（`[L, N+1, ...]`）。backend 在 forward 内通过 `layer.layer_id` 调 `recurrent_state_pool.get_linear_recurrent_layer_cache(layer_id)` 切出单层 view 再操作（对齐 sglang `req_to_token_pool.mamba2_layer_cache(layer.layer_id)`）
- Request 维度：`self.forward_metadata.recurrent_indices`（`[B] int32`，JIT 外由 `recurrent_state_pool.get_recurrent_indices(req_pool_indices)` 计算，通过 metadata pytree 穿越 JIT 边界）

#### `recurrent_state_pool` 传递路径（路线 B：独立 JIT 参数逐层透传）

`memory_pool` 作为 `jitted_run_model` 的独立 donated 参数，沿调用链显式透传。两种 attention 的 dispatch 入口各自从 `memory_pool` 中取出需要的 pool：

```text
jitted_run_model(forward_batch, memory_pool, ...)
                                ↑ donated JIT 参数（含 token_to_kv_pool + recurrent_state_pool）
  → model.forward(..., memory_pool)
    │
    ├─ RadixAttention(..., token_to_kv_pool=memory_pool.token_to_kv_pool)
    │   → backend.forward(..., token_to_kv_pool)
    │
    └─ KimiDecoderLayer — memory_pool.recurrent_state_pool 取出
        → KimiDeltaAttention.__call__(..., recurrent_state_pool)
            → RadixLinearAttention.__call__(..., recurrent_state_pool)
                → forward_batch.attn_backend.forward(..., recurrent_state_pool)
                  → KDAAttnBackend.forward_extend/decode(..., recurrent_state_pool)
                      ├─ layer_cache = recurrent_state_pool.get_linear_recurrent_layer_cache(layer.layer_id)
                      │     → conv_states, ssm_states（单层 view）
                      ├─ self.forward_metadata.recurrent_indices  → pool slot 索引
                      └─ kernel dispatch → 返回 updated states
```

model 层透传 `memory_pool`，上层模型组装代码（`KimiDecoderLayer`，后续 issue）从中取出 `recurrent_state_pool` 传给 `KimiDeltaAttention`。

#### 写回机制

Backend 在 kernel dispatch 后直接调用 `recurrent_state_pool.set_linear_recurrent_layer_cache(layer_id, req_indices, new_recurrent, new_conv)` 原地写回，不需要沿 forward 链返回新 buffer。与 KV cache 的 Pallas in-place update kernel 模式一致。

> **依赖时序**：上述接口必须由 RFC-0015 先 land；本 RFC 依赖该 PR。

### 4.6 KimiDeltaAttention Layer

**文件**：`python/sgl_jax/srt/models/kimi_linear.py`

**`__init__` 入参**：`config, layer_idx, mesh, dtype=jnp.bfloat16, *, rngs`

**字段**（与 HF [`modeling_kimi.py:514-540`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py) 1:1 对齐）：

| 类别 | 字段 | shape / 备注 |
|---|---|---|
| QKV 投影 | `q_proj` / `k_proj` / `v_proj` | `[hidden, num_heads * head_dim]`，与 dump weight 形态一致 |
| Beta / forget gate / output gate | `b_proj` / `f_a_proj` + `f_b_proj` / `g_a_proj` + `g_b_proj` | f / g 是 bottleneck rank=`head_dim` |
| KDA gate 参数 | `A_log` / `dt_bias` | `[num_heads]` log(Uniform(1,16)) / `[num_heads, head_dim]` ones |
| 短卷积（depthwise W=4） | `q_conv1d` / `k_conv1d` / `v_conv1d` | `feature_group_count = num_heads_local * head_dim`，CAUSAL；conv tail buffer 由 RFC-0015 持有 |
| 输出 | `o_norm` (FusedRMSNormGated) / `o_proj` | RMSNorm 沿 `head_dim`（K=V 假设，§3.5） |
| Dispatch 入口 | `attn = RadixLinearAttention(layer_id, num_q_heads, ..., conv_weights, A_log, dt_bias)` | §4.3.0 |

**`__call__` 入参 / 返回**：

```python
def __call__(self, hidden_states, positions, forward_batch, recurrent_state_pool)
    -> jax.Array
```

对齐 sglang GPU `KimiDeltaAttention.forward(hidden_states, positions, forward_batch, zero_allocator)`。sgl-jax 不需要 `zero_allocator`（JAX 不做 in-place bump alloc），取而代之的是 `recurrent_state_pool`（由上层模型组装代码从 `memory_pool` 中取出后传入，详见 §4.5）。`backend` 不作为入参（从 `forward_batch.attn_backend` 取得）。

**8 步前向**（对齐 sglang GPU `KimiDeltaAttention.forward`，L368-408）：

```python
# 1. QKV + beta + forget gate + g projections
mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(hidden_states)

# 2. prefill: unflatten forget_gate → [T, H, K]，beta sigmoid → fp32
#    decode: 跳过（kernel 内部处理）

# 3. dispatch（§4.3.0）
core_attn_out = self.attn(
    forward_batch,
    mixed_qkv=mixed_qkv,
    a=forget_gate,
    b=beta,
    recurrent_state_pool=recurrent_state_pool,
)

# 4. output gate + FusedRMSNormGated + o_proj
norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
core_attn_out = self.o_norm(core_attn_out, norm_gate)
out = self.o_proj(core_attn_out.flatten(-2))
```

> **`apply_qkv_conv_with_state` helper**：拼接 `[prev_tail | current]` → 沿时间维做 depthwise causal conv → 切回 current 段 + silu → 取尾 `W-1` 作为 new_tail。q/k/v 三路共享同一段函数（输入 dim 不同；helper 内部 split）。

---

## 5. 集成说明

> **注意**：Kimi-Linear 完整模型组装（`KimiLinearModel` / `KimiDecoderLayer` / 3:1 hybrid 调度）由后续 issue / RFC 负责。本节仅说明 KDA 推理组件的对外 API。

### 5.1 模型代码使用方式

```python
# 后续 KimiLinearModel 中，KDA backend + layer 的构造（示意）
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention

# state pool 由 RFC-0015 的 ModelRunner.init_memory_pool 创建，本 RFC 假定 model_runner.memory_pools.recurrent_state_pool 已就绪。

# KDA backend：全 model 共享一个实例，kernel 由 kernels/kda/__init__.py 自动选择
kda_backend = KDAAttnBackend(mesh=mesh)

kda_layer_ids = config.linear_attn_config.kda_layers   # e.g. [1, 2, 3, 5, 6, ...]
kda_layers = [
    KimiDeltaAttention(
        config=config,
        layer_idx=global_layer_idx,
        mesh=mesh, rngs=rngs,
    )
    for global_layer_idx in kda_layer_ids
]
```

### 5.2 ForwardBatch 接口与 `memory_pool` 传递

KDA layer 接收标准 `ForwardBatch` 实例 + `recurrent_state_pool`（由上层模型组装代码从 `memory_pool` 取出后传入）。Backend 通过 `forward_batch.forward_mode` 派发到 chunk / recurrent 路径（§4.4）。

**Cache slot 索引**：

| 维度 | 来源 | 怎么到 backend 手里 |
|---|---|---|
| Layer | `recurrent_state_pool.get_linear_recurrent_layer_cache(layer_id)` → 单层 view | 上层模型组装代码从 `memory_pool` 取出 `recurrent_state_pool`，经 `KimiDeltaAttention` → `RadixLinearAttention` 传给 backend |
| Request | `self.forward_metadata.recurrent_indices`（`[B] int32`） | JIT 外由 `recurrent_state_pool.get_recurrent_indices(req_pool_indices)` 计算，存入 `HybridLinearAttentionBackendMetadata`，作为 backend pytree child 穿越 JIT 边界 |

**KDA 与 KV cache 是两条索引路径**：MLA 等普通 attention 走 `out_cache_loc` + `req_to_token_pool.req_to_token`（按 token 索引）；KDA 走上面这套（按 request 索引）。hybrid 模型同时用两套，由 `HybridReqToTokenPool` 协调（详见 RFC-0015）。

### 5.3 不影响的现有组件

- `MHATokenToKVPool` / `SWAKVPool` — 与 KDA 无关 layer 仍然使用
- `FlashAttentionBackend` / `NativeAttention` / `LinearAttentionBackend` / MLA backend
- 所有现有 model（Qwen / Llama / Bailing 等）—— 不受影响；`memory_pool` 签名变更由 RFC-0015 统一处理
- `ServerArgs` CLI / 配置文件

---

## 6. 测试计划

### 6.1 数值参照 — `kda_gpu/dumps`

`kda_gpu/dumps` 是本 RFC 的唯一数值 ground truth。生成方式：拿 HF 上游 [`moonshotai/Kimi-Linear-48B-A3B-Instruct/modeling_kimi.py`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py) 中的 `KimiDeltaAttention`，先修掉 `fused_kda_gate` 形参错位 bug（详见 §6.1.5），再在 H100 上跑出每个 case 的 inputs / outputs / weights，按 fp32 + bf16 各一份存成 `.npz`。sgl-jax 端的 attention module 直接拿这些 dump 做跨平台数值对比。要点：

- **环境**：H100 80GB + driver 535.216.01 + `torch 2.7.1+cu128` + `triton 3.3.1` + `fla-core 0.4.2` + `transformers 4.56.2`
- **配置**：使用 HF checkpoint 实际配置（`hidden_size=4096, num_heads=32, head_dim=128, conv_size=4, rms_norm_eps=1e-6`），weights 直接从 HF checkpoint 提取，与 Step 1 测试共用同一套
- **12 个 case**：
  - Single-seq: `T=1, 8, 64, 65, 128, 256, 1024`
  - Varlen: `balanced_4x32`, `unbalanced[5,17,1,41]`, `single_T128`
  - Initial-state: `single_T128_initstate`, `varlen_initstate`
- **Schema**（每 case 一个 .npz）：
  - **Inputs**：`hidden_states`, optional `cu_seqlens`, optional `initial_recurrent_state`
  - **Outputs**：`out_fp32`（chunk 路径），`out_bf16`（bf16 重跑），`o_kda_chunk`, `recurrent_state_chunk`, `o_kda_fused_recurrent`, `recurrent_state_fused_recurrent`
  - **Weights**：`weights.npz`（一份共享，按 HF `KimiDeltaAttention.__init__` 的参数树存）
  - **本地 debug 仅保留，不进 CI**：`q_after_conv`, `k_after_conv`, `v_after_conv`, `g`, `beta`, `g_out`, `o_norm`（layer 内中间值；CI 走 final output 对齐就够，中间值用于失败时定位）
  - **不**包含 `q_l2 / k_l2`：上游走 `use_qk_l2norm_in_kernel=True`，L2 在 kernel 内做、未在 intermediate 暴露；JAX 端比对前自己做 L2（详见 §3.4）

### 6.1.5 上游 bug 在 dump 侧的处理

HF `modeling_kimi.py:560` 调 `fused_kda_gate` 时传错了参数。在 H100 + `torch 2.7.1+cu128` + `fla-core 0.4.2` + `triton 3.3.1` 上跑会直接抛 `TypeError`。`fused_kda_gate` 的真实签名是 `(g, A_log, dt_bias=None, lower_bound=None, output_dtype=torch.float32)`。

**dump 侧的修复**：subclass `KimiDeltaAttention`、override `forward`，按正确签名调 `fused_kda_gate(g.view(B, T, H, D), self.A_log, dt_bias=self.dt_bias)`。这跟 sglang fork 在 `chunk_kda_fwd` 里 fused gate 的语义等价。dump 包内还 pin 了 `modeling_kimi.py` 的 md5（`337ae1fc58c7010db4051e30fa23563e`）和 fla-core 版本，保证 dump 可复现。

**对 sgl-jax 的影响**：sgl-jax 端按修正后的语义实现（即 §4.6 的 `g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)`），与 dump 一致。**风险**：如果上游修了这个 bug，但 dump 还在用旧的 fixed-module 版本，sgl-jax 端可能跟 dump 对不上而引发误报。

### 6.2 精度测试 — attention module level

目标：把 `KimiDeltaAttention` 整个 layer 当黑盒，确认 jax 端组装的 layer 在等价 weight / 等价输入下，输出与 HF 一致。同时验证 prefill→decode invariance（state 跨 mode 衔接正确）。

- **Input / Output 参考**：[HF `modeling_kimi.py`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py)。mock HF L505（`KimiDeltaAttention.forward` 入口）的 `hidden_states` 和 `cache_params`（`KimiDynamicCache`），调用 layer 拿到 output 与 HF reference 输出比对。
- **Weight 提取**：直接从 HF safetensors 按 key pattern（`model.layers.{layer_idx}.self_attn.*`）提取单层各种 attention weights，`numpy→jax.array` 后按 JAX layer 的 pytree 结构灌入。不依赖完整 model loader，加载链路可参考 `python/sgl_jax/test/srt/test_model_loader.py` 中 `get_model_loader` → `download_model` → `load_model` 的调用方式。
- **Ref 实现**：HF，详见 §6.1.5
- **Invariance**：`prefill(seq[:T])` 最后一步 ≈ `prefill(seq[:T-k])` → k=8 步 decode 最后一步
- **Tolerance**：见 §2.3

### 6.3 集成测试（TODO，由后续 issue 落地）

- KDA layer 在 Kimi-Linear 完整模型中的 forward（待模型组装 issue）
- 3:1 KDA:MLA hybrid 端到端 forward
- HF 权重加载 + MMLU-Pro 评测（FinalGoal 验收）

### 6.4 CI 注册

不注册到 `test/srt/run_suite.py`。dump 脚本独立维护，dump 文件存 GCS。等 Kimi-Linear 完整模型集成的后续 issue 一起注册。

---

## 7. 参考文档

1. **Kimi-Linear 论文**: <https://arxiv.org/pdf/2510.26692>
2. **HF 模型卡**: <https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct>
3. **HF `modeling_kimi.py`** (含已知 bug): <https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py>
4. **sglang GPU `KimiDeltaAttention`**: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_linear.py>
5. **sglang GPU `KDAAttnBackend`**: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py>
6. **sglang GPU `chunk_kda` (Triton)**: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/kda.py>
7. **fla-core 库**: <https://github.com/fla-org/flash-linear-attention>
8. **上游 Pallas 内核 `tops.ops.kda`**: <https://github.com/primatrix/pallas-kernel/tree/main/tops/ops/kda>
9. **姊妹 RFC (训练侧)**: [RFC-0008 KDA MaxText 实现](https://github.com/primatrix/wiki/blob/main/docs/rfc/0008-kda-maxtext-implementation.md)
10. **配套 RFC (公共 state pool)**: [RFC-0015 Hybrid Architecture Recurrent State 管理](https://github.com/primatrix/wiki/pull/99) — `RecurrentStatePool` / `MemoryPools` / `HybridReqToTokenPool`
