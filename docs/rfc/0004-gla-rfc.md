# GLA Refactor RFC

**日期**：2026-04-02
**状态**：Implemented
**目标**：将 ant-pretrain 中 GLA (Gated Linear Attention) 相关代码重构后合入 maxtext 上游
**PR**：<https://github.com/primatrix/maxtext/pull/16>

---

## 1. 重构概述

将 ant-pretrain 中 GLA 线性注意力的实现重构后推回 maxtext 上游。重构核心内容：

1. 将算子与 Module 层分离，算子放入 `kernels/gla/` 目录
2. Pallas kernel 支持 fused chunk GLA（消除 HBM 中间张量往返），fallback 到 non-fused 路径
3. 删除仅在测试中使用的 kernel 函数（`naive_chunk_simple_gla`、`parallel_chunk_simple_gla`、`naive_recurrent_simple_gla`、`chunk_gla`、`repeat_kv`）及纯 JAX fallback（`_chunk_gla_fixed`）
4. Module 层添加 `checkpoint_name` 支持 remat 策略，`jax.named_scope` 支持 XProf 可视化
5. `tops` (pallas-kernel) 作为必选依赖加入 `pyproject.toml`
6. 保留端到端 Megatron 精度对比测试

---

## 2. 现状分析

### 2.1 ant-pretrain 中 GLA 代码结构（重构前）

```text
ant-pretrain/
├── src/MaxText/
│   ├── kernels/
│   │   └── gla_pallas.py              # Pallas kernel wrapper (271 行)
│   └── layers/
│       └── attention_gla.py           # 混合文件：纯函数算子 + Module (938 行)
│           ├── build_slope_tensor()        # slope 构建
│           ├── naive_chunk_simple_gla()    # 参考实现（仅测试用）
│           ├── parallel_chunk_simple_gla() # 参考实现（仅测试用）
│           ├── _chunk_gla_fixed()          # Pallas 引入前的生产实现（纯 JAX）
│           ├── chunk_gla()                 # 统一入口（仅测试用，已被 Module 绕过）
│           ├── naive_recurrent_simple_gla()# 参考实现（仅测试用）
│           ├── repeat_kv()                 # 死代码（GQA groups=1 永不触发）
│           └── BailingMoeV2LinearAttention # 主 Module (含 checkpoint_name + named_scope)
└── tests/
    ├── test_attention_gla.py          # 纯函数精度测试 (1048 行)
    ├── test_chunk_gla_pallas.py       # Pallas kernel 测试 (582 行)
    ├── test_chunk_gla_pallas_gpu_vs_triton.py  # GPU 测试（不推上游）
    └── gla_compare_test.py            # Megatron 端到端精度对比 (497 行)
```

### 2.2 问题

1. `attention_gla.py` 混合了算子实现和 Module 定义，职责不清
2. 多个纯函数仅用于测试，不在生产路径中，应删除
3. `repeat_kv()` 是死代码（`num_query_heads == num_kv_heads`）
4. `_chunk_gla_fixed()` 是 Pallas 引入前的生产实现，tops 作为必选依赖后已无需保留
5. `chunk_gla()` 函数已被 Module 中的 `shard_map` + `pallas_chunk_gla` 直接调用所替代
6. Pallas kernel 通过 `sys.path` 硬编码导入 `tops` 包
7. Pallas 不可用时无告警，用户无法感知性能降级

---

## 3. 重构设计

### 3.1 目标结构

```text
maxtext/
├── src/maxtext/
│   ├── kernels/
│   │   └── gla/                                # 新增子目录
│   │       ├── __init__.py                     # 统一入口 + build_slope_tensor
│   │       └── pallas.py                       # pallas_chunk_gla (TPU kernel, fused + non-fused)
│   └── layers/
│       ├── normalizations.py                   # 修改：新增 GroupRMSNorm
│       └── attention_gla.py                    # 新增：BailingMoeV2LinearAttention Module
├── .github/
│   └── ci/
│       └── unit-tests-job.yaml                 # 修改：新增 gla_compare_test 步骤
├── tests/
│   └── unit/
│       └── gla_compare_test.py                 # Megatron 端到端精度对比
└── pyproject.toml                              # 修改：新增 tops 必选依赖
```

### 3.2 模块架构图

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    BailingMoeV2LinearAttention                      │
│                     (layers/attention_gla.py)                       │
│                                                                     │
│  hidden_states ─► QKV Proj ─► QK Norm ─► RoPE ─► GLA Kernel ─►    │
│                   (DenseGeneral)  (RMSNorm)  (RotaryEmb)    │       │
│                 [qkv_proj]   [query/key/value_proj]         │       │
│                                                             ▼       │
│                                    [gla_context] GroupRMSNorm ─►    │
│                                                 Sigmoid Gate ─►    │
│                                                [gate_proj]         │
│                                                 Dense ─► output    │
│                                                [out_proj]          │
│                                                                     │
│  jax.named_scope: qkv_proj, rope, gla_recurrence,                  │
│                   group_norm, gate, out_proj                        │
│  checkpoint_name: qkv_proj, query_proj, key_proj, value_proj,      │
│                   gla_context, gate_proj, out_proj                  │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │     kernels/gla/__init__.py        │
                     │                                    │
                     │  chunk_gla(q, k, v, g_gamma, ...)  │
                     │  build_slope_tensor(n_heads)       │
                     │                                    │
                     └────────────────┬───────────────────┘
                                      │
                        ┌─────────────┴───────────┐
                        │ kernels/gla/pallas.py    │
                        │                          │
                        │ pallas_chunk_gla         │
                        │  (custom_vjp)            │
                        │                          │
                        │  _use_fused?             │
                        │    ╱       ╲             │
                        │  Yes       No            │
                        │   │         │            │
                        │ FUSED    CHUNK           │
                        │ _CHUNK   mode            │
                        │   │         │            │
                        │   ▼         ▼            │
                        │  ┌──────────┐            │
                        │  │   tops   │            │
                        │  │ (pallas- │            │
                        │  │  kernel) │            │
                        │  │ 必选依赖  │            │
                        │  │ v0.3     │            │
                        │  └──────────┘            │
                        └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据流 (前向):

  hidden_states [B, T, emb_dim]
       │
       ├──► query_key_value ──► [B, T, H_total, head_dim]
       │        split ──► Q [B,T,H,K], K [B,T,H,K], V [B,T,H,K]
       │                    │
       │              QK RMSNorm
       │                    │
       │                  RoPE
       │                    │
       │    g_gamma = -slope_base * slope_scale   ← build_slope_tensor(H)
       │              (H,) 常量                      layer_idx 相关缩放
       │                    │
       │              chunk_gla(Q, K, V, g_gamma) ← 统一调度
       │              ├─ fused path (无 initial_state 时)
       │              └─ non-fused fallback
       │                    │
       │              [B, T, H, K] → reshape → [B, T, H*K]
       │                    │
       │              GroupRMSNorm
       │                    │
       ├──► g_proj ──► sigmoid ──► element-wise multiply
       │                    │
       │              reshape → [B, T, H, K]
       │                    │
       │              dense (output projection)
       │                    │
       └──────────────► output [B, T, emb_dim]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

测试覆盖:

  ┌──────────────┴──────────────┐
  │ gla_compare_test.py         │ ← Module 端到端 vs Megatron dump
  │  (e2e test, 需 DUMP_DIR)    │    前向 + 反向激活梯度 + 权重梯度
  └─────────────────────────────┘
```

---

## 4. 详细设计

### 4.1 `kernels/gla/__init__.py` — 统一入口

```python
from maxtext.kernels.gla.pallas import pallas_chunk_gla


def build_slope_tensor(n_attention_heads: int) -> jnp.ndarray:
    """构建 ALiBi-style per-head slope tensor。"""
    ...


def chunk_gla(q, k, v, g_gamma, scale=None, initial_state=None,
              output_final_state=False, chunk_size=64):
    """统一入口：调用 pallas_chunk_gla。"""
    return pallas_chunk_gla(q, k, v, g_gamma, initial_state,
                            scale, output_final_state, chunk_size)
```

- `tops` 作为必选依赖，导入失败直接报错（由 `pallas.py` 抛出 `ImportError`）

### 4.2 `kernels/gla/pallas.py` — Pallas kernel wrapper（含 fused 支持）

**来源**：ant-pretrain `src/MaxText/kernels/gla_pallas.py`（271 行）

**双路径架构**：

| 路径 | 条件 | API | 说明 |
|------|------|-----|------|
| Fused | `initial_state is None and not output_final_state` | `SimpleGLAKernelMode.FUSED_CHUNK` | 合并 h 传播 + A 重计算 + 输出为单次 pallas_call，消除 HBM 往返 |
| Non-fused | 有 `initial_state` 或 `output_final_state` | `SimpleGLAKernelMode.CHUNK` | 支持初始状态和最终状态输出 |

**主要函数**：

- `pallas_chunk_gla()` — `jax.custom_vjp` 封装，自动选择 fused/non-fused 路径
- `_pallas_chunk_gla_fwd()` — 前向规则，residuals 首元素用 `True/False` sentinel 区分路径
- `_pallas_chunk_gla_bwd()` — 反向规则，根据 sentinel 调用对应 backward，g_gamma 梯度返回零
- `_use_fused_kernels()` — 路径选择逻辑
- `_pad_inputs_fused()` — fused 路径的 padding（无 h0）
- 辅助函数：`_resolve_scale`、`_to_1d_g_gamma`、`_pad_axis`、`_pad_inputs`

**导入方式**：

```python
try:
    from tops.ops.simple_gla import simple_gla_bwd
    from tops.ops.simple_gla import simple_gla_fwd
    from tops.ops.simple_gla import SimpleGLAKernelMode
except ImportError as e:
    raise ImportError(
        "pallas-kernel (tops) is required for Pallas GLA kernels. "
        "Install it from the pallas-kernel submodule."
    ) from e
```

- 去除 `sys.path` 硬编码，通过 `pyproject.toml` 必选依赖安装 `tops`
- 使用新 API：`simple_gla_fwd/bwd` + `SimpleGLAKernelMode`（替代旧的 `chunk_simple_gla_fwd/bwd`）

### 4.3 `layers/attention_gla.py` — Module 层

**来源**：ant-pretrain `src/MaxText/layers/attention_gla.py` 第 688-938 行

仅保留：

- `BailingMoeV2LinearAttention` (nnx.Module)

**新增功能**：

| 功能 | 说明 |
|------|------|
| `checkpoint_name` | 7 个 remat 检查点：qkv_proj, query_proj, key_proj, value_proj, gla_context, gate_proj, out_proj |
| `jax.named_scope` | 6 个 XProf 分组：qkv_proj, rope, gla_recurrence, group_norm, gate, out_proj |

**移除内容**：

| 函数 | 行数 | 移除原因 |
|------|------|---------|
| `naive_chunk_simple_gla()` | 58-162 | 仅测试用参考实现，删除 |
| `parallel_chunk_simple_gla()` | 170-314 | 仅测试用参考实现，删除 |
| `naive_recurrent_simple_gla()` | 602-659 | 仅测试用参考实现，删除 |
| `chunk_gla()` | 488-594 | 已被 `__init__.py` 调度逻辑替代，删除 |
| `repeat_kv()` | 662-680 | 死代码（`num_key_value_groups` 恒为 1），删除 |
| `_chunk_gla_fixed()` | 322-485 | Pallas 引入前的纯 JAX 实现，tops 必选后无需保留，删除 |
| `build_slope_tensor()` | 32-50 | 移入 `kernels/gla/__init__.py` |

**Module 适配要点**：

- 导入路径：`MaxText.xxx` → `maxtext.xxx`
- 从 `kernels.gla` 导入 `chunk_gla` 和 `build_slope_tensor`
- 移除 `repeat_kv` 相关代码及 `num_key_value_groups` 字段
- Module `__call__` 中通过 `shard_map` 调用 `kernels.gla.chunk_gla()`
- 新增 `from jax.ad_checkpoint import checkpoint_name` 导入

### 4.4 `layers/normalizations.py` — 新增 GroupRMSNorm

**来源**：ant-pretrain `src/MaxText/layers/normalizations.py` 第 86-128 行

新增 `GroupRMSNorm` (nnx.Module)，在现有 `RMSNorm` 类之后添加。将特征维度分为 `group_norm_size` 个组，每组独立做 RMS 归一化。

### 4.5 `pyproject.toml` — tops 必选依赖

```toml
dependencies = [
    "tops @ git+https://github.com/primatrix/pallas-kernel.git@release/v0.3",
]
```

通过 `pyproject.toml` 的 `dependencies` 字段将 `tops`（pallas-kernel）声明为必选依赖，指向 `release/v0.3` 分支。`allow-direct-references = true` 已配置，支持 git URL 直接引用。

安装方式：`pip install -e .`

---

## 5. 测试策略

### 5.1 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `tests/unit/gla_compare_test.py` | 端到端测试 | Megatron dump 精度对比（前向+反向+权重梯度） |

### 5.2 `gla_compare_test.py`

端到端 Megatron 精度对比测试，验证内容：

| 测试方法 | 验证内容 | 容差标准 |
|---------|---------|---------|
| `test_forward_bf16` | 前向输出 vs Megatron | allclose + bf16 ULP |
| `test_backward_bf16` | 反向激活梯度 vs Megatron | allclose + bf16 ULP |
| `test_weight_grads_bf16` | 权重梯度 vs Megatron | allclose + bf16 ULP |

**外部依赖**：

- `DUMP_DIR` 环境变量（Megatron dump 数据目录）
- `argus` 包（`list_tensors`、`load_tensor`）
- Orbax checkpoint（默认路径 `/models/gpu-ckpt-ling2.5/...`）
- 无 `DUMP_DIR` 时自动 skip

### 5.3 不推送的测试

| 文件 | 原因 |
|------|------|
| `test_chunk_gla_pallas_gpu_vs_triton.py` | GPU + Triton 环境特定，RFC 明确排除 |

---

## 6. CI 配置

修改 `.github/ci/unit-tests-job.yaml`，在单元测试阶段新增：

```yaml
python3 tests/unit/gla_compare_test.py -v
```

需确保 CI 环境中 `DUMP_DIR` 已配置。

---

## 7. 涉及文件清单

| 操作 | 文件路径 | 行数 |
|------|---------|------|
| 新增 | `src/maxtext/kernels/gla/__init__.py` | 46 |
| 新增 | `src/maxtext/kernels/gla/pallas.py` | 296 |
| 新增 | `src/maxtext/layers/attention_gla.py` | 268 |
| 修改 | `src/maxtext/layers/normalizations.py` | +49 |
| 新增 | `tests/unit/gla_compare_test.py` | 487 |
| 修改 | `pyproject.toml` | +3 |
| 修改 | `.github/ci/unit-tests-job.yaml` | +170 |
| 修改 | `.github/workflows/unit_tests.yml` | +23 |

---

## 8. 前置依赖

- **PR1（配置扩展）**：GLA 相关配置字段（`group_norm_size`、`use_linear_silu`、`linear_attn_type`、`linear_conv_kernel_dim`）需先合入 `types.py` 和 `base.yml`

---

## 9. 不在本 PR 范围

| 内容 | 归属 PR |
|------|--------|
| `DecoderBlockType` 枚举新增 | PR6 (AL Model) |
| `decoders.py` 注册 decoder block | PR6 |
| `al_model.py` HybridAttention 调度 | PR6 |
| 配置字段 `types.py` / `base.yml` | PR1 |
| Argus dump hooks（`@dumpable`/`dump_tensor`） | 上游独立目录 |

---

## 10. 验证计划

1. **导入验证**：`import maxtext.kernels.gla` 正常工作
2. **端到端精度**：`DUMP_DIR=... python3 tests/unit/gla_compare_test.py -v`

---

## 11. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-03-31 | RFC 初始版本，状态 Final |
| 2026-04-02 | 同步 ant-pretrain 最新变更：fused chunk GLA kernels、checkpoint_name、jax.named_scope；tops 改为必选依赖；删除 gla_kernels_test.py；删除 naive.py（纯 JAX fallback）；添加 CI 配置；状态改为 Implemented |
