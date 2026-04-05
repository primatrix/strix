# 代码规范

本文定义 pallas-kernel 项目的代码风格、命名约定和组织规范。

---

## 1. 项目结构

```text
pallas-kernel/
├── src/tops/
│   ├── __init__.py
│   └── kernels/
│       ├── __init__.py
│       ├── matmul/
│       │   ├── __init__.py       # 公开 API 导出
│       │   └── pallas.py         # Pallas kernel 实现
│       └── gla/
│           ├── __init__.py
│           └── pallas.py
├── tests/
│   ├── conftest.py               # 共享 fixtures
│   ├── test_matmul.py
│   └── test_gla.py
├── benchmarks/
│   ├── matmul_bench.py
│   └── gla_bench.py
├── pyproject.toml
└── README.md
```

### 1.1 每个 Kernel 的文件组织

每个 kernel 在 `src/tops/kernels/` 下有独立子目录：

| 文件 | 职责 |
|------|------|
| `__init__.py` | 公开 API 导出，统一入口函数 |
| `pallas.py` | Pallas kernel 实现（`pallas_call`、`custom_vjp` 等） |

对应的测试和 benchmark 分别在 `tests/` 和 `benchmarks/` 顶层目录：

| 文件 | 命名规则 |
|------|---------|
| `tests/test_{kernel_name}.py` | 正确性测试 |
| `benchmarks/{kernel_name}_bench.py` | 性能测试 |

### 1.2 `__init__.py` 导出规则

每个 kernel 的 `__init__.py` 应导出面向用户的统一入口函数，隐藏内部实现细节：

```python
# src/tops/kernels/gla/__init__.py
from tops.kernels.gla.pallas import pallas_chunk_gla

def chunk_gla(q, k, v, g_gamma, *, scale=None, chunk_size=64):
    """GLA (Gated Linear Attention) chunk 前向/反向。"""
    return pallas_chunk_gla(q, k, v, g_gamma, scale=scale, chunk_size=chunk_size)
```

---

## 2. 命名约定

### 2.1 Kernel 函数

| 类别 | 格式 | 示例 |
|------|------|------|
| 公开入口 | `{op}_{variant}` | `chunk_gla`, `matmul_tiled` |
| Pallas 实现 | `pallas_{op}_{variant}` | `pallas_chunk_gla`, `pallas_matmul_tiled` |
| 前向实现 | `_{op}_fwd` | `_chunk_gla_fwd` |
| 反向实现 | `_{op}_bwd` | `_chunk_gla_bwd` |
| Kernel body | `_{op}_kernel` | `_matmul_tiled_kernel` |

### 2.2 参数命名

| 参数 | 命名 | 说明 |
|------|------|------|
| Block 大小 | `block_{dim}` 或 `B{DIM}` | `block_k`, `BK` |
| Grid 维度 | `grid_{dim}` | `grid_b`, `grid_h` |
| Scratch buffer | `{name}_scratch_ref` | `k_scratch_ref` |
| 信号量 | `sem_ref` 或 `sems` | `sems.at[0, buf]` |

### 2.3 类型别名

```python
from typing import TypeAlias

BlockSize: TypeAlias = int
DType: TypeAlias = jnp.dtype
```

---

## 3. 代码风格

### 3.1 工具链

| 工具 | 用途 | 配置位置 |
|------|------|---------|
| ruff | lint + format | `pyproject.toml` `[tool.ruff]` |
| mypy | 类型检查 | `pyproject.toml` `[tool.mypy]` |

### 3.2 JAX 风格要求

- **纯函数优先**：避免 side effects，所有状态通过参数和返回值传递
- **显式 dtype**：所有 `jnp.zeros` / `jnp.ones` 等调用必须指定 `dtype`
- **避免隐式广播**：使用 `jnp.newaxis`（或 `None` 索引）显式扩展维度

### 3.3 Docstring 要求

每个公开 kernel 函数必须包含以下内容：

```python
def chunk_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    *,
    scale: float | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunk-wise Gated Linear Attention (GLA) 前向计算。

    算法：将序列按 chunk_size 分块，在每个 chunk 内部计算线性注意力，
    chunk 间通过隐状态 h 传递信息。

    Args:
        q: Query tensor, shape [B, T, H, K]
        k: Key tensor, shape [B, T, H, K]
        v: Value tensor, shape [B, T, H, V]
        g_gamma: Per-head decay rate, shape [H]
        scale: 缩放因子，默认 1/sqrt(K)
        chunk_size: 分块大小，必须是 128 的倍数

    Returns:
        Output tensor, shape [B, T, H, V]

    性能特征：
        - compute-bound（MXU 为主）
        - Arithmetic Intensity: ~25.6 (K=128, chunk_size=512)
    """
```

---

## 4. Kernel 实现规范

### 4.1 自动微分

- 用于**训练**的 kernel 必须提供 `jax.custom_vjp` 以支持自动微分
- 纯**推理** kernel 可不要求
- Forward 和 Backward 必须分离实现

```python
@jax.custom_vjp
def pallas_chunk_gla(q, k, v, g_gamma, ...):
    return _chunk_gla_fwd(q, k, v, g_gamma, ...)[0]

pallas_chunk_gla.defvjp(_chunk_gla_fwd, _chunk_gla_bwd)
```

### 4.2 Magic Number

所有 magic number 必须提取为命名常量并注释来源：

```python
# MXU systolic array 维度，见 TPU v4 spec
MXU_DIM = 128

# 最小高效 MatMul 维度，见 performance-tpu.md Section 3.2
MIN_MATMUL_M = 512
```

### 4.3 硬件无关性

禁止硬编码芯片型号相关参数。不同 TPU 代次的差异（如 VMEM 大小）通过配置传入：

```python
# 正确
def _get_vmem_limit(tpu_version: str) -> int:
    ...

# 错误
VMEM_LIMIT = 128 * 1024 * 1024  # 不要硬编码
```

---

## 5. 依赖管理

### 5.1 运行时依赖

仅依赖 JAX 核心，不引入额外重型依赖：

```toml
[project]
dependencies = [
    "jax>=0.7.0",
]
```

### 5.2 开发依赖

通过 `optional-dependencies` 管理：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "mypy>=1.10",
]
```
