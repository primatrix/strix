# Benchmark 规范

本文标准化性能测试方法，确保 kernel 之间的性能数据可比较、可复现。

---

## 1. 方法论

### 1.1 Roofline Model

pallas-kernel 采用 **Roofline Bound** 级别的性能分析，即单算子物理极限分析。不涉及多设备并行的 System Bound 级别（那是上层框架的职责）。

Roofline Model 将每个算子定位在 compute-bound 或 memory-bound 区域：

```text
Performance     │          ╱  Peak FLOPS
(TFLOPS)        │        ╱
                │      ╱
                │    ╱ ← Roofline
                │  ╱
                │╱
                └───────────────────
                  Arithmetic Intensity (FLOP/Byte)
```

### 1.2 关键指标

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **MFU** | Model FLOPs Utilization | 实际 FLOPS / Peak FLOPS |
| **TFLOPS** | 每秒万亿浮点运算 | FLOPs / 耗时(s) / 1e12 |
| **HBM 带宽利用率** | HBM 带宽使用比例 | 实际带宽 / Peak 带宽 |
| **Arithmetic Intensity** | 算术强度 | FLOPs / 访存字节数 |

### 1.3 Arithmetic Intensity 计算

以 MatMul `[M, K] × [K, N]` 为例（bf16）：

```text
FLOPs = 2 × M × K × N
Bytes = (M × K + K × N + M × N) × 2  # bf16 = 2 bytes
AI = FLOPs / Bytes
```

AI 越高越趋向 compute-bound，越低越趋向 memory-bound。

---

## 2. Benchmark 编写规范

### 2.1 脚本结构

每个 benchmark 脚本遵循统一结构：

```python
"""MatMul Tiled Kernel Benchmark."""
import argparse
import json
import time

import jax
import jax.numpy as jnp

from tops.kernels.matmul import matmul_tiled


def benchmark_single(fn, *args, warmup: int = 5, repeats: int = 20):
    """统一 benchmark 框架：warmup → 多次运行 → 统计。"""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    mean = sum(times) / len(times)
    return {
        "mean_ms": mean * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": (sum((t - mean)**2 for t in times) / len(times)) ** 0.5 * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="1024,1024,1024")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    M, K, N = map(int, args.shape.split(","))
    dtype = getattr(jnp, args.dtype)

    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (M, K), dtype=dtype)
    b = jax.random.normal(key, (K, N), dtype=dtype)

    # Pallas kernel
    pallas_result = benchmark_single(matmul_tiled, a, b)

    # JAX native baseline
    native_result = benchmark_single(jnp.matmul, a, b)

    # Compute metrics
    flops = 2 * M * K * N
    pallas_tflops = flops / (pallas_result["mean_ms"] / 1000) / 1e12

    results = {
        "shape": [M, K, N],
        "dtype": args.dtype,
        "pallas": pallas_result,
        "native": native_result,
        "speedup": native_result["mean_ms"] / pallas_result["mean_ms"],
        "tflops": pallas_tflops,
    }

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

### 2.2 参数矩阵

每个 kernel 的 benchmark 应覆盖以下参数组合：

| 维度 | 典型值 |
|------|--------|
| 输入形状 | 小 (128²)、中 (1024²)、大 (4096²)、非对齐 (127, 513) |
| dtype | float32, bfloat16 |
| batch size | 1, 4, 16（如适用） |

### 2.3 输出格式

- **JSON**：机器可读，用于回归检测
- **表格**：人类可读，用于 PR 描述

---

## 3. Profiling 工具链

### 3.1 JAX Profiler

```python
# 标准 profiling 流程
output = kernel(...)
jax.block_until_ready(output)  # 预热

jax.profiler.start_trace("/path/to/profile")
for _ in range(3):
    output = kernel(...)
    jax.block_until_ready(output)
jax.profiler.stop_trace()
```

**关键注意**：profiling 必须在顶层 `jax.jit` 包裹下运行，否则会产生碎片化 HLO 模块，导致 xprof 分析失效。详见 [JAX TPU 显存分析指南](/onboarding/profiling-hbm-guide)。

### 3.2 xprof / TensorBoard

xprof 提供以下分析视图：

| 视图 | 用途 |
|------|------|
| **Trace Viewer** | 时间线分析，定位耗时最长的 HLO module |
| **Graph Viewer** | HLO 计算图可视化 |
| **Memory Viewer** | HBM/VMEM 峰值占用、buffer 生命周期 |

### 3.3 关键关注指标

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| Scalar ALU 利用率 | 过高说明标量计算是瓶颈 | 减少索引计算，使用 SMEM 查找表 |
| MXU 利用率 | 过低说明 MXU 未被充分利用 | 增大 tile 尺寸，减少 DMA 等待 |
| Vector Spills | 有 spill 说明向量寄存器溢出 | 减小 tile 尺寸 |
| Vector Store 利用率 | 接近 100% 说明写回是瓶颈 | 考虑异步写回 |
| DMA wait vs hbm_to_vmem 数量 | 不一致可能是多发起单等待的优化 | 正常现象 |

### 3.4 HBM 显存分析

当遇到 OOM 或需要优化内存占用时：

1. 使用 `XLA_FLAGS` dump HLO 和 buffer assignment
1. 查看 `buffer-assignment.txt` 中的峰值 HBM 占用
1. 在 xprof Memory Viewer 中定位最大的 Temporary buffer
1. 对大 buffer 考虑：减小精度、recomputation、避免 layout copy

完整指南参见 [JAX TPU 显存分析指南](/onboarding/profiling-hbm-guide)。

---

## 4. 性能回归检测

### 4.1 基线管理

每个 kernel 版本的 benchmark 结果记录在 `benchmarks/baselines/` 目录：

```text
benchmarks/baselines/
├── matmul_v0.1.0.json
├── matmul_v0.2.0.json
└── gla_v0.2.0.json
```

### 4.2 回归阈值

| 指标 | 允许波动 | 触发告警 |
|------|---------|---------|
| mean_ms | ±5% | 超过 5% 退化 |
| tflops | ±5% | 低于基线 95% |

### 4.3 CI 集成（可选）

在 TPU 集成测试中加入 benchmark 回归检测：

```python
def test_no_regression():
    baseline = load_baseline("matmul_v0.1.0.json")
    current = benchmark_single(matmul_tiled, a, b)
    assert current["mean_ms"] <= baseline["mean_ms"] * 1.05, \
        f"Performance regression: {current['mean_ms']:.2f}ms vs baseline {baseline['mean_ms']:.2f}ms"
```
