# pallas-kernel

pallas-kernel（PyPI 包名 `tops`）是 primatrix 团队的 TPU Pallas kernel 集合，为 MaxText、sglang-jax 等上层项目提供高性能、可测试、可复用的 Pallas kernel 实现。

## 核心目标

- **高性能**：每个 kernel 经过 Roofline 分析和 profiling 验证，逼近硬件理论上限
- **可测试**：每个 kernel 提供与 NumPy/PyTorch 参考实现的数值精度对比测试
- **可复用**：统一的 API 设计，支持通过 `pip install` 作为依赖引入

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.12 |
| JAX | ≥ 0.7.0 |
| Pallas API | `jax.experimental.pallas.tpu` |
| 构建工具 | uv |

## 目标 TPU 硬件

项目主要面向以下 TPU 代次，kernel 文档中需标注各代次的兼容性：

| TPU 代次 | 说明 |
|----------|------|
| v4 | 基础支持 |
| v5e | 推理优化 |
| v6e | 当前开发主力 |
| v7x | 高端训练 |

## 文档导航

| 文档 | 说明 |
|------|------|
| [硬件约束与 API 限制](./hardware-constraints) | TPU 硬件架构、Pallas API 硬性约束、内存对齐规则、性能陷阱 |
| [代码规范](./coding-standards) | 项目结构、命名约定、代码风格、kernel 实现规范 |
| [CI/CD 流水线](./ci-cd) | 持续集成阶段、TPU 集成测试、pre-commit 钩子、版本发布 |
| [开发流程](./development-workflow) | 分支策略、kernel 开发生命周期、PR 规范、测试要求 |
| [Benchmark 规范](./benchmarking) | Roofline 分析方法、benchmark 编写规范、profiling 工具链 |
| [Kernel 文档模板](./kernel-template) | 新增 kernel 时复制使用的标准化文档模板 |

## Quick Start

### 环境搭建

```bash
# 克隆仓库
git clone git@github.com:primatrix/pallas-kernel.git
cd pallas-kernel

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 运行测试

```bash
# CPU 模式单元测试
pytest tests/ -v

# TPU 上运行（需要 TPU 环境）
pytest tests/ -v --tpu
```

### 运行 Benchmark

```bash
python benchmarks/matmul_bench.py --shape 1024,1024,1024
```

## 与 onboarding 文档的关系

wiki 中的 [Pallas Kernel 编写经验总结](/onboarding/pallas-kernel-guide) 和 [TPU 性能优化指南](/onboarding/performance-tpu) 是面向所有新人的通用入门材料。本项目文档是面向 pallas-kernel 仓库开发者的**项目级规范**，包含项目特有的约束、流程和标准。两者互补，建议先阅读 onboarding 文档建立基础认知。
