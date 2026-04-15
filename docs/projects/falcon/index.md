# Falcon: 性能分析平台

Falcon 是一个多层架构的性能分析平台。它通过标准化的实验流水线，将不同场景（算法实验、Benchmark、精度对齐、Profiling、Kernel Bench）统一管理，解决团队间 benchmark 不透明、工具和方法不统一的问题，并通过 CLI（支持 `--json` 结构化输出）提供对人类和 AI 友好的操作方式。

## 核心理念

Falcon 采用多层数据模型，场景驱动实验，实验产出制品：

```text
场景 (Benchmark / Profiling / 精度对齐 / ...)
  └─→ 实验 + 配置 (parallelism, remat, batch size, ...)
        └─→ 任务 (集群调度、资源分配)
              └─→ 制品 (XProf / Metrics / Log / Trace)
                    └─→ 数据分析 → 分析结果
```

实验是连接场景与任务的桥梁，完整记录每次运行的上下文：

```text
┌─────────────────────────────────────────────────┐
│              Experiment                         │
│                                                 │
│  WHO: author, commit, branch, PR                │
│  WHAT: parallelism config, remat, batch size    │
│  WHERE: cluster, accelerator, device_num        │
│  RESULT: step_time, tflops, MFU, memory         │
│  TRACE: xplane, llo, hlo, trace                 │
│  STATUS: pending → running → completed          │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 解决什么问题

| 痛点 | 现状 | Falcon 方案 |
|------|------|-------------|
| Benchmark 不透明 | 跑了什么、结果如何，只有跑的人知道 | 实验自动注册到共享 Registry，全团队可见 |
| 工具不统一 | 每人各自写脚本解析 profiling 数据 | 标准化 CLI（`--json` 结构化输出），统一查询/分析/对比 |
| 结果难复现 | 配置参数靠口头沟通或散落在 log 中 | 实验完整保存 config + env + result |
| 优化重复劳动 | 不知道别人是否跑过类似配置 | 查重、目标认领、优化链追踪 |
| 分析门槛高 | XProf/trace 分析需要专业知识 | AI 驱动的自然语言分析，降低使用门槛 |
| 跨集群不通 | 不同集群各自为政 | 多集群统一调度与管理 |

## 整体架构

详细的架构图与各模块职责分界，请参阅：[整体架构与模块分界](./architecture.md)

## 文档目录

- [集群编排层设计](./cluster-layer-design.md) — ClusterFabric 架构、xpk 任务生命周期、Cloud Logging 日志收集
