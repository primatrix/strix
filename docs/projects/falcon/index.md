# Falcon: Benchmark Session 性能分析平台

Falcon 是一个以 **Benchmark Session** 为核心的性能分析平台。它通过统一的数据模型保存每次 benchmark 运行的完整上下文（配置、指标、trace、错误信息），解决团队间 benchmark 不透明、工具和方法不统一的问题，并通过 CLI（支持 `--json` 结构化输出）提供对人类和 AI 友好的操作方式。

## 核心理念

每一次 benchmark 运行都是一个 **Session**，Session 是 Falcon 一切能力的基础单元。

```text
┌─────────────────────────────────────────────────┐
│              Benchmark Session                  │
│                                                 │
│  WHO: author, commit, branch, PR                │
│  WHAT: parallelism config, remat, batch size    │
│  WHERE: cluster, accelerator, device_num        │
│  RESULT: step_time, tflops, MFU, memory         │
│  TRACE: xplane, llo, hlo, trace                 │
│  STATUS: pending → running → completed          │
│  CONTEXT: optimization_goal, tags, notes        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 解决什么问题

| 痛点 | 现状 | Falcon 方案 |
|------|------|-------------|
| Benchmark 不透明 | 跑了什么、结果如何，只有跑的人知道 | Session 自动注册到共享 Registry，全团队可见 |
| 工具不统一 | 每人各自写脚本解析 profiling 数据 | 标准化 CLI（`--json` 结构化输出），统一查询/分析/对比 |
| 结果难复现 | 配置参数靠口头沟通或散落在 log 中 | Session 完整保存 config + env + result |
| 优化重复劳动 | 不知道别人是否跑过类似配置 | 查重、目标认领、优化链追踪 |
| 分析门槛高 | XProf/trace 分析需要专业知识 | AI 驱动的自然语言分析，降低使用门槛 |
| 跨集群不通 | 不同集群各自为政 | 多集群统一调度与管理 |

## 整体架构

Falcon 采用了清晰的分层架构（展示层、业务逻辑层、数据访问层、存储层、集群编排层），以实现良好的模块解耦和高度可扩展性。
详细的整体架构图与各个层次的核心模块职责分界，请参阅：

- [整体架构与模块分界](./architecture.md)

## 文档目录

- [集群编排层设计](./cluster-layer-design.md) — ClusterFabric 架构、xpk 任务生命周期、Cloud Logging 日志收集
