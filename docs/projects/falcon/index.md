# Falcon: Benchmark Session 性能分析平台

Falcon 是一个以 **Benchmark Session** 为核心的性能分析平台。它通过统一的数据模型保存每次 benchmark 运行的完整上下文（配置、指标、trace、错误信息），解决团队间 benchmark 不透明、工具和方法不统一的问题，并通过 MCP 协议和 CLI 提供对 AI 友好的操作方式。

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
| 工具不统一 | 每人各自写脚本解析 profiling 数据 | 标准化 MCP 工具 + CLI，统一查询/分析/对比 |
| 结果难复现 | 配置参数靠口头沟通或散落在 log 中 | Session 完整保存 config + env + result |
| 优化重复劳动 | 不知道别人是否跑过类似配置 | 查重、目标认领、优化链追踪 |
| 分析门槛高 | XProf/trace 分析需要专业知识 | AI 驱动的自然语言分析，降低使用门槛 |
| 跨集群不通 | 不同集群各自为政 | 多集群统一调度与管理 |

## 整体架构

```text
┌──────────────────────────────────────────────────────────┐
│                     Falcon Platform                      │
│                                                          │
│  ┌────────┐  ┌───────────┐  ┌──────────┐                │
│  │  CLI   │  │ MCP Tools │  │ [Web API]│   ← 展示层     │
│  └───┬────┘  └─────┬─────┘  └────┬─────┘                │
│      │             │              │                      │
│  ┌───┴─────────────┴──────────────┴───┐                  │
│  │         Service Layer              │   ← 业务逻辑    │
│  │  SessionManager / JobScheduler     │                  │
│  │  ProfileCollector / Analyzers      │                  │
│  └───┬─────────────┬─────────────┬────┘                  │
│      │             │             │                       │
│  ┌───┴───┐  ┌──────┴────┐  ┌────┴─────┐                 │
│  │Session│  │ Artifact  │  │ Analysis │   ← 数据访问层  │
│  │ Store │  │   Store   │  │   Store  │                  │
│  └───┬───┘  └─────┬─────┘  └────┬─────┘                 │
│      │            │             │                        │
│  ┌───┴────────────┴─────────────┴────┐                   │
│  │  PostgreSQL          GCS          │   ← 存储层       │
│  │  (元数据+索引)       (大文件)      │                   │
│  └───────────────────────────────────┘                   │
│                                                          │
│  ┌───────────────────────────────────┐                   │
│  │      Multi-Cluster Fabric         │   ← 集群层       │
│  │  ┌────────┐ ┌────────┐ ┌───────┐  │                   │
│  │  │GKE TPU │ │GKE TPU │ │GKE GPU│  │                   │
│  │  │  v4    │ │  v5e   │ │ A100  │  │                   │
│  │  └────────┘ └────────┘ └───────┘  │                   │
│  └───────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────┘
```

## 文档目录

- [Phase 1 架构设计](./phase1-design.md) — BenchmarkSession 元数据重构、多 K8s 任务调度、Profile 数据存储与展示
- [Phase 2 架构设计](./phase2-design.md) — Goal-Centric 对比系统、ComparisonEngine、自动触发与报告
- [Phase 3 架构设计](./phase3-design.md) — 深度 Profile 分析与跨 Session 对比、算子级回归检测、瓶颈迁移检测
