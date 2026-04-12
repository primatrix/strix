# Falcon 整体架构与模块分界

本文档对 Falcon 平台的整体架构设计以及各核心模块的职责分界进行了详细拆分与说明，作为项目开发的全局视图。通过明确“做什么”与“不做什么”，确保各层级与模块之间的解耦与高内聚。

## 整体架构图

Falcon 的系统架构遵循清晰的分层设计，各模块职责明确，通过定义良好的接口进行交互。

```text
┌──────────────────────────────────────────────────────────┐
│                     Falcon Platform                      │
│                                                          │
│  ┌───────────────────────────────────────────────┐       │
│  │    Agent Skills / CLI Workflows (AI 编排层)     │   ← 编排层     │
│  └───────────────────────┬───────────────────────┘       │
│                          │                               │
│  ┌────────┐  ┌───────────┴──────────┐  ┌──────────┐      │
│  │  CLI   │  │       MCP Tools      │  │ [Web API]│   ← 展示层     │
│  └───┬────┘  └───────────┬──────────┘  └────┬─────┘      │
│      │                   │                  │            │
│  ┌───┴───────────────────┴──────────────────┴───┐        │
│  │         Service Layer                        │   ← 业务逻辑层  │
│  │  SessionManager / JobScheduler               │        │
│  │  ProfileCollector / Analyzers                │        │
│  │  ComparisonEngine / ReportRenderer           │        │
│  └───┬─────────────┬─────────────┬──────────────┘        │
│      │             │             │                       │
│  ┌───┴───┐  ┌──────┴────┐  ┌────┴─────┐                  │
│  │Session│  │ Artifact  │  │ Analysis │   ← 数据访问层  │
│  │ Store │  │   Store   │  │   Store  │                  │
│  └───┬───┘  └─────┬─────┘  └────┬─────┘                  │
│      │            │             │                        │
│  ┌───┴────────────┴─────────────┴────┐                   │
│  │  PostgreSQL          GCS          │   ← 存储层       │
│  │  (元数据+索引)       (大文件)      │                   │
│  └───────────────────────────────────┘                   │
│                                                          │
│  ┌───────────────────────────────────┐                   │
│  │      Multi-Cluster Fabric         │   ← 集群编排层   │
│  │  ┌────────┐ ┌────────┐ ┌───────┐  │                   │
│  │  │GKE TPU │ │GKE TPU │ │GKE GPU│  │                   │
│  │  │  v4    │ │  v5e   │ │ A100  │  │                   │
│  │  └────────┘ └────────┘ └───────┘  │                   │
│  └───────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────┘
```

## 核心模块分界与职责

### 0. AI 编排层 (Agent Skills & Workflows Layer)

位于展示层之上，专门面向 AI Agent (如 Gemini、Claude Code) 或复杂人类工作流的场景编排。

**✅ 做什么 (Do):**

* **场景化指引 (Skills):** 提供给 AI Agent 的高阶使用说明（Prompt/System Instructions），例如“当用户要求定位内存泄漏时，应该按什么顺序调用哪些 MCP 工具”。
* **组合与编排:** 将基础的 CLI 命令或 MCP Tools 组合成完成特定大目标（如自动认领优化目标、执行对比并生成分析报告）的标准作业流程 (SOP)。
* **上下文感知:** 在特定上下文中提供必要的环境约束和背景知识，指导 Agent 做出正确决策。

**❌ 不做什么 (Don't):**

* **不包含底层实现:** 这一层全是自然语言指令、Markdown 文档或是简单的 Shell 脚本串联，不包含任何 Python 业务代码。
* **不替代基础工具:** 编排层依赖展示层 (MCP/CLI) 提供的一系列原子能力，不应该自己去直接解析日志或调用服务。

### 1. 展示层 (Presentation Layer)

负责处理用户交互，提供友好的操作界面。包含：

* **CLI (Command Line Interface):** 基于 `click` 框架，提供用户日常使用的命令行工具。包括：`session` 管理与提交、`goal` 比较与报告、`profile` 获取与深度分析等。
* **MCP Tools:** 基于 `FastMCP` 协议，提供 AI 友好的语义化接口（如 `session_submit`、`goal_report`、`deep_diff`），允许大语言模型自动理解和操作 Falcon 平台进行性能定位和分析。
* **Web API (规划中):** 预留的 RESTful 接口，为未来的 Web Dashboard 与可视化图表组件提供数据支持。

**✅ 做什么 (Do):**

* **解析输入:** 将用户输入 (命令行参数、MCP 协议语义) 转换为对内部 Service Layer 的标准方法调用。
* **格式化输出:** 负责结果的最终呈现与序列化 (CLI 表格渲染、终端颜色控制、Markdown 报告打印、JSON 响应)。
* **协议对接:** 处理特定协议的生命周期与身份认证 (如 FastMCP 的 Tool 注册、Web API 的鉴权)。

**❌ 不做什么 (Don't):**

* **不直接访问数据:** 绝对不能绕过业务逻辑层直接连接 PostgreSQL 或 GCS。
* **不包含复杂计算:** 不能在这一层编写配置差异比对、Metrics 分析等核心业务逻辑。
* **无状态:** 不要在展示层维护全局的业务状态变量。

### 2. 业务逻辑层 (Service Layer)

承载 Falcon 的核心业务规则、状态流转和编排计算。包含 SessionManager、JobScheduler、ProfileCollector、Analyzers、ComparisonEngine 等。

**✅ 做什么 (Do):**

* **状态机管理:** 维护 `BenchmarkSession` 生命周期的流转 (PENDING → PROVISIONING → RUNNING → COMPLETED 等)。
* **集群调度控制:** 根据队列深度、资源空闲度制定探测决策，通过底层编排接口提交任务，并执行轮询驱动的智能日志收集。
* **数据捕获与分析:** 在任务生命周期内捕获提交意图配置，回收实际生效配置；调用 `XProfAnalyzer` 或 `DeepProfileAnalyzer` 生成结构化分析摘要。
* **计算与对比:** 以 `OptimizationGoal` 为核心，执行无状态的指标对比、配置差异计算和性能回归检测（算子级 Diff、Roofline 演进等）。

**❌ 不做什么 (Don't):**

* **不感知存储细节:** 不能硬编码 SQL 语句，不能在业务逻辑中直接写 GCS 的底层连接代码，必须通过 Data Access Layer 进行交互。
* **不处理样式渲染:** 像 `ComparisonEngine` 只能输出结构化的 `dataclass` 结果，将样式渲染交给 `ReportRenderer` 或展现层处理。
* **不处理集群原生 API:** 不要直接在此层引入 `kubectl` 命令行细节解析，而是交由底层的 Cluster Layer 封装。

### 3. 数据访问层 (Data Access Layer)

封装对底层存储的读写操作，向业务层提供统一的数据模型抽象，包括 SessionStore、ArtifactStore、AnalysisStore。

**✅ 做什么 (Do):**

* **强类型抽象:** 将 DB 的元组/行数据映射为 Python 的强类型 `dataclass` 对象 (如 `BenchmarkSession`, `OptimizationGoal`)。
* **CRUD 与索引查询:** 提供实体的增删改查方法，封装 PostgreSQL 的事务，以及基于 JSONB/GIN 索引的复杂条件查询（例如通过嵌套配置项查询）。
* **透明的引用映射:** 屏蔽大文件存储位置，对应用层表现为统一的数据存取接口（记录 GCS URI 并在需要时透明拉取）。

**❌ 不做什么 (Don't):**

* **不包含业务流转逻辑:** 不能在这里决定“Job 失败时应该收集哪些日志”或者“分析结束后应该更改什么状态”。
* **不执行跨实体的业务合并计算:** 如对比分析两份 Profile 的差异，这应该在 Service Layer (ComparisonEngine) 中完成。

### 4. 存储层 (Storage Layer)

分离核心元数据关系与巨型 Profiling 产物，保证可扩展性与检索性能。

**✅ 做什么 (Do):**

* **PostgreSQL:** 存储具有强结构或高频搜索需要的元数据（Session 状态、Job 信息、Goal 目标关联以及分析结果概要）。利用 JSONB 存储灵活的配置字段。
* **GCS:** 作为不可变数据的安全挂载点，存放体积庞大的文件（XPlane Protobuf、LLO IR Dumps、HLO Modules、Chrome Traces、原始长日志及 YAML 资源副本）。

**❌ 不做什么 (Don't):**

* **不存放超大字段:** 数据库 (PostgreSQL) 中严禁直接存储超过数 MB 级别的大文本/大二进制对象，必须写入 GCS 并保留外键 URI。

### 5. 集群编排层 (Cluster Layer)

负责实际的计算资源供给与基准测试负载隔离执行。

**✅ 做什么 (Do):**

* **统一访问代理:** 提供跨可用区、多平台（TPU v4/v5e 与 GPU）集群的统一访问接口（如基于 xpk 的 `XpkBackend`）。
* **资源探测反馈:** 支持利用健康度 Probe 策略探测可用环境，返回队列深度与运行中 Job 状态，供上层进行流量分配。
* **任务生命周期代理:** 代理具体的 K8s Job 提交、取消、以及特定 Pod 的日志获取。

**❌ 不做什么 (Don't):**

* **不关心业务概念:** 这一层不应该知道什么是 `BenchmarkSession` 或 `OptimizationGoal`，它只接受标准化的 `JobSpec`。
* **不主动拉取配置:** 不去解析 Falcon 内部模型，只负责执行传入的指令与挂载卷策略。
