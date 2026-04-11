# Phase 1 架构设计：平台化改造

> **日期:** 2026-04-09
> **状态:** Approved
> **范围:** BenchmarkSession 元数据重构 / 多 K8s 任务调度 / Profile 数据存储与展示 / 配置捕获与解析

---

## 设计目标

将 Falcon 从一个 MCP 工具集合，改造为以 **BenchmarkSession** 为核心的性能分析平台：

1. **BenchmarkSession 元数据重构** — 统一数据模型，引入生命周期状态机，支持多种 Profile 产物
2. **多 K8s 任务调度与管理** — Falcon 作为编排者，通过 xpk 向多个 GKE 集群提交 benchmark job
3. **Profile 数据存储与展示** — DB + GCS 分离存储，CLI/MCP 优先的展示层，预留 Web 接口
4. **配置捕获与解析** — Schema-less 配置存储，两阶段捕获（意图 + 实际），原始配置完整保存

---

## Topic 1: BenchmarkSession 元数据重构

### 生命周期状态机

Session 从创建到归档有明确的状态流转：

```text
PENDING → PROVISIONING → RUNNING → COLLECTING → COMPLETED → ANALYZING → ARCHIVED
                                                                │
任何阶段均可 → FAILED / CANCELLED ◄─────────────────────────────┘
```

| 状态 | 含义 | 触发条件 |
|------|------|---------|
| `PENDING` | Session 已创建，等待调度 | 用户提交 benchmark 请求 |
| `PROVISIONING` | 正在申请 TPU/GPU 资源 | xpk workload create 已执行 |
| `RUNNING` | Benchmark job 正在执行 | Pod 进入 Running 状态 |
| `COLLECTING` | Job 完成，正在收集数据 | Job succeeded/failed |
| `COMPLETED` | 数据收集完毕，可分析 | Metrics + profile artifacts 已入库 |
| `ANALYZING` | 正在执行自动分析 | 触发 XProf/LLO 等分析器 |
| `ARCHIVED` | 分析完成，Session 归档 | 所有分析器完成 |
| `FAILED` | 失败终态 | 任何阶段发生不可恢复错误 |
| `CANCELLED` | 取消终态 | 用户主动取消 |

### 数据模型

将 `RunRecord` 重构为 `BenchmarkSession`，拆分为职责清晰的子结构：

```python
class SessionStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    ANALYZING = "analyzing"
    ARCHIVED = "archived"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SessionSource:
    """Session 的来源信息"""
    author: str                        # 提交人
    repo: str                          # "primatrix/ant-pretrain"
    branch: str | None
    commit: str | None
    pr: int | None
    trigger: str                       # "manual" | "ci" | "scheduled"
    workflow_run_id: int | None
    run_number: int | None

@dataclass
class SessionTarget:
    """Session 的目标运行环境"""
    cluster: str                       # 集群名 "tpu-v4-prod"
    accelerator: str                   # "tpu-v4-64" | "gpu-a100-8"
    device_type: str                   # "tpu" | "gpu"
    device_num: int

class WorkloadType(str, Enum):
    """Workload 类型 — 决定 Config 和 Metrics 的具体类型"""
    TRAINING = "training"
    INFERENCE = "inference"
    SERVING = "serving"
    RL = "rl"
    KERNEL = "kernel"

# ── 配置模型（Schema-less） ────────────────────────────────

@dataclass
class RawConfig:
    """原始配置 — 完整保存用户提交的配置，用于复现和审计"""
    cli_args: list[str]                    # 原始命令行参数 ["--tp=4", "--dp=8", ...]
    config_files: dict[str, str]           # {filename: content} 如 {"base.yml": "...", "override.yml": "..."}
    env_overrides: dict[str, str]          # {"XLA_FLAGS": "--xla_tpu_..."}
    effective_config: dict | None = None   # Job 完成后回收的实际生效配置（resolved）

@dataclass
class SessionConfig:
    """运行配置 — workload_type 分类 + 自由存储"""
    workload_type: WorkloadType            # 保留枚举，用于分类和路由
    config: dict                           # 扁平化关键配置 {"tp": 4, "dp": 8, "model": "llama-70b", ...}
    raw: RawConfig                         # 原始配置完整保存
    extra: dict | None = None              # 保留扩展口

# ── 分场景 Metrics ─────────────────────────────────────────

@dataclass
class TrainingMetrics:
    """训练聚合指标"""
    step_time_ms: float | None = None
    tflops_per_device: float | None = None
    mfu: float | None = None
    tokens_per_sec: float | None = None
    peak_memory_gb: float | None = None
    compile_time_s: float | None = None
    loss: float | None = None

@dataclass
class InferenceMetrics:
    """离线推理指标"""
    prefill_time_ms: float | None = None
    decode_time_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    tokens_per_sec: float | None = None
    peak_memory_gb: float | None = None
    batch_throughput: float | None = None

@dataclass
class ServingMetrics:
    """在线服务指标"""
    p50_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_qps: float | None = None
    tokens_per_sec: float | None = None
    peak_memory_gb: float | None = None

@dataclass
class RLMetrics:
    """强化学习指标"""
    step_time_ms: float | None = None
    tflops_per_device: float | None = None
    mfu: float | None = None
    tokens_per_sec: float | None = None
    peak_memory_gb: float | None = None
    reward_mean: float | None = None
    reward_std: float | None = None
    kl_divergence: float | None = None

@dataclass
class KernelMetrics:
    """Kernel benchmark 指标"""
    kernel_time_us: float | None = None
    flops: float | None = None
    bandwidth_gb_s: float | None = None
    roofline_pct: float | None = None
    peak_memory_gb: float | None = None

WorkloadMetrics = TrainingMetrics | InferenceMetrics | ServingMetrics | RLMetrics | KernelMetrics

@dataclass
class ErrorInfo:
    """错误信息"""
    error_class: str                   # "hbm_oom" | "gke_auth" | "tpu_timeout" | ...
    summary: str
    log_snippet: str | None

@dataclass
class SessionResult:
    """实测结果 — metrics 类型与 workload_type 对齐"""
    metrics: WorkloadMetrics            # 按场景分发
    raw_metrics: list[dict] | None = None  # 原始指标序列（训练=step、kernel=iteration、serving=request）
    error: ErrorInfo | None = None

@dataclass
class ProfileArtifact:
    """一种 profile 的产物"""
    artifact_id: str
    type: str                          # "xplane" | "llo" | "hlo" | "trace" | "tensorboard"
    uri: str                           # GCS path
    size_bytes: int | None
    collected_at: str
    metadata: dict                     # type-specific 元数据

@dataclass
class AnalysisResult:
    """分析工具对某个 artifact 的分析结果"""
    analysis_id: str
    analyzer: str                      # "xprof" | "llo_analyzer" | "roofline" | "memory"
    artifact_id: str
    summary: dict                      # 分析器特定的结构化结果
    created_at: str

@dataclass
class ProfileData:
    """一次 Session 产生的所有 profile 数据"""
    artifacts: list[ProfileArtifact]   # 多种 profile 产物
    analyses: list[AnalysisResult]     # 挂载的分析结果

@dataclass
class Prediction:
    """仿真预测 + 与实测的偏差"""
    step_time_ms: float | None
    mfu: float | None
    total_memory_gb: float | None
    bottleneck: str | None
    time_breakdown: dict | None
    memory_breakdown: dict | None
    delta: dict | None

@dataclass
class Collaboration:
    """协作信息"""
    optimization_goal: str | None
    parent_session_id: str | None
    related_sessions: list[str]
    related_pr: int | None
    tags: list[str]
    notes: str | None

@dataclass
class BenchmarkSession:
    """Falcon 核心实体 — 一次 benchmark 运行的完整上下文"""
    # 身份
    session_id: str                    # "bench-20260409-abc12345"
    workload_type: WorkloadType        # training | inference | serving | rl | kernel
    status: SessionStatus
    created_at: str
    updated_at: str
    tags: list[str]                    # 自由标签（"e2e" / "operator" / "alignment" 等）

    # 上下文
    source: SessionSource
    target: SessionTarget
    config: SessionConfig

    # K8s Job
    job: JobInfo | None                # 见 Topic 2

    # 结果
    result: SessionResult | None

    # Profile 数据（多种产物 + 多种分析结果）
    profile: ProfileData | None

    # 仿真
    prediction: Prediction | None

    # 协作
    collaboration: Collaboration
```

### ID 生成规则

```text
session_id:  "bench-{YYYYMMDD}-{8hex}"
artifact_id: "art-{session_id_suffix}-{type}-{4hex}"
analysis_id: "ana-{session_id_suffix}-{analyzer}-{4hex}"
job_id:      "job-{session_id_suffix}-{4hex}"
```

### 类型注册表

```python
# Config 已改为 Schema-less (dict)，无需注册表
# Metrics 仍保留强类型注册表 — 指标字段稳定，需要数值查询和排序

WORKLOAD_METRICS_REGISTRY: dict[WorkloadType, type] = {
    WorkloadType.TRAINING: TrainingMetrics,
    WorkloadType.INFERENCE: InferenceMetrics,
    WorkloadType.SERVING: ServingMetrics,
    WorkloadType.RL: RLMetrics,
    WorkloadType.KERNEL: KernelMetrics,
}
```

### 向后兼容

旧 `RunRecord` (JSONL) 到新 `BenchmarkSession` 的迁移通过 `from_run_record()` 类方法实现无损转换，旧记录直接标记为 `ARCHIVED` 状态。

迁移映射规则：

| 旧字段 | 迁移策略 |
|--------|---------|
| `type: "e2e"` | `workload_type: "training"`, `tags: ["e2e"]` |
| `type: "operator"` | `workload_type: "kernel"`, `tags: ["operator"]` |
| `type: "alignment"` | `workload_type: "training"`, `tags: ["alignment"]` |
| `config.parallelism` | 展平到 `config.config` 的各并行维度字段 |
| `result.steps` | 转为 `result.raw_metrics` |
| 无对应字段 | `config.raw` 置空（旧记录无原始配置） |

---

## Topic 2: 基于多个 K8s 的任务调度与管理

### 架构概览

```text
                       Falcon Platform
                            │
                     ┌──────┴──────┐
                     │ JobScheduler │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        XpkBackend    XpkBackend    XpkBackend
              │             │             │
         Cluster A     Cluster B     Cluster C
        (TPU v4-128)  (TPU v5e-256) (GPU A100×8)
```

Falcon 作为编排者，通过统一的 `JobScheduler` 接口向多个 GKE 集群提交 benchmark job。所有集群统一使用 `xpk` 调度 — xpk 同时支持 TPU（`--tpu-type`）和 GPU（`--device-type`），无需引入 kubectl 作为额外调度后端。

### 集群注册与配置

扩展 `repos.yaml`，新增 `clusters` 段：

```yaml
clusters:
  tpu-v4-prod:
    provider: gke
    project: primatrix-tpu
    zone: us-central2-b
    cluster_id: tpu-v4-cluster
    accelerator_types:
      - type: tpu-v4-64
        priority: 1                # 此集群是 tpu-v4-64 的首选
      - type: tpu-v4-128
        priority: 1
    default_namespace: default
    gcs_bucket: gs://falcon-tpu-v4
    max_queue_depth: 5             # 队列深度阈值，QUEUED workload 超过此值则跳过
    probe_timeout_s: 15            # xpk probe 命令超时（秒）
    labels:
      env: prod
      team: pretrain

  tpu-v5e-dev:
    provider: gke
    project: primatrix-dev
    zone: us-east1-c
    cluster_id: tpu-v5e-cluster
    accelerator_types:
      - type: tpu-v5e-256
        priority: 1
    default_namespace: default
    gcs_bucket: gs://falcon-tpu-v5e
    max_queue_depth: 3
    probe_timeout_s: 15
    labels:
      env: dev

  gpu-a100:
    provider: gke
    project: primatrix-gpu
    zone: us-central1-a
    cluster_id: gpu-cluster
    accelerator_types:
      - type: nvidia-a100-80g
        priority: 1
    default_namespace: benchmark
    gcs_bucket: gs://falcon-gpu
    max_queue_depth: 5
    probe_timeout_s: 15
    labels:
      env: prod
      team: gpu
```

### 调度引擎

### 集群选择策略（Probe + 优先级）

用户只需指定 accelerator 类型，Falcon 自动选择最优集群：

```text
用户请求: accelerator="tpu-v4-128"
│
├─ 1. 查找候选集群
│     从 clusters 配置中筛选 accelerator_types 包含 tpu-v4-128 的集群
│     按 priority 排序 → [tpu-v4-prod(p=1), tpu-v5e-backup(p=2)]
│
├─ 2. Probe 首选集群
│     ├─ 可达性: xpk workload list --cluster=... (超时 probe_timeout_s)
│     ├─ 队列深度: 统计 QUEUED 状态的 workload 数量
│     │   如果 queue_depth > cluster.max_queue_depth → 跳过
│     └─ 通过 → 选中此集群
│
├─ 3. 首选失败/队列满 → probe 下一个候选
│
├─ 4. 所有候选不可用 → 返回错误
│     error_class: "no_available_cluster"
│
└─ 5. 记录 SchedulingDecision 到 JobInfo（用于审计）
```

```python
@dataclass
class ClusterProbeResult:
    """集群探测结果"""
    cluster: str
    reachable: bool
    queue_depth: int | None        # QUEUED workload 数量
    running_count: int | None      # RUNNING workload 数量
    error: str | None
    probed_at: str

@dataclass
class SchedulingDecision:
    """调度决策记录 — 持久化到 JobInfo 用于审计"""
    requested_accelerator: str
    candidates: list[str]          # 候选集群名称列表
    selected: str                  # 选中的集群
    reason: str                    # 选择原因
    probe_results: list[ClusterProbeResult]
    decided_at: str

```

### 日志收集（轮询驱动 + TTL 安全窗口 + 智能多 Pod 策略）

#### 核心约束

```text
poll_interval  <  pod_ttl
    30s        <   600s (xpk 默认 ttlSecondsAfterFinished)

安全余量: 600 - 30 = 570s
```

`JobScheduler` 初始化时强制校验 `poll_interval < pod_ttl`，确保 job 完成后有足够时间收集日志。

#### 收集流程

```text
scheduler.poll() 循环 (每 30s)
│
├─ xpk workload list → 检查 workload 状态
│
├─ 状态: QUEUED/RUNNING → 记录心跳, 继续轮询
│
├─ 状态: COMPLETED / FAILED
│   │
│   ├─ Session: RUNNING → COLLECTING
│   │
│   ├─ Step 1: 收集日志（智能多 Pod 策略）
│   │   ├─ 获取 job 关联的所有 pod 列表
│   │   │   kubectl get pods -l xpk.google.com/workload={workload_name}
│   │   │
│   │   ├─ 判断收集范围
│   │   │   ├─ Job COMPLETED → 只收集 rank-0 pod 日志
│   │   │   └─ Job FAILED    → 收集所有 pod 日志（用于排查）
│   │   │
│   │   ├─ 拉取日志
│   │   │   kubectl logs {pod_name} (对每个需要收集的 pod)
│   │   │
│   │   └─ 上传到 GCS
│   │       gs://falcon-{cluster}/profiles/{session_id}/logs/
│   │         ├─ rank-0.log            (必定收集)
│   │         ├─ rank-1.log ... rank-N.log  (仅失败时)
│   │         └─ collection_meta.json  (收集元数据)
│   │
│   ├─ Step 2: 从 rank-0 日志解析 metrics
│   │   MetricsParser.parse(log_content, workload_type)
│   │
│   ├─ Step 3: 扫描 GCS profile 目录
│   │   gcloud storage ls → 发现 xplane/, llo/, hlo/ artifacts
│   │
│   └─ Step 4: Session → COMPLETED / FAILED
│
└─ 状态: 未知/超时 → 记录 warning, 继续轮询
```

#### Pod 发现与 Rank 识别

```python
@dataclass
class PodInfo:
    name: str
    rank: int              # 从 pod label 或 hostname 解析
    status: str            # Running / Succeeded / Failed
    exit_code: int | None

@dataclass
class CollectionResult:
    logs: dict[str, str]   # {"rank-0": "...", "rank-1": "..."}
    meta: dict             # 收集元数据
    rank0_log: str | None  # rank-0 日志（用于 metrics 解析）

class LogCollector:
    """日志收集器"""
    ...
```

#### GCS 日志目录结构

```text
gs://falcon-{cluster}/profiles/{session_id}/logs/
  ├─ rank-0.log              # 主 worker 日志（必定收集）
  ├─ rank-1.log              # 仅 job 失败时收集
  ├─ ...
  ├─ rank-N.log              # 仅 job 失败时收集
  └─ collection_meta.json    # 收集元数据：策略、时间、pod 状态
```

### Session ↔ Job 完整生命周期

```text
用户: "在 v4-128 上跑 EP8 DP4 FSDP4 profiling"
│
├─ 1. 创建 Session (PENDING)
│     SessionStore.create(session)  →  DB INSERT
│
├─ 2. 构建 JobSpec
│     cluster="tpu-v4-prod", accelerator="tpu-v4-128"
│     docker_image="gcr.io/.../ant-pretrain:latest"
│     command=["python", "train.py", "--profile", ...]
│     env={PROFILE_OUTPUT: "gs://falcon-tpu-v4/profiles/{session_id}/"}
│
├─ 3. scheduler.submit(session, spec)
│     xpk workload create ...
│     Session: PENDING → PROVISIONING
│     DB UPDATE: status, job info
│
├─ 4. scheduler.poll(session)
│     xpk workload list → status=running
│     Session: PROVISIONING → RUNNING
│
├─ 5. scheduler.poll(session)
│     Job succeeded
│     Session: RUNNING → COLLECTING
│
├─ 6. scheduler.collect(session)
│     ├── 获取 pod logs → 解析 metrics
│     ├── 扫描 GCS → 发现 xplane/, llo/, hlo/ artifacts
│     └── Session: COLLECTING → COMPLETED
│
├─ 7. 触发自动分析 (可选)
│     ├── XProfAnalyzer.analyze(session, artifact)
│     ├── LLOAnalyzer.analyze(session, artifact)
│     └── Session: COMPLETED → ANALYZING → ARCHIVED
│
└─ 8. 返回给用户
      "Session bench-20260409-a1b2c3d4 已完成
       step_time: 2.3s, MFU: 45.2%, memory: 28.6GB
       3 个 profile artifacts 可用 (xplane, llo, hlo)"
```

---

## Topic 3: Profile 数据的存储与展示

### 存储架构

```text
┌───────────────────────────────────────────────────┐
│                  Storage Layer                    │
│                                                   │
│  ┌────────────────┐     ┌───────────────────────┐ │
│  │  PostgreSQL    │     │        GCS            │ │
│  │                │     │                       │ │
│  │  sessions      │─ref─│  gs://falcon-{cluster}│ │
│  │  jobs          │     │    /profiles/          │ │
│  │  artifacts (meta)    │      /{session_id}/    │ │
│  │  analyses      │     │        xplane/         │ │
│  │  clusters      │     │        llo/            │ │
│  │  goals         │     │        hlo/            │ │
│  │                │     │        trace/           │ │
│  │  元数据+索引   │     │        logs/            │ │
│  └────────────────┘     └───────────────────────┘ │
└───────────────────────────────────────────────────┘
```

**设计原则：**

- 元数据（Session、Job、Artifact 索引、Analysis 结果）存 PostgreSQL，支持复杂查询
- 大文件（XPlane trace、LLO dump、HLO、训练日志）存 GCS，通过 URI 引用
- Profile artifacts 在 DB 中只存元数据（type、uri、size），实际文件在 GCS
- Analysis 结果是结构化数据，直接存 DB（JSONB）

### 数据库 Schema

```sql
-- 核心 Session 表
CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    workload_type TEXT NOT NULL CHECK (workload_type IN (
        'training', 'inference', 'serving', 'rl', 'kernel'
    )),
    status        TEXT NOT NULL CHECK (status IN (
        'pending', 'provisioning', 'running', 'collecting',
        'completed', 'analyzing', 'archived', 'failed', 'cancelled'
    )),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    tags          TEXT[] DEFAULT '{}', -- 自由标签（"e2e" / "operator" / "alignment" 等）
    source        JSONB NOT NULL,        -- {author, repo, branch, commit, pr, trigger, ...}
    target        JSONB NOT NULL,        -- {cluster, accelerator, device_type, device_num}
    config        JSONB NOT NULL,        -- {workload_type, config: {...}, raw: {...}, extra}
    result        JSONB,                 -- {metrics: {...}, raw_metrics: [...], error}
    prediction    JSONB,                 -- {predicted, delta}
    collaboration JSONB NOT NULL DEFAULT '{}'
);

-- 常用查询索引
CREATE INDEX idx_sessions_status        ON sessions(status);
CREATE INDEX idx_sessions_created       ON sessions(created_at DESC);
CREATE INDEX idx_sessions_workload_type ON sessions(workload_type);
CREATE INDEX idx_sessions_author        ON sessions((source->>'author'));
CREATE INDEX idx_sessions_repo          ON sessions((source->>'repo'));
CREATE INDEX idx_sessions_cluster       ON sessions((target->>'cluster'));
CREATE INDEX idx_sessions_config        ON sessions USING GIN(config);
CREATE INDEX idx_sessions_tags          ON sessions USING GIN(tags);

-- K8s Job 表
CREATE TABLE jobs (
    job_id         TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    cluster        TEXT NOT NULL,
    namespace      TEXT NOT NULL,
    job_name       TEXT NOT NULL,
    xpk_workload   TEXT,
    status         TEXT NOT NULL,
    submitted_at   TIMESTAMPTZ,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    attempt        INT NOT NULL DEFAULT 1,
    exit_code      INT,
    resource_usage JSONB,
    scheduling_decision JSONB    -- ClusterProbeResult 列表 + 选择原因
);

CREATE INDEX idx_jobs_cluster ON jobs(cluster);
CREATE INDEX idx_jobs_status  ON jobs(status);

-- Profile Artifacts 表（元数据在 DB，文件在 GCS）
CREATE TABLE profile_artifacts (
    artifact_id   TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    type          TEXT NOT NULL CHECK (type IN ('xplane', 'llo', 'hlo', 'trace', 'tensorboard')),
    uri           TEXT NOT NULL,          -- GCS path
    size_bytes    BIGINT,
    collected_at  TIMESTAMPTZ,
    metadata      JSONB DEFAULT '{}'
);

CREATE INDEX idx_artifacts_session ON profile_artifacts(session_id);
CREATE INDEX idx_artifacts_type    ON profile_artifacts(type);

-- Analysis Results 表
CREATE TABLE analysis_results (
    analysis_id   TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    artifact_id   TEXT REFERENCES profile_artifacts(artifact_id) ON DELETE CASCADE,
    analyzer      TEXT NOT NULL,
    summary       JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_analysis_session  ON analysis_results(session_id);
CREATE INDEX idx_analysis_analyzer ON analysis_results(analyzer);

-- Cluster 配置表
CREATE TABLE clusters (
    name              TEXT PRIMARY KEY,
    provider          TEXT NOT NULL,
    project           TEXT NOT NULL,
    zone              TEXT NOT NULL,
    cluster_id        TEXT NOT NULL,
    accelerator_types JSONB NOT NULL,
    gcs_bucket        TEXT NOT NULL,
    labels            JSONB DEFAULT '{}',
    config            JSONB DEFAULT '{}'
);

-- 优化目标表
CREATE TABLE optimization_goals (
    goal_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    owner         TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    claimed_at    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### GCS 目录规范

```text
gs://falcon-{cluster}/
  profiles/
    {session_id}/
      xplane/                       # XPlane protobuf 文件
        *.xplane.pb
      llo/                          # LLO IR dump
        *.llo
        *.llo.txt
      hlo/                          # HLO module dump
        module_*.hlo.txt
        module_*.hlo.pb
      trace/                        # Chrome trace format
        trace.json.gz
      tensorboard/                  # TensorBoard event 文件
        events.out.tfevents.*
      logs/                         # Pod 日志（智能收集）
        rank-0.log                  # 主 worker 日志（必定收集）
        rank-1.log ... rank-N.log   # 仅 job 失败时收集
        collection_meta.json        # 收集元数据：策略、时间、pod 状态
      metadata.json                 # 产物清单（与 DB 同步的冗余备份）
```

### 数据访问层

```python
class SessionStore:
    """Session CRUD + 查询 — CLI 和 MCP 的统一数据源"""
    ...

class ArtifactStore:
    """Profile 产物元数据管理"""
    ...

class AnalysisStore:
    """分析结果管理"""
    ...
```

### 展示层

#### CLI 命令

```bash
# Session 管理
falcon session list [--cluster X] [--author Y] [--workload training] [--status completed] [--last 7d]
falcon session get <session-id>
falcon session compare <id1> <id2>
falcon session trend --metric mfu [--author Y] [--last 30d]
falcon session stats

# 任务提交与管理（--workload 指定场景类型）
falcon session submit \
  --workload training \
  --cluster tpu-v4-prod \
  --accelerator tpu-v4-128 \
  --image gcr.io/.../ant-pretrain:latest \
  --config "tp=4,dp=8,fsdp=4,remat=full,seq_len=2048"

falcon session submit \
  --workload kernel \
  --cluster tpu-v4-prod \
  --accelerator tpu-v4-128 \
  --config "kernel_name=flash_attention,dtype=bf16,m=2048,n=2048"

falcon session submit \
  --workload serving \
  --cluster gpu-a100 \
  --accelerator nvidia-a100-80g \
  --image gcr.io/.../vllm-server:latest \
  --config "tp=4,engine=vllm,quantization=fp8"

falcon session cancel <session-id>

# Profile 数据
falcon profile list <session-id>
falcon profile download <artifact-id> .
falcon profile analyze <session-id> --analyzer xprof
falcon profile analyze <session-id> --analyzer llo

# 集群管理
falcon cluster list
falcon cluster status <cluster-name>

# Job 监控
falcon job list [--cluster X] [--status running]
falcon job logs <session-id>
falcon job cancel <session-id>

# 协作
falcon goal list
falcon goal claim <goal-name> --description "..."
falcon report weekly [--last 7d]
falcon report pr <pr-number>

# 数据迁移
falcon migrate import-jsonl <path-to-runs.jsonl>
```

#### MCP 工具扩展

在现有工具基础上新增：

| 工具 | 用途 | 示例 |
|------|------|------|
| `session_submit` | 提交 benchmark 到集群 | "在 v4-128 上跑 training EP8 DP4 profiling" / "在 v4-128 上跑 kernel flash_attention" |
| `session_status` | 查看 Session 完整状态 | "bench-xxx 现在什么状态" |
| `job_list` | 跨集群查看运行中的 job | "现在有哪些 job 在跑" |
| `job_cancel` | 取消 job | "取消 bench-xxx" |
| `cluster_list` | 查看可用集群和资源 | "有哪些集群可用" |
| `cluster_status` | 查看集群队列和资源使用 | "v4-prod 集群现在忙不忙" |
| `profile_list` | 列出 Session 的所有 profile 产物 | "bench-xxx 有哪些 profile 数据" |
| `profile_analyze` | 对指定 artifact 执行分析 | "分析 bench-xxx 的 LLO 数据" |

#### 预留 Web API（不在 Phase 1 实现）

```text
GET    /api/v1/sessions?author=X&cluster=Y&workload_type=training&status=completed
GET    /api/v1/sessions/{session_id}
POST   /api/v1/sessions
PATCH  /api/v1/sessions/{session_id}
GET    /api/v1/sessions/{session_id}/artifacts
GET    /api/v1/sessions/{session_id}/analyses
POST   /api/v1/sessions/{session_id}/analyze
GET    /api/v1/jobs?cluster=X&status=running
POST   /api/v1/jobs/{job_id}/cancel
GET    /api/v1/clusters
GET    /api/v1/clusters/{name}/status
```

---

## Topic 4: 配置捕获与解析

### 设计动机

实际 benchmark 任务中存在大量非规范化的自定义配置：训练框架配置文件（YAML）、命令行参数、环境变量覆盖等。不同 benchmark 的参数集合差异大，强类型 Config 无法覆盖所有场景。

**核心策略：** Config 采用 Schema-less 存储（JSONB dict），Metrics 保留强类型。配置在两个阶段捕获 — 提交时记录用户意图，job 完成后回收实际生效配置。

### 两阶段捕获流程

```text
阶段 1: 提交时捕获（用户意图）
─────────────────────────────
falcon session submit \
  --workload training \
  --accelerator tpu-v4-128 \
  --config-file base.yml \              ← 读取文件内容存入 raw.config_files
  --config-file override.yml \
  --env "XLA_FLAGS=--xla_tpu_..." \     ← 存入 raw.env_overrides
  -- python train.py \                  ← 「--」后面全部存入 raw.cli_args
    --tp=4 --dp=8 --fsdp=4 \
    --model_name=llama-70b \
    --per_device_batch_size=2

  ↓ ConfigExtractor 从 cli_args + config_files 中提取关键字段
  ↓ 写入 config.config = {"tp": 4, "dp": 8, "fsdp": 4, "model": "llama-70b", ...}

阶段 2: Job 完成后回收（实际生效）
───────────────────────────────
  ↓ 从 rank-0 日志中解析 resolved config（MaxText 启动时会打印）
  ↓ 或从 GCS 中回收 effective_config.json
  ↓ 写入 raw.effective_config = {...}
  ↓ 同步更新 config.config — 用 effective 中的 well-known keys 覆盖 intent 值
  ↓ 可选：对比 intent vs effective，标记 diff
```

### ConfigExtractor — 关键字段提取

`ConfigExtractor` 始终在提交阶段对**内存中的原始内容**执行提取，此时配置文件尚未上传到 GCS。GCS 大文件 URI 替换（> 64KB 时）发生在 `SessionStore.create()` 持久化阶段，即 `extract()` 之后。因此 `_parse_file` 不需要处理 GCS URI。

```python
class ConfigExtractor:
    """从原始配置中提取关键字段到扁平 dict"""

    # well-known keys — 提取时识别，但不强制存在
    WELL_KNOWN_KEYS = {
        # 并行度
        "tp", "dp", "pp", "ep", "fsdp", "cp",
        # 训练
        "per_device_batch_size", "seq_len", "grad_accum", "remat",
        # 模型
        "model_name", "model_config",
        # 推理
        "batch_size", "max_decode_len", "quantization", "engine",
        # Kernel
        "kernel_name", "dtype", "num_iterations",
    }
```

### Effective Config 回收

Job 完成后，`LogCollector` 在收集日志的同时回收实际生效的配置，并同步更新 `config.config` 中的关键字段，确保索引查询反映实际运行状态：

### GCS 配置目录结构

```text
gs://falcon-{cluster}/profiles/{session_id}/
  config/                              ← 新增
    submitted/                         # 提交时的原始配置
      base.yml                         # 原始配置文件副本
      override.yml
      cli_args.json                    # ["--tp=4", "--dp=8", ...]
      env_overrides.json               # {"XLA_FLAGS": "..."}
    effective/                         # Job 完成后回收
      resolved_config.json             # 实际生效的完整配置
      config_diff.json                 # intent vs effective 的差异
```

### DB 存储策略

`sessions` 表的 `config` 列保持 JSONB，内部结构变更：

```json
{
  "workload_type": "training",
  "config": {"tp": 4, "dp": 8, "fsdp": 4, "model_name": "llama-70b", "seq_len": 2048},
  "raw": {
    "cli_args": ["--tp=4", "--dp=8", "--fsdp=4", "--model_name=llama-70b"],
    "config_files": {"base.yml": "...内容..."},
    "env_overrides": {"XLA_FLAGS": "--xla_tpu_enable_async_collective_fusion=true"},
    "effective_config": null
  },
  "extra": null
}
```

当 `config_files` 总大小 > 64KB 时，文件内容只存 GCS，DB 中存 GCS URI 引用：

```json
"config_files": {"base.yml": "gs://falcon-tpu-v4/profiles/bench-.../config/submitted/base.yml"}
```

GIN 索引自动覆盖 `config` 字段的 JSONB 路径查询，无需额外索引变更。

### CLI 命令扩展

```bash
# 提交时带配置文件
falcon session submit \
  --workload training \
  --accelerator tpu-v4-128 \
  --config-file configs/base.yml \
  --config-file configs/large_model.yml \
  --env "XLA_FLAGS=--xla_tpu_enable_async_collective_fusion=true" \
  -- python train.py --tp=4 --dp=8 --profile

# 查看 session 的配置
falcon session config <session-id>                # 显示提取的关键字段
falcon session config <session-id> --raw          # 显示原始配置
falcon session config <session-id> --effective    # 显示实际生效配置
falcon session config <session-id> --diff         # 对比意图 vs 实际

# 按配置字段查询（JSONB 路径语法）
falcon session list --where "config.tp=4 AND config.dp>=8"
```

### MCP 工具扩展

| 工具 | 用途 | 示例 |
|------|------|------|
| `session_config` | 查看 Session 的配置详情 | "bench-xxx 用了什么配置" |
| `session_config_diff` | 对比意图配置与实际生效配置 | "bench-xxx 的配置有没有被覆盖" |
| `session_compare_config` | 对比两个 Session 的配置差异 | "这两次跑的配置有什么不同" |

---

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| DB | PostgreSQL 15+ | JSONB 原生支持、GIN 索引、成熟生态 |
| DB 客户端 | psycopg 3 | 同步模式，MCP server 单线程足够 |
| GCS | gcloud CLI | 已验证可用，避免引入额外依赖 |
| 调度 | xpk CLI | 团队已在使用 xpk，同时支持 TPU（`--tpu-type`）和 GPU（`--device-type`），统一调度后端 |
| CLI 框架 | click | 轻量、命令分组、Python 标准 |
| MCP | FastMCP | 已验证可用 |
| 配置解析 | PyYAML + tomllib | YAML 为主（MaxText），tomllib 为标准库内置 |

## 部署变更

MCP Server 容器新增 `DB_URL` 环境变量，连接 PostgreSQL：

- **开发/小团队:** K8s 内嵌 PostgreSQL StatefulSet + PVC
- **生产:** Cloud SQL for PostgreSQL + Auth Proxy sidecar

## 迁移策略

1. `falcon migrate init` — 创建表结构
2. `falcon migrate import-jsonl <path>` — 旧数据通过 `from_run_record()` 转换后导入
3. 可选双写过渡期 — 新数据同时写 DB 和 JSONL
4. 验证完整后停止 JSONL 写入

## 不在 Phase 1 范围

- Web Dashboard UI
- 分析工具插件化框架
- Claude Code Skill 定义
- 自动调度策略（根据队列深度选集群）
- 多租户和权限管理
- Session 自动清理和归档策略
