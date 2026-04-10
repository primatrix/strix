# Phase 1 架构设计：平台化改造

> **日期:** 2026-04-09
> **状态:** Approved
> **范围:** BenchmarkSession 元数据重构 / 多 K8s 任务调度 / Profile 数据存储与展示

---

## 设计目标

将 Falcon 从一个 MCP 工具集合，改造为以 **BenchmarkSession** 为核心的性能分析平台：

1. **BenchmarkSession 元数据重构** — 统一数据模型，引入生命周期状态机，支持多种 Profile 产物
2. **多 K8s 任务调度与管理** — Falcon 作为编排者，通过 xpk/kubectl 向多个 GKE 集群提交 benchmark job
3. **Profile 数据存储与展示** — DB + GCS 分离存储，CLI/MCP 优先的展示层，预留 Web 接口

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

@dataclass
class Parallelism:
    """并行策略维度"""
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1
    fsdp: int = 1
    cp: int = 1

    @property
    def total_devices(self) -> int:
        return self.tp * self.dp * self.pp * self.ep * self.fsdp * self.cp

@dataclass
class SessionConfig:
    """运行配置"""
    parallelism: Parallelism
    remat: str | None                  # "full" | "minimal" | None
    per_device_batch_size: int | None
    grad_accum: int | None
    seq_len: int | None
    model_config: dict | None          # 模型相关配置（可扩展）
    extra: dict | None                 # 其他配置项

@dataclass
class StepMetrics:
    """单步指标"""
    step: int
    seconds: float
    tflops: float | None
    tokens_per_sec: float | None
    loss: float | None

@dataclass
class Metrics:
    """聚合指标"""
    step_time_ms: float | None
    tflops_per_device: float | None
    mfu: float | None
    peak_memory_gb: float | None
    tokens_per_sec: float | None
    compile_time_s: float | None

@dataclass
class ErrorInfo:
    """错误信息"""
    error_class: str                   # "hbm_oom" | "gke_auth" | "tpu_timeout" | ...
    summary: str
    log_snippet: str | None

@dataclass
class SessionResult:
    """实测结果"""
    metrics: Metrics
    steps: list[StepMetrics]
    error: ErrorInfo | None

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
    type: str                          # "e2e" | "operator" | "alignment"
    status: SessionStatus
    created_at: str
    updated_at: str

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

### 向后兼容

旧 `RunRecord` (JSONL) 到新 `BenchmarkSession` 的迁移通过 `from_run_record()` 类方法实现无损转换，旧记录直接标记为 `ARCHIVED` 状态。

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
        XpkBackend    XpkBackend    KubectlBackend
              │             │             │
         Cluster A     Cluster B     Cluster C
        (TPU v4-128)  (TPU v5e-256) (GPU A100×8)
```

Falcon 作为编排者，通过统一的 `JobScheduler` 接口向多个 GKE 集群提交 benchmark job。调度后端分为 `xpk`（TPU 为主）和 `kubectl`（GPU 或通用场景）。

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
      - tpu-v4-64
      - tpu-v4-128
    default_namespace: default
    scheduling: xpk
    gcs_bucket: gs://falcon-tpu-v4
    labels:
      env: prod
      team: pretrain

  tpu-v5e-dev:
    provider: gke
    project: primatrix-dev
    zone: us-east1-c
    cluster_id: tpu-v5e-cluster
    accelerator_types:
      - tpu-v5e-256
    default_namespace: default
    scheduling: xpk
    gcs_bucket: gs://falcon-tpu-v5e
    labels:
      env: dev

  gpu-a100:
    provider: gke
    project: primatrix-gpu
    zone: us-central1-a
    cluster_id: gpu-cluster
    accelerator_types:
      - nvidia-a100-80g
    default_namespace: benchmark
    scheduling: kubectl
    gcs_bucket: gs://falcon-gpu
    labels:
      env: prod
      team: gpu
```

### 调度引擎

```python
class SchedulingBackend(ABC):
    """调度后端抽象"""

    @abstractmethod
    def submit(self, spec: JobSpec, cluster: ClusterConfig) -> JobInfo: ...

    @abstractmethod
    def poll(self, job: JobInfo, cluster: ClusterConfig) -> JobInfo: ...

    @abstractmethod
    def cancel(self, job: JobInfo, cluster: ClusterConfig) -> None: ...

    @abstractmethod
    def get_logs(self, job: JobInfo, cluster: ClusterConfig) -> str: ...


class XpkBackend(SchedulingBackend):
    """基于 xpk 的 TPU 任务调度"""

    def submit(self, spec: JobSpec, cluster: ClusterConfig) -> JobInfo:
        # xpk workload create \
        #   --cluster={cluster.cluster_id} \
        #   --project={cluster.project} \
        #   --zone={cluster.zone} \
        #   --workload={workload_name} \
        #   --tpu-type={spec.accelerator} \
        #   --num-slices={spec.num_slices} \
        #   --docker-image={spec.docker_image} \
        #   --command="{' '.join(spec.command)}" \
        #   --priority={spec.priority}
        ...


class KubectlBackend(SchedulingBackend):
    """基于 kubectl 的通用任务调度（GPU 等）"""

    def submit(self, spec: JobSpec, cluster: ClusterConfig) -> JobInfo:
        # 1. 渲染 Job YAML 模板
        # 2. kubectl apply --context={context} -f job.yaml
        ...


class JobScheduler:
    """统一调度入口"""

    def __init__(self, clusters: dict[str, ClusterConfig]):
        self.clusters = clusters
        self.backends: dict[str, SchedulingBackend] = {}
        for name, cfg in clusters.items():
            if cfg.scheduling == 'xpk':
                self.backends[name] = XpkBackend()
            elif cfg.scheduling == 'kubectl':
                self.backends[name] = KubectlBackend()

    def submit(self, session: BenchmarkSession, spec: JobSpec) -> JobInfo:
        """提交 job 并关联到 Session"""
        # Session: PENDING → PROVISIONING
        # 注入 session 元数据到 job labels
        # 调用对应后端 submit

    def poll(self, session: BenchmarkSession) -> JobInfo:
        """查询并更新 Session 状态"""
        # PROVISIONING → RUNNING (pod started)
        # RUNNING → COLLECTING (job finished)

    def cancel(self, session: BenchmarkSession) -> None:
        """取消 job，Session → CANCELLED"""

    def collect(self, session: BenchmarkSession) -> None:
        """Job 完成后收集结果"""
        # 1. 获取 pod logs → 解析 metrics
        # 2. 扫描 GCS → 收集 profile artifacts
        # 3. Session: COLLECTING → COMPLETED / FAILED

    def select_cluster(self, accelerator: str) -> ClusterConfig:
        """根据 accelerator 类型选择集群"""
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
    type          TEXT NOT NULL CHECK (type IN ('e2e', 'operator', 'alignment')),
    status        TEXT NOT NULL CHECK (status IN (
        'pending', 'provisioning', 'running', 'collecting',
        'completed', 'analyzing', 'archived', 'failed', 'cancelled'
    )),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source        JSONB NOT NULL,        -- {author, repo, branch, commit, pr, trigger, ...}
    target        JSONB NOT NULL,        -- {cluster, accelerator, device_type, device_num}
    config        JSONB NOT NULL,        -- {parallelism, remat, batch_size, seq_len, ...}
    result        JSONB,                 -- {metrics, steps[], error}
    prediction    JSONB,                 -- {predicted, delta}
    collaboration JSONB NOT NULL DEFAULT '{}'
);

-- 常用查询索引
CREATE INDEX idx_sessions_status     ON sessions(status);
CREATE INDEX idx_sessions_created    ON sessions(created_at DESC);
CREATE INDEX idx_sessions_type       ON sessions(type);
CREATE INDEX idx_sessions_author     ON sessions((source->>'author'));
CREATE INDEX idx_sessions_repo       ON sessions((source->>'repo'));
CREATE INDEX idx_sessions_cluster    ON sessions((target->>'cluster'));
CREATE INDEX idx_sessions_config     ON sessions USING GIN(config);
CREATE INDEX idx_sessions_tags       ON sessions USING GIN((collaboration->'tags'));

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
    resource_usage JSONB
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
    scheduling        TEXT NOT NULL,
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
      logs/                         # 训练原始日志
        training.log
        stderr.log
      metadata.json                 # 产物清单（与 DB 同步的冗余备份）
```

### 数据访问层

```python
class SessionStore:
    """Session CRUD + 查询 — CLI 和 MCP 的统一数据源"""

    def create(self, session: BenchmarkSession) -> None
    def update(self, session: BenchmarkSession) -> None
    def get(self, session_id: str) -> BenchmarkSession | None
    def delete(self, session_id: str) -> None

    def list(self, *, type=None, status=None, author=None,
             repo=None, cluster=None, accelerator=None,
             branch=None, tags=None, ep=None, dp=None,
             fsdp=None, since=None, last_n=20) -> list[BenchmarkSession]

    def compare(self, id1: str, id2: str) -> dict
    def trend(self, metric: str, **filters) -> list[dict]
    def stats(self) -> dict
    def find_similar(self, config: dict) -> list[BenchmarkSession]
    def import_from_jsonl(self, jsonl_path: str) -> int


class ArtifactStore:
    """Profile 产物元数据管理"""
    def add(self, artifact: ProfileArtifact) -> None
    def list_by_session(self, session_id: str) -> list[ProfileArtifact]
    def get(self, artifact_id: str) -> ProfileArtifact | None


class AnalysisStore:
    """分析结果管理"""
    def add(self, result: AnalysisResult) -> None
    def list_by_session(self, session_id: str) -> list[AnalysisResult]
    def list_by_analyzer(self, analyzer: str) -> list[AnalysisResult]
    def get(self, analysis_id: str) -> AnalysisResult | None
```

### 展示层

#### CLI 命令

```bash
# Session 管理
falcon session list [--cluster X] [--author Y] [--type e2e] [--status completed] [--last 7d]
falcon session get <session-id>
falcon session compare <id1> <id2>
falcon session trend --metric mfu [--author Y] [--last 30d]
falcon session stats

# 任务提交与管理
falcon session submit \
  --cluster tpu-v4-prod \
  --accelerator tpu-v4-128 \
  --image gcr.io/.../ant-pretrain:latest \
  --config "ep=8,dp=4,fsdp=4,remat=full"
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
| `session_submit` | 提交 benchmark 到集群 | "在 v4-128 上跑 EP8 DP4 profiling" |
| `session_status` | 查看 Session 完整状态 | "bench-xxx 现在什么状态" |
| `job_list` | 跨集群查看运行中的 job | "现在有哪些 job 在跑" |
| `job_cancel` | 取消 job | "取消 bench-xxx" |
| `cluster_list` | 查看可用集群和资源 | "有哪些集群可用" |
| `cluster_status` | 查看集群队列和资源使用 | "v4-prod 集群现在忙不忙" |
| `profile_list` | 列出 Session 的所有 profile 产物 | "bench-xxx 有哪些 profile 数据" |
| `profile_analyze` | 对指定 artifact 执行分析 | "分析 bench-xxx 的 LLO 数据" |

#### 预留 Web API（不在 Phase 1 实现）

```text
GET    /api/v1/sessions?author=X&cluster=Y&status=completed
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

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| DB | PostgreSQL 15+ | JSONB 原生支持、GIN 索引、成熟生态 |
| DB 客户端 | psycopg 3 | 同步模式，MCP server 单线程足够 |
| GCS | gcloud CLI | 已验证可用，避免引入额外依赖 |
| 调度 | xpk CLI + kubectl | 团队已在使用 xpk，kubectl 覆盖 GPU 场景 |
| CLI 框架 | click | 轻量、命令分组、Python 标准 |
| MCP | FastMCP | 已验证可用 |

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
