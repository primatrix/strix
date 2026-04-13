# 集群编排层设计文档

> **日期:** 2026-04-13
> **状态:** Draft
> **范围:** 集群编排层 (Cluster Layer) 的完整设计 — 集群配置管理、健康探测、集群选择策略、xpk 任务生命周期、Cloud Logging 日志收集
> **前置:** [整体架构与模块分界](./architecture.md)

---

## 设计目标

为 Falcon 平台提供统一的多集群编排能力，屏蔽底层 GKE/xpk/Cloud Logging 的实现细节，向业务逻辑层（Service Layer）暴露简洁、稳定的接口。

**核心原则：**

1. **业务无感知** — 集群层不知道 `BenchmarkSession`、`OptimizationGoal` 等业务概念，只处理标准化的 `WorkloadSpec`（对应 architecture.md 中的 `JobSpec` 概念，此处使用 `WorkloadSpec` 命名以与 xpk 的 workload 术语保持一致）
2. **xpk 统一调度** — 所有集群操作通过 xpk CLI 完成，同时支持 TPU 和 GPU，不引入其他调度后端
3. **日志零内存加载** — Pod 日志通过 Cloud Logging API 流式读取或直接导出到 GCS，不在内存中存储完整日志
4. **探测驱动选择** — 集群选择基于实时健康探测结果，而非静态配置

---

## 整体架构

```text
业务逻辑层 (JobScheduler / SessionManager)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   ClusterFabric                      │
│              (集群编排层唯一对外入口)                   │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ClusterRegistry│  │ClusterProber │                  │
│  │  集群配置管理  │  │  健康探测     │                  │
│  └──────┬───────┘  └──────┬───────┘                  │
│         │                  │                          │
│  ┌──────┴──────────────────┴───────┐                  │
│  │       ClusterSelector           │                  │
│  │  (候选筛选 + probe + 优先级选择)  │                  │
│  └─────────────────────────────────┘                  │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────┐          │
│  │XpkWorkloadManager│  │CloudLogRetriever │          │
│  │  xpk 任务生命周期 │  │  Cloud Logging   │          │
│  │  提交/取消/查询   │  │  Pod 日志拉取    │          │
│  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────┘
```

`ClusterFabric` 是业务层与集群层的唯一交互点。内部按职责拆分为 5 个独立可测的子模块。

### 模块职责总览

| 模块 | 职责 | 外部依赖 |
|------|------|---------|
| `ClusterRegistry` | 加载/查询集群配置，按 accelerator 类型筛选候选集群 | `repos.yaml` (PyYAML) |
| `ClusterProber` | 对单个集群执行健康探测（队列深度、可达性） | `xpk` CLI |
| `ClusterSelector` | 组合 Registry + Prober，实现集群选择策略 | 无额外依赖 |
| `XpkWorkloadManager` | 封装 xpk CLI 的任务提交、取消、状态查询 | `xpk` CLI |
| `CloudLogRetriever` | 通过 Cloud Logging API 流式读取或导出 Pod 日志 | `google-cloud-logging` SDK |
| `ClusterFabric` | 对外门面，组合以上模块，提供统一接口 | 无额外依赖 |

---

## 数据模型

集群层的数据模型全部是不可变的值对象（`frozen=True`），不包含业务状态。

### 集群配置

```python
@dataclass(frozen=True)
class AcceleratorSlot:
    """集群支持的加速器类型及其优先级"""
    type: str          # "tpu-v4-64", "tpu-v5e-256", "nvidia-a100-80g"
    device_class: str  # "tpu" | "gpu" — 显式标识设备类别，驱动 xpk 参数选择
    priority: int      # 优先级，数值越小优先级越高（1 最优先），排序时升序

@dataclass(frozen=True)
class ClusterConfig:
    """单个集群的完整配置 — 从 repos.yaml 加载"""
    name: str                          # "tpu-v4-prod"
    provider: str                      # "gke"
    project: str                       # GCP project ID
    zone: str                          # "us-central2-b"
    cluster_id: str                    # GKE cluster name
    accelerator_slots: list[AcceleratorSlot]
    default_namespace: str             # "default"
    gcs_bucket: str                    # "gs://falcon-tpu-v4"
    max_queue_depth: int               # QUEUED workload 超过此值则跳过
    probe_timeout_s: int               # xpk probe 超时秒数
    labels: dict[str, str]             # 自由标签 {"env": "prod", "team": "pretrain"}
```

### 探测与选择结果

```python
@dataclass(frozen=True)
class ClusterProbeResult:
    """单个集群的健康探测结果"""
    cluster: str                       # 集群名称
    reachable: bool                    # 是否可达
    queue_depth: int | None            # QUEUED workload 数量
    running_count: int | None          # RUNNING workload 数量
    error: str | None                  # 失败原因（可达时为 None）
    probed_at: str                     # ISO 8601

@dataclass(frozen=True)
class ClusterSelection:
    """集群选择的完整决策记录 — 用于审计"""
    selected_cluster: str              # 选中的集群名称
    accelerator: str                   # 请求的 accelerator 类型
    candidates: list[str]             # 候选集群名称列表
    probe_results: list[ClusterProbeResult]  # 所有探测记录
    reason: str                        # 选择原因
    decided_at: str                    # ISO 8601
```

### 任务相关

```python
@dataclass(frozen=True)
class WorkloadSpec:
    """业务层构建、传给集群层的任务规格"""
    name: str                          # workload 名称
    docker_image: str                  # 容器镜像
    command: list[str]                 # 执行命令
    accelerator_type: str              # "tpu-v4-128", "nvidia-a100-80g"
    priority: str                      # "medium" | "high"
    env: dict[str, str]                # 环境变量
    namespace: str | None              # None 时使用集群默认 namespace
    extra_args: list[str]              # 透传给 xpk 的额外参数

@dataclass(frozen=True)
class WorkloadHandle:
    """任务提交后的句柄 — 后续查询/取消操作的标识"""
    workload_name: str
    cluster: str
    namespace: str
    submitted_at: str                  # ISO 8601

@dataclass(frozen=True)
class WorkloadStatus:
    """任务当前状态"""
    workload_name: str
    state: str                         # "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "UNKNOWN"
    start_time: str | None
    end_time: str | None
    exit_code: int | None

@dataclass(frozen=True)
class WorkloadInfo:
    """xpk workload list 返回的条目摘要"""
    name: str
    state: str                         # "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "UNKNOWN"
    accelerator_type: str
    start_time: str | None
```

### 日志相关

```python
@dataclass(frozen=True)
class LogFilter:
    """日志过滤与范围控制"""
    ranks: list[int] | None            # None = 全部 rank；[0] = 只取 rank-0
    since: str | None                  # ISO 8601，只取此时间之后的日志
    grep: str | None                   # Cloud Logging filter 表达式，用于关键字过滤
    tail_lines: int | None             # 限制返回行数

@dataclass(frozen=True)
class LogExportResult:
    """日志导出到 GCS 的结果"""
    workload_name: str
    cluster: str
    exported_ranks: list[int]          # 实际导出的 rank 列表
    gcs_uris: dict[int, str]           # {rank: "gs://falcon-.../logs/rank-0.log"}
    exported_at: str                   # ISO 8601
```

---

## 模块详细设计

### 1. ClusterRegistry

**职责：** 从静态配置文件 `repos.yaml` 中加载集群配置，提供按 accelerator 类型筛选候选集群的查询能力。

**存储与使用方式：**

- **存储介质（非数据库）**：集群配置**不存储在数据库中**，也不在运行时动态调用 GitHub API 获取，而是完全基于本地静态配置文件 `repos.yaml` 进行持久化（配置即代码，Configuration as Code）。
- **版本控制**：该 `repos.yaml` 文件随代码库一起提交至 Git/GitHub 进行版本控制，集群的变更（如新增集群、修改配额）通过正常的 PR (Pull Request) 代码审查流程进行，保证变更可追溯。
- **加载与使用机制**：
  1. **启动时加载**：应用服务（`ClusterFabric`）在初始化时，调用 `load()` 将配置一次性加载并缓存到内存中。
  2. **运行时只读**：系统运行期间配置在内存中是完全只读的，以保障极高的查询性能和并发安全。
  3. **变更生效**：若需修改集群配置，修改者需提交代码变更，合并后通过 CI/CD 流程触发服务的重新部署（或通过发送 HUP 信号/Webhook 触发重新加载），从而使新配置生效。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load` | `config_path: str` | `None` | 从 repos.yaml 加载 clusters 段 |
| `list_clusters` | — | `list[ClusterConfig]` | 列出所有已注册集群 |
| `get_cluster` | `name: str` | `ClusterConfig` | 按名查询，不存在时抛出 `KeyError` |
| `find_by_accelerator` | `accelerator_type: str` | `list[tuple[ClusterConfig, int]]` | 按 accelerator 类型筛选候选集群，返回 (配置, 优先级) 列表，按 priority 升序排序 |

**依赖：** PyYAML

**配置格式（repos.yaml）：**

```yaml
clusters:
  tpu-v4-prod:
    provider: gke
    project: primatrix-tpu
    zone: us-central2-b
    cluster_id: tpu-v4-cluster
    accelerator_types:
      - type: tpu-v4-64
        device_class: tpu
        priority: 1
      - type: tpu-v4-128
        device_class: tpu
        priority: 1
    default_namespace: default
    gcs_bucket: gs://falcon-tpu-v4
    max_queue_depth: 5
    probe_timeout_s: 15
    labels:
      env: prod
      team: pretrain

  gpu-a100:
    provider: gke
    project: primatrix-gpu
    zone: us-central1-a
    cluster_id: gpu-cluster
    accelerator_types:
      - type: nvidia-a100-80g
        device_class: gpu
        priority: 1
    default_namespace: benchmark
    gcs_bucket: gs://falcon-gpu
    max_queue_depth: 5
    probe_timeout_s: 15
    labels:
      env: prod
      team: gpu
```

---

### 2. ClusterProber

**职责：** 对单个集群执行健康探测，返回可达性和队列深度信息。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `probe` | `cluster: ClusterConfig` | `ClusterProbeResult` | 探测集群健康状态 |

**探测流程：**

```text
probe(cluster)
│
├─ 执行: xpk workload list \
│        --cluster={cluster.cluster_id} \
│        --project={cluster.project} \
│        --zone={cluster.zone}
│    超时: cluster.probe_timeout_s 秒
│
├─ 成功 → 解析输出
│   ├─ 统计 state=QUEUED 的条目 → queue_depth
│   ├─ 统计 state=RUNNING 的条目 → running_count
│   └─ 返回 ClusterProbeResult(reachable=True, ...)
│
├─ 超时 → ClusterProbeResult(reachable=False, error="probe timed out after {N}s")
│
└─ 命令失败 → ClusterProbeResult(reachable=False, error=stderr)
```

**依赖：** `xpk` CLI（通过 `subprocess` 调用）

---

### 3. ClusterSelector

**职责：** 组合 ClusterRegistry 和 ClusterProber，实现完整的集群选择策略。给定 accelerator 类型，自动选择最优可用集群。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `select` | `accelerator_type: str` | `ClusterSelection` | 选择最优集群，全部失败时抛出 `NoAvailableClusterError` |

**选择算法：**

```text
select(accelerator_type)
│
├─ 1. Registry.find_by_accelerator(accelerator_type)
│     → 候选列表，按 priority 升序（1 最优先）
│     → 空列表 → 抛出 NoAvailableClusterError("no cluster supports {type}")
│
├─ 2. 并行 Prober.probe(candidate) — 对所有候选同时探测
│     使用 concurrent.futures.ThreadPoolExecutor
│     每个探测独立超时（probe_timeout_s）
│     → 收集所有 probe_results
│
├─ 3. 从通过探测的候选中（reachable=True 且 queue_depth <= max_queue_depth），
│     按 priority 选取最优 → 返回 ClusterSelection
│     reason: "selected {name}: reachable, queue_depth={N} <= max={M}"
│
└─ 4. 全部失败 → 抛出 NoAvailableClusterError
      附带所有 probe_results 用于调试
```

**依赖：** `ClusterRegistry`, `ClusterProber`

---

### 4. XpkWorkloadManager

**职责：** 封装 xpk CLI 的任务生命周期操作，包括提交、查询、取消、列表。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `submit` | `cluster: ClusterConfig, spec: WorkloadSpec` | `WorkloadHandle` | 提交任务 |
| `get_status` | `handle: WorkloadHandle` | `WorkloadStatus` | 查询任务状态 |
| `cancel` | `handle: WorkloadHandle` | `bool` | 取消任务 |
| `list_workloads` | `cluster: ClusterConfig, state_filter: str \| None` | `list[WorkloadInfo]` | 列出任务 |

**submit 命令构建：**

```text
xpk workload create \
  --cluster={cluster.cluster_id} \
  --project={cluster.project} \
  --zone={cluster.zone} \
  --workload={spec.name} \
  --docker-image={spec.docker_image} \
  --command="{spec.command joined}" \
  --tpu-type={spec.accelerator_type}    # TPU 场景
  --device-type={spec.accelerator_type}  # GPU 场景
  --priority={spec.priority} \
  --namespace={spec.namespace or cluster.default_namespace} \
  --env={key=value for each spec.env} \
  {spec.extra_args}
```

TPU 与 GPU 的区分通过 `AcceleratorSlot.device_class` 显式决定：

- `device_class: "tpu"` → 使用 `--tpu-type`
- `device_class: "gpu"` → 使用 `--device-type`

**依赖：** `xpk` CLI（通过 `subprocess` 调用）

---

### 5. CloudLogRetriever

**职责：** 通过 Google Cloud Logging API 获取 GKE Pod 日志，支持流式读取和 GCS 批量导出两种模式。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `stream_log` | `cluster: ClusterConfig, workload_name: str, namespace: str, rank: int, filter: LogFilter` | `Iterator[str]` | 流式读取单个 rank 的日志，按行 yield |
| `export_logs` | `cluster: ClusterConfig, workload_name: str, namespace: str, filter: LogFilter, gcs_dest: str` | `LogExportResult` | 将日志导出到 GCS |

**Pod 发现策略：**

Cloud Logging 中 GKE 容器日志使用 `k8s_container` 资源类型。通过以下 Cloud Logging filter 定位属于指定 workload 的 Pod 日志：

```text
resource.type="k8s_container"
resource.labels.project_id="{cluster.project}"
resource.labels.cluster_name="{cluster.cluster_id}"
resource.labels.namespace_name="{namespace}"
labels."k8s-pod/xpk-workload"="{workload_name}"
```

**Rank 识别策略：**

优先通过 Kubernetes label 识别 rank。xpk 创建的 Pod 会带有 `batch.kubernetes.io/job-completion-index` label（对应 rank 编号），Cloud Logging 中可通过 `labels."k8s-pod/batch.kubernetes.io/job-completion-index"` 获取。如果该 label 不存在，回退到从 `resource.labels.pod_name` 解析（xpk Pod 名称遵循 `{workload_name}-{index}-{hash}` 格式）。

**流式读取（stream_log）：**

- 调用 Cloud Logging API 的 `list_entries()` 方法，设置 `page_size` 控制批次
- 按时间排序返回日志条目
- 每条 entry 提取 `text_payload` 或 `json_payload.message` 作为日志行
- 支持 `filter.grep` 在 Cloud Logging 侧过滤（减少数据传输）
- 支持 `filter.tail_lines` 限制返回行数

**GCS 导出（export_logs）：**

- 按 rank 分别查询日志
- 使用 GCS 客户端库的流式上传（resumable upload），将 Cloud Logging 查询结果直接流式写入 GCS，不经过本地磁盘
- 返回每个 rank 的 GCS URI

**GCS 日志目录结构（业务层传入的 `gcs_dest` 示例，集群层不构建此路径）：**

```text
gs://falcon-{cluster}/profiles/{session_id}/logs/
  ├─ rank-0.log              # 主 worker 日志（必定收集）
  ├─ rank-1.log              # 按需收集
  ├─ ...
  └─ collection_meta.json    # 收集元数据
```

**依赖：** `google-cloud-logging` Python SDK, `google-cloud-storage` Python SDK

---

### 6. ClusterFabric（门面）

**职责：** 作为集群编排层的唯一对外入口，组合内部子模块，为业务逻辑层提供统一接口。

**对外接口：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `select_cluster` | `accelerator_type: str` | `ClusterSelection` | 选择最优可用集群 |
| `submit_workload` | `cluster: str, spec: WorkloadSpec` | `WorkloadHandle` | 提交任务到指定集群 |
| `get_workload_status` | `handle: WorkloadHandle` | `WorkloadStatus` | 查询任务状态 |
| `cancel_workload` | `handle: WorkloadHandle` | `bool` | 取消任务 |
| `list_workloads` | `cluster: str, state_filter: str \| None` | `list[WorkloadInfo]` | 列出指定集群的任务 |
| `stream_pod_log` | `handle: WorkloadHandle, rank: int, filter: LogFilter` | `Iterator[str]` | 流式读取 Pod 日志 |
| `export_pod_logs` | `handle: WorkloadHandle, filter: LogFilter, gcs_dest: str` | `LogExportResult` | 导出 Pod 日志到 GCS |
| `list_clusters` | — | `list[ClusterConfig]` | 列出所有已注册集群 |
| `get_cluster_status` | `cluster: str` | `ClusterProbeResult` | 获取集群健康状态 |

**初始化：**

```text
ClusterFabric(config_path: str)
│
├─ ClusterRegistry.load(config_path)
├─ ClusterProber()
├─ ClusterSelector(registry, prober)
├─ XpkWorkloadManager()
└─ CloudLogRetriever()
```

所有接口方法内部委托给对应的子模块。`cluster: str` 参数通过 `Registry.get_cluster()` 解析为 `ClusterConfig`。日志方法接受 `WorkloadHandle`，从中提取 `cluster`、`workload_name` 和 `namespace`。

**select → submit 竞态说明：** `select_cluster` 返回后到 `submit_workload` 调用之间，集群状态可能已变化（队列增长或不可达）。这是可接受的 — 如果 submit 失败，业务层负责重试或重新选择集群。集群层不保证 select 结果的时效性。

---

## 错误处理

集群层定义以下异常类型，向业务层提供清晰的错误语义：

| 异常 | 触发条件 | 建议的业务层处理 |
|------|---------|----------------|
| `NoAvailableClusterError` | 所有候选集群不可达或队列满 | Session → FAILED，附带 probe_results |
| `WorkloadSubmitError` | `xpk workload create` 失败 | Session → FAILED，记录 stderr |
| `WorkloadNotFoundError` | 查询/取消不存在的 workload | 记录 warning，继续 |
| `ClusterUnreachableError` | 单个集群探测超时或命令失败 | ClusterSelector 跳过该集群 |
| `LogRetrievalError` | Cloud Logging API 调用失败 | Session 仍可 COMPLETED，日志标记为缺失 |
| `ClusterConfigError` | repos.yaml 配置缺少必要字段或格式错误 | 启动时快速失败 |

---

## 外部依赖与权限要求

### 1. 软件依赖

| 依赖 | 用途 | 安装方式 |
|------|------|---------|
| `xpk` CLI | 任务调度（workload create/list/delete）+ 集群探测 | `pip install xpk` |
| `gcloud` CLI | Google Cloud SDK，为 xpk 提供底层认证与 API 调用支持 | [官方文档安装](https://cloud.google.com/sdk/docs/install) |
| `google-cloud-logging` SDK | Pod 日志的流式读取和查询 | `pip install google-cloud-logging` |
| `google-cloud-storage` SDK | 日志流式上传到 GCS | `pip install google-cloud-storage` |
| `PyYAML` | 读取 repos.yaml 集群配置 | `pip install pyyaml` |

### 2. Google Cloud 权限与套件依赖 (xpk 环境要求)

xpk 是一个基于 GKE (Google Kubernetes Engine) 封装的 TPU/GPU 调度工具。要使 xpk 正常工作，运行 Falcon 集群编排层的环境（如调度服务所在的 VM 或 Pod）必须具备正确的 GCP 套件和凭据。

#### 2.1 基础套件要求

- **Google Cloud SDK (`gcloud`)**：xpk 底层强依赖 `gcloud` 命令行工具来获取集群凭据和执行部分云资源查询。必须确保运行环境中 `gcloud` 可执行，且版本符合 xpk 的要求。
- **Kubeconfig 写入权限**：xpk 每次与集群交互前，可能会调用 `gcloud container clusters get-credentials`。程序运行时的用户需要对本地 `~/.kube/config` 或由 `KUBECONFIG` 环境变量指定的路径拥有写权限。

#### 2.2 身份凭证机制

调度程序不能使用个人账号运行，应使用 Google Cloud Service Account (GSA)。认证方式按部署环境区分：

- **GKE 内部署 (推荐)**：使用 Workload Identity (WIF)。将 Kubernetes Service Account 绑定到具有权限的 GSA，应用代码和 CLI 会自动获取凭证，无需下载密钥文件。
- **GCE VM 部署**：在创建 VM 实例时绑定 GSA。
- **本地开发**：使用 `gcloud auth application-default login` 获取开发者凭据。

#### 2.3 核心 IAM 权限要求

用于执行调度的 Service Account 必须在目标 GCP Project 具备以下角色（或组合等效自定义权限）：

| 权限领域 | 推荐角色 | 用途说明 |
|---------|---------|----------|
| **GKE 集群交互** | `roles/container.developer` <br>(Kubernetes Engine Developer) | 这是最核心的权限。允许 xpk 获取集群的 admin/developer 凭据，并执行 Kubernetes 资源的 CRUD 操作（包括提交 Job/Pod）。 |
| **日志获取** | `roles/logging.viewer` <br>(Logs Viewer) | 允许 `CloudLogRetriever` 通过 API 查询属于该项目的 GKE Pod 容器日志（对应 `resource.type="k8s_container"`）。 |
| **云存储读写** | `roles/storage.objectAdmin` <br>(Storage Object Admin) | 允许 `CloudLogRetriever` 将日志流式导出至指定的 GCS bucket，以及允许底层引擎读写 checkpoint/profile 数据（权限可以收敛到特定 Bucket）。 |
| **计算资源查看** | `roles/compute.viewer` <br>(Compute Viewer) | xpk 内部可能需要查询节点池拓扑、加速器配额或网络可达性。 |

> **并发注意事项**：由于 `ClusterProber` 会并发向多个集群发起探测（执行 `xpk workload list`），每个线程底层都可能会调用 `gcloud` 获取凭据。建议在服务启动阶段，提前遍历 `ClusterRegistry` 中的所有集群，预热（预先执行并缓存）所有集群的 `kubeconfig`，以避免高并发探测时引发竞争或过多的 API 限流 (Rate Limit) 错误。

---

## 测试验收标准

### 1. ClusterRegistry

| 测试项 | 验收标准 |
|--------|---------|
| 加载合法配置 | 正确解析 repos.yaml 中的 clusters 段，生成 `ClusterConfig` 列表 |
| 配置校验 | 缺少必要字段（project/zone/cluster_id）时抛出 `ClusterConfigError` |
| 按 accelerator 查询 | `find_by_accelerator("tpu-v4-128")` 返回包含该类型的集群，按 priority 升序排序 |
| 无匹配 accelerator | 返回空列表 |
| 重复加载 | 多次 `load()` 覆盖旧配置，不产生重复 |

### 2. ClusterProber

| 测试项 | 验收标准 |
|--------|---------|
| 正常探测 | 调用 `xpk workload list`，正确解析 QUEUED/RUNNING 数量 |
| 探测超时 | 在 `probe_timeout_s` 内未返回 → `reachable=False`, `error` 包含超时信息 |
| xpk 命令失败 | 非零退出码 → `reachable=False`, `error` 包含 stderr |
| 空集群 | 无 workload 运行 → `queue_depth=0, running_count=0, reachable=True` |
| probed_at 时间 | 返回 ISO 8601 格式的探测时间戳 |

### 3. ClusterSelector

| 测试项 | 验收标准 |
|--------|---------|
| 单候选命中 | 唯一候选可达且队列未满 → 选中，`reason` 包含选择原因 |
| 优先级排序 | 多个候选时，并行探测后按 priority 选取最优的可用集群 |
| 队列满跳过 | 候选集群 `queue_depth > max_queue_depth` → 自动跳过，选取下一优先级 |
| 全部失败 | 所有候选不可达 → 抛出 `NoAvailableClusterError`，包含所有 `probe_results` |
| 无候选 | 无集群支持该 accelerator → 抛出 `NoAvailableClusterError` |
| 决策记录完整 | `ClusterSelection` 包含 `candidates`、`probe_results`、`selected_cluster`、`reason` |

### 4. XpkWorkloadManager

| 测试项 | 验收标准 |
|--------|---------|
| 提交任务 | 正确构建 `xpk workload create` 命令，包含 image/command/accelerator/env/extra_args |
| TPU vs GPU | `device_class="tpu"` 使用 `--tpu-type`；`device_class="gpu"` 使用 `--device-type` |
| 提交失败 | xpk 返回非零退出码 → 抛出 `WorkloadSubmitError`，包含 stderr |
| 查询状态 | 正确解析 `xpk workload list` 输出，映射到 `WorkloadStatus.state` |
| 取消任务 | 调用 `xpk workload delete`，成功返回 True |
| 取消不存在的任务 | 抛出 `WorkloadNotFoundError` |
| namespace 默认值 | `WorkloadSpec.namespace=None` 时使用 `ClusterConfig.default_namespace` |
| extra_args 透传 | `spec.extra_args` 原样追加到 xpk 命令末尾 |

### 5. CloudLogRetriever

| 测试项 | 验收标准 |
|--------|---------|
| 流式读取 rank-0 | `stream_log(rank=0)` 返回迭代器，按行 yield rank-0 Pod 的日志 |
| grep 过滤 | `filter.grep="step_time"` 在 Cloud Logging 侧过滤，只返回匹配行 |
| tail 限制 | `filter.tail_lines=100` 只返回最后 100 行 |
| since 过滤 | `filter.since="2026-04-13T00:00:00Z"` 只返回该时间之后的日志 |
| GCS 导出 | `export_logs(gcs_dest=...)` 将日志写入指定 GCS 路径，返回每个 rank 的 URI |
| 多 rank 导出 | `filter.ranks=None` → 导出所有 rank；`filter.ranks=[0]` → 只导出 rank-0 |
| Pod/Rank 发现 | 正确通过 Cloud Logging resource labels 识别 workload 关联的 Pod 和 rank |
| API 失败 | Cloud Logging API 异常 → 抛出 `LogRetrievalError`，包含原始错误信息 |

### 6. ClusterFabric（集成测试）

| 测试项 | 验收标准 |
|--------|---------|
| 初始化 | 从 repos.yaml 路径初始化，所有内部模块正确创建 |
| 端到端提交 | `select_cluster → submit_workload → get_workload_status` 完整流程正常 |
| 集群名称解析 | `submit_workload(cluster="tpu-v4-prod", ...)` 正确解析为 ClusterConfig |
| 不存在的集群 | 传入未注册集群名 → 抛出 `KeyError` |
| 错误透传 | 内部模块抛出的异常通过 Fabric 透传给调用方，不吞异常 |

---

## 与其他层的交互

### 业务逻辑层调用集群编排层的场景

| 业务场景 | ClusterFabric 方法 |
|---------|-------------------|
| Session 提交：选择集群 | `select_cluster(accelerator_type)` |
| Session 提交：创建 workload | `submit_workload(cluster, spec)` |
| Session 轮询：检查 workload 状态 | `get_workload_status(handle)` |
| Session 取消 | `cancel_workload(handle)` |
| 数据收集：获取 rank-0 日志用于 metrics 解析 | `stream_pod_log(handle, rank=0, filter)` |
| 数据收集：归档日志到 GCS | `export_pod_logs(handle, filter, gcs_dest)` |
| CLI `falcon cluster list` | `list_clusters()` |
| CLI `falcon cluster status` | `get_cluster_status(cluster)` |
| CLI `falcon job list` | `list_workloads(cluster, state_filter)` |

### 不在集群编排层的职责

以下能力属于业务逻辑层（Service Layer），不在集群编排层实现：

- Session 生命周期状态机管理（`PENDING → PROVISIONING → RUNNING → ...`）
- 轮询循环调度（`poll_interval`、重试策略）
- Metrics 解析（从日志中提取 step_time、MFU 等指标）
- Profile artifact 扫描（GCS 目录扫描发现 xplane/llo/hlo 文件）
- 分析触发（调用 XProfAnalyzer/DeepProfileAnalyzer）
- 调度决策持久化（写入 DB 的 `jobs.scheduling_decision` 字段）

---

## 不在本文档范围

- 多租户与权限管理
- 集群自动扩缩容
- 跨区域负载均衡
- 任务重试策略（属于业务层 JobScheduler 的职责）
- GCS artifact 扫描（属于业务层 ProfileCollector 的职责）
- Web API 接口（属于展示层）
