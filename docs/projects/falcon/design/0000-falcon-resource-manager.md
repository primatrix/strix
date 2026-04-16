# Falcon 设计文档

## 背景

团队当前在多个 GCP 项目下维护着多个 GKE 集群，用于运行 TPU 相关的工作负载（profiling、benchmark、精度对齐、模型训练等）。这些集群分布在不同的 project 和 region 中，各自拥有不同的 TPU 类型和预留资源。

上层业务组件需要向这些边缘集群下发任务，并在任务完成后获取其状态和产物（GCS 路径），以支持后续的产物索引、报告生成等流程。

目前缺少一个统一的中心化任务下发和状态回收机制，各集群的任务管理依赖手动操作或分散的脚本，无法规模化。

## 动机

1. **跨项目任务下发**: 边缘集群分布在不同 GCP 项目下，需要一个安全的、基于 IAM 的跨项目访问方案，而非静态 kubeconfig 或手动操作
2. **统一的任务生命周期管理**: 上层组件写入任务、controller 下发执行、状态回写数据库，形成闭环
3. **产物与日志**: 产物和日志路径由上层组件约定，Job 通过 GCS FUSE 挂载自行写入 GCS，falcon-resource-manager 不参与数据面
4. **规模化**: 随着 TPU 资源（v6e、v7x 等）和集群数量增长（当前 2-5 个），需要多集群支持从第一天就内建于架构中

## 目标（Goals）

- **G1**: 实现一个运行在中心 GKE 集群上的 Go controller，能够从数据库读取待执行任务，根据 TPU 类型自动选择目标集群，向边缘集群提交 K8s Job
- **G2**: 支持多集群——通过集群注册表（`clusters` 表）管理边缘集群，新增集群只需 IAM 授权 + 数据库插入一行记录
- **G3**: Node pool 按需创建与延迟删除——controller 在提交 Job 前创建所需的 TPU node pool（或复用已有的同拓扑 node pool），Job 终态后检查是否有后续同拓扑任务可复用，无则删除
- **G4**: 任务状态闭环——controller 跟踪 Job 执行状态（running / completed / failed / timed_out），将最终状态回写 job 表
- **G5**: Job 日志收集——Job 的 Pod 通过 GCS FUSE 挂载将 stdout/stderr 重定向写入 GCS，日志路径按模板可推导（`gs://<bucket>/logs/<job_id>/<pod_name>.log`），controller 不参与日志收集
- **G6**: 基于 GCP Workload Identity 实现零静态凭证的跨项目认证，已通过 POC 验证（2026-04-15）

## 非目标（Non-Goals）

- **NG1**: falcon-resource-manager 不决定"运行什么"——任务的创建由上层组件负责，falcon-resource-manager 负责集群选择和任务下发
- **NG2**: 不管理产物——产物由 Job 自行写入 GCS（路径由上层组件约定），falcon-resource-manager 不记录、不读取、不转移产物
- **NG3**: 不替代现有 CI/CD 流水线（GitHub Actions + ARC runners），Falcon 面向的是 CI 之外的按需任务下发场景
- **NG4**: 不提供 Web UI——MVP 阶段通过数据库记录 + 日志完成可观测性

## 成功指标（Success Metrics）

1. 任务写入数据库后，GKE node pool 应成功创建，K8s Job 应成功生成
2. 除非集群无可用资源，Job 对应的 Pod 应成功启动
3. Job 无论成功或失败，job 表状态应被更新至终态
4. Job 运行期间，Pod 日志应通过 GCS FUSE 实时写入 GCS

## 影响范围

| 组件 | 影响说明 |
|------|----------|
| 中心 GKE 集群 (tpu-service-473302/tpu-service) | 部署 falcon-resource-manager，新增 `falcon` namespace 和 Workload Identity 绑定 |
| 边缘 GKE 集群 (poc-tpu-partner 等) | 接收 controller 提交的 Job，需在对应 project 配置 IAM 授权 |
| GCP IAM | 中心 project 创建 SA (`cluster-controller`)，边缘 project 授予 `container.admin` + `container.clusterViewer`。MVP 使用 `container.admin` 是因为 node pool 的创建/删除需要较高权限；后续应收敛为自定义角色，仅包含 `container.nodePools.*`、`container.jobs.*`、`container.pods.get/list` 等必要权限 |
| 数据库 (MySQL) | `job` 表由上层组件定义和创建，Falcon 读写其中的调度相关字段；`clusters` 表由 Falcon 管理 |
| 上层业务组件 | 拥有 `job` 表的 schema 定义权；负责写入 job 记录、读取状态；产物路径由上层自行约定并写入 GCS |
| GCS | 各边缘集群的 Job 将产物写入各自的 GCS bucket，bucket 信息在 `clusters` 表中注册 |

## 高层架构

```text
                         ┌──────────┐
                         │ 上层组件  │
                         └────┬─────┘
                              │ 写入任务
                         ┌────▼─────┐      ┌────────┐
                         │API Server│─────►│ MySQL  │
                         └────┬─────┘      └────────┘
                              │
                    拉取任务 / 回写状态
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│  中心集群 (tpu-service-473302/tpu-service)                      │
│  ┌──────────────────────────┼────────────────────────────────┐ │
│  │  falcon-resource-manager ▼                                │ │
│  │  ns: falcon, sa: falcon-controller (Workload Identity)    │ │
│  │                                                           │ │
│  │  ┌─────────────┐                                          │ │
│  │  │ Task Poller │ ── 轮询 API Server 拾取 pending 任务      │ │
│  │  └──────┬──────┘                                          │ │
│  │         ▼                                                 │ │
│  │  ┌──────────────┐                                         │ │
│  │  │NodePool Mgr  │ ── GKE API 创建/删除 node pool          │ │
│  │  └──────┬───────┘                                         │ │
│  │         ▼                                                 │ │
│  │  ┌──────────────┐                                         │ │
│  │  │Job Submitter │ ── K8s API 提交 Job                     │ │
│  │  └──────┬───────┘                                         │ │
│  │         ▼                                                 │ │
│  │  ┌──────────────┐                                         │ │
│  │  │Status Reconc.│ ── K8s Informer 监听 Job 状态            │ │
│  │  └──────────────┘                                         │ │
│  └───────────────────────────┬───────────────────────────────┘ │
└──────────────────────────────┼─────────────────────────────────┘
                               │
                      Workload Identity
                               │
          ┌────────────────────┼──────────────────────┐
          │                    ▼                      │
          │  ┌──────────────────────────────────────┐ │
          │  │ 边缘集群 1                            │ │
          │  │ (poc-tpu-partner/tpuv7x-64-node)     │ │
          │  │ Jobs → GCS Bucket A                  │ │
          │  └──────────────────────────────────────┘ │
          │  ┌──────────────────────────────────────┐ │
          │  │ 边缘集群 2                            │ │
          │  │ (project-b/cluster-y)                │ │
          │  │ Jobs → GCS Bucket B                  │ │
          │  └──────────────────────────────────────┘ │
          │            ... (2-5 个集群)               │
          └───────────────────────────────────────────┘
```

## 关键设计决策与权衡

### 1. 认证方案：Workload Identity + GKE API

**选择**: Workload Identity 自动凭证 + GKE API 动态发现集群端点

**权衡**:

- (+) 零静态凭证，token 自动刷新，安全审计友好
- (+) 跨项目只需 IAM binding，无需网络特殊配置
- (-) 依赖边缘集群 API server 为 public endpoint（private cluster 需额外配置 VPN / Private Service Connect）
- (-) token 有效期 1 小时，长连接（Informer）需要 transport 层自动刷新，增加了 client 构造复杂度

已通过 POC 验证全链路：credentials 获取 → 集群 endpoint 发现 → Job 提交 → Job 执行完成

### 2. 多集群支持：注册表模式

**选择**: 数据库 `clusters` 表管理边缘集群

**`clusters` 表 Schema**（由 Falcon 管理）:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 集群 ID |
| name | VARCHAR | 集群显示名（如 "tpuv7x-64-node"） |
| project | VARCHAR | GCP 项目 ID |
| location | VARCHAR | GKE location（如 "us-central1"） |
| tpu_type | VARCHAR | 集群提供的 TPU 类型（如 "v7x"、"v6e"），用于任务匹配 |
| gcs_bucket | VARCHAR | 该集群产物的默认 GCS bucket |
| default_reservation | VARCHAR | TPU 预留名称（如 "ghostfish-kankgdm2b4m7f"），为空则仅使用 flex-start。有值时优先使用 reservation，资源用尽自动 fallback 到 flex-start |
| status | ENUM | active / maintenance / offline |
| created_at | TIMESTAMPTZ | 创建时间 |

**权衡**:

- (+) 新增集群只需 IAM 授权 + 插入一行记录，运维成本极低
- (+) 集群可设置状态（active / maintenance / offline），controller 自动跳过非活跃集群
- (-) 引入了数据库作为集群注册的 source of truth，集群元数据变更需要同步更新数据库
- (-) 无法利用 K8s 原生的多集群方案（如 KubeFed），但避免了引入额外的 K8s 组件

### 3. 集群选择策略

任务不指定具体集群，而是声明所需的 TPU 类型（如 "v7x"）。Controller 根据 TPU 类型匹配所有 active 状态的候选集群，按以下优先级选择：

1. **优先复用 node pool**: 候选集群中存在空闲的、拓扑匹配的 node pool（由延迟删除机制保留），直接复用，省去 5-10 分钟创建时间
2. **队列深度**: 无可复用 node pool 时，选择当前排队任务最少的集群，均衡负载

**权衡**:

- (+) 上层组件无需感知集群拓扑细节，只需声明 TPU 类型
- (+) node pool 复用与集群选择形成闭环——有空闲 node pool 的集群自然被优先选中
- (-) 引入了调度决策，增加了 controller 复杂度
- (-) 队列深度信息需要 controller 维护，增加了状态管理

### 4. 排队与调度模型

**队列粒度**: per (cluster, topology)。每个队列同时只允许一个任务处于资源获取阶段（provisioning / dispatching），已 Running 的任务不占队列位。

- 资源获取并行度 = 集群数量 x 每个集群的拓扑种类数（每个队列同时一个任务在获取资源）
- 执行并行度取决于实际 TPU 资源，不受队列限制
- 不同拓扑之间不阻塞（64chip 等资源不影响 4chip）
- 不同集群之间不阻塞

**放行条件**: 当前资源获取中的任务满足以下任一条件后，放行下一个：

- Pod 确认 Running（资源到手）
- 任务到达终态（failed / timed_out）

**Pick Job 规则**: 队列槽位空出时，从 waiting 任务中选择下一个执行：

1. **用户内 FIFO** — 同一个 actor 的任务按提交顺序执行
2. **用户间 Round-Robin** — 上一个获得槽位的 actor 让位给其他 actor

选择流程：收集 waiting 任务 → 按 actor 分组，每组取最早提交的 → 如果候选 actor > 1，排除上次获得槽位的 actor → 剩余中取最早提交的

**超时兜底**: 每个任务有 `timeout_seconds`。flex-start 场景下 Pod 长时间 pending 拿不到资源，超时后强制终态（timed_out），释放队列位置。

**权衡**:

- (+) FIFO 保证排队公平，无论 reservation 还是 flex-start
- (+) Round-robin 防止单一 actor 垄断资源
- (+) per (cluster, topology) 粒度避免不同拓扑互相阻塞
- (-) 串行化资源获取阶段降低了瞬时吞吐（同队列不能同时创建多个 node pool）
- (-) Controller 需要维护每个队列的状态和 actor 轮转记录

### 5. Node pool 按需创建与延迟删除

**选择**: Job 独享 node pool 执行，Job 结束后延迟删除——若有同拓扑的 pending 任务，保留 node pool 供下一个 Job 复用；若无，立即删除

**供给模式选择**: 创建 node pool 前，通过 GCP Compute API 查询 reservation 剩余容量（`count - inUseCount`），按以下逻辑决策：

1. 集群配置了 `default_reservation` 且剩余容量 >= 所需节点数 → 使用 reservation 创建
2. 集群配置了 `default_reservation` 但剩余不足 → fallback 到 flex-start 创建
3. 集群未配置 `default_reservation` → 直接使用 flex-start 创建

**Autoscaler 空闲超时**: 根据实际使用的供给模式（而非集群配置）设置不同的 `scaleDownUnneededTime`：

| 实际供给模式 | scaleDownUnneededTime | 原因 |
|-------------|----------------------|------|
| reservation | 短（默认 10min） | 资源预留，释放后仍可重新获取 |
| flex-start | 长（如 6h） | 资源稀缺，释放后难以重新获取 |

**权衡**:

- (+) TPU 预留资源按需使用，无任务时不持有 node pool
- (+) 执行期间仍为单 Job 独享，保持资源隔离
- (+) 同拓扑任务连续到达时，省去 node pool 创建的 5-10 分钟延迟
- (-) 延迟删除期间 node pool 短暂空闲，占用少量 quota（但时间窗口很短——仅为检查 pending 队列的耗时）
- (-) 需要在 deprovisioning 阶段查询 pending 队列，增加了 NodePool Manager 与调度逻辑的耦合

### 6. 任务与 Job 的关系

一个 task 对应一个 K8s Job。task 记录中的 `spec` 字段包含 Job 渲染所需的参数（镜像、命令、TPU 拓扑等），controller 负责将其渲染为完整的 K8s Job manifest。

### 7. 日志收集

**选择**: Pod 内通过 GCS FUSE 挂载将 stdout/stderr 重定向写入 GCS，controller 不参与日志数据面

**日志存储格式**: 每个 Pod 一个日志文件，路径按模板可推导，默认 `gs://<bucket>/logs/<job_id>/<pod_name>.log`。多 Pod 场景（如多节点训练）下各 Pod 日志独立存储。

Controller 在渲染 Job manifest 时注入 GCS FUSE volume 和重定向配置，Pod 启动后日志实时写入 GCS。

**权衡**:

- (+) 控制面与数据面解耦——controller 崩溃不影响日志收集
- (+) 无需在边缘集群部署额外的日志收集组件
- (+) 日志实时写入，不依赖 Pod 存续状态，无需管理 Job 删除时机
- (+) 日志路径按模板可推导，无需额外存储路径信息
- (-) 需要 Job 容器支持 stdout/stderr 重定向（controller 在 Job manifest 中注入）
- (-) 依赖 GCS FUSE 可用性

## 任务全生命周期编排

一个任务从创建到结束的完整流程，包括各组件的协作关系、状态流转和异常路径：

```text
╔══════════════════════════════════════════════════════════════╗
║  Phase 1: pending → waiting               [Task Poller]     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  上层组件 ── API Server ──► DB (status=pending)              ║
║                              │                               ║
║                   Task Poller 轮询 API Server 拾取            ║
║                              │                               ║
║                      校验字段、匹配 TPU 类型                   ║
║                      选择目标集群（优先有空闲 node pool）       ║
║                              │                               ║
║                      status → waiting                        ║
║                      进入 (cluster, topology) 队列            ║
║                              ▼                               ║
╠══════════════════════════════════════════════════════════════╣
║  Phase 2: waiting → provisioning          [Scheduler]        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║        检查队列槽位（同队列是否有任务正在 provisioning/        ║
║        dispatching）                                         ║
║        ┌──── 占满 ────┐       ┌──── 空闲 ────┐               ║
║        ▼              │       ▼              │               ║
║   继续等待            │  Pick Job 规则选中:                    ║
║   (等待放行条件:       │  1. 同 actor FIFO                    ║
║    Pod Running 或      │  2. 跨 actor Round-Robin             ║
║    任务终态)           │       │                              ║
║        │              │       │                              ║
║        └──────────────┘  status → provisioning               ║
║                              ▼                               ║
╠══════════════════════════════════════════════════════════════╣
║  Phase 3: provisioning → dispatching  [NodePool Manager]    ║
╠══════════════════════════════════════════════════════════════╣
║                              │                               ║
║               检查目标集群是否有同拓扑空闲 node pool           ║
║               ┌──── 有 ────┐       ┌──── 无 ────┐            ║
║               ▼            │       ▼            │            ║
║          复用已有          │  查询 reservation 剩余容量       ║
║               │            │  (GCP Compute API)              ║
║               │            │  ┌─ 充足 ─┐  ┌─ 不足/无 ─┐     ║
║               │            │  ▼        │  ▼           │     ║
║               │            │  reservation  flex-start        ║
║               │            │  └────────┴──────────────┘     ║
║               │            │  GKE API: CreateNodePool()      ║
║               │            │       │                         ║
║               │            │  轮询等待 node pool RUNNING     ║
║               │            │  ╳ 超时/失败 →                  ║
║               │            │    error_phase=provisioning     ║
║               │            │    status=failed, 结束          ║
║               └────────────┴───────┘                         ║
║                              │                               ║
║                      记录 node_pool_name                     ║
║                      status → dispatching                    ║
║                              ▼                               ║
╠══════════════════════════════════════════════════════════════╣
║  Phase 4: dispatching → running           [Job Submitter]   ║
╠══════════════════════════════════════════════════════════════╣
║                              │                               ║
║               从 spec 渲染 K8s Job manifest                  ║
║               注入 nodeSelector, resources, volumes          ║
║               注入 GCS FUSE volume + 日志重定向              ║
║                              │                               ║
║               提交 Job 到边缘集群                             ║
║               ╳ 失败 → error_phase=dispatching               ║
║                 error_message=K8s API 错误详情                ║
║                 status=failed, 触发删除 node pool             ║
║                              │                               ║
║                      记录 job_uid                            ║
║                      status → running                        ║
║                              ▼                               ║
╠══════════════════════════════════════════════════════════════╣
║  Phase 5: running → (终态)       [Status Reconciler]        ║
╠══════════════════════════════════════════════════════════════╣
║                              │                               ║
║               Informer 监听边缘集群 Job 状态                  ║
║               (label: managed-by=falcon)                     ║
║                              │                               ║
║                    ┌─────────┴─────────┐                     ║
║                    ▼                   ▼                      ║
║              Job Complete        Job Failed / 超时            ║
║                    │                   │                      ║
║                    └─────────┬─────────┘                     ║
║                              ▼                               ║
╠══════════════════════════════════════════════════════════════╣
║  Phase 6: deprovisioning → 终态       [NodePool Manager]    ║
╠══════════════════════════════════════════════════════════════╣
║                              │                               ║
║               Controller 删除 Job                            ║
║               检查是否有 pending 任务需要同拓扑 node pool     ║
║               ┌──── 有 ────┐       ┌──── 无 ────┐            ║
║               ▼            │       ▼            │            ║
║          保留 node pool    │  GKE API: DeleteNodePool()      ║
║          供下一个 Job 复用  │  等待删除完成                    ║
║               │            │  ╳ 失败 → 重试, 最终记录 error   ║
║               │            │    (error_phase=deprovisioning)  ║
║               └────────────┴───────┘                         ║
║                              │                               ║
║                      status → completed / failed / timed_out ║
║                          (取决于 Job 原始结果)                ║
║                              ▼                               ║
║                            结束                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 异常路径汇总

| 异常场景 | error_phase | error_message 来源 | 处理方式 |
|----------|-------------|-------------------|----------|
| node pool 创建超时 | provisioning | GKE API 超时信息 | status→failed，无需清理 node pool |
| node pool 创建失败 | provisioning | GKE API 错误响应（quota 不足、reservation 无效等） | status→failed |
| Job 提交失败 | dispatching | K8s API 错误响应（资源冲突、权限不足等） | status→failed，触发 node pool 删除 |
| Job 执行失败 | running | Job condition message + Pod 事件（OOM、镜像拉取失败等） | Controller 删除 Job→检查复用→无则删除 node pool→status→failed |
| Job 超时 | running | 超时信息（dispatched_at + timeout_seconds） | 取消 Job→Controller 删除 Job→检查复用→无则删除 node pool→status→timed_out |
| node pool 删除失败 | deprovisioning | GKE API 错误响应 | 重试，最终失败则追加 error_message，status 仍按 Job 原始结果设置 |
| 孤儿 node pool | —（启动时） | — | 扫描 `falcon-*` 前缀的 node pool，无对应非终态 job 则清理 |

## 实施阶段

### Phase 1: 核心循环（MVP）

- Controller 框架：Task Poller + Job Submitter + Status Reconciler
- 多集群支持（clusters 注册表）
- 集群选择（TPU 类型匹配 + node pool 复用优先 + 队列深度均衡）
- Node pool 按需创建与延迟删除复用
- 数据库 schema（`clusters` 表由 Falcon 管理，`job` 表由上层定义）
- 基础 Job 渲染
- 状态回写
- 日志通过 GCS FUSE Pod 内重定向写入 GCS
- 使用 poc-tpu-partner/tpuv7x-64-node 作为首个边缘集群验证

### Phase 2: 健壮性

- 重试逻辑（可配置退避策略）
- 超时强制执行
- 优雅关闭（drain in-flight reconciliation）
- 健康检查 / readiness probe

### Phase 3: 可观测性

**任务事件上报**: controller 在任务状态流转时写入事件，供上层组件查询任务进度，减少手动 describe job / 集群操作。事件示例：

- "排队中，队列位置 #3"
- "正在创建 node pool (reservation)..."
- "reservation 不足，fallback 到 flex-start..."
- "Node pool 就绪，提交 Job..."
- "Pod pending，等待资源调度..."

**集群与资源观测**: controller 暴露各集群维度的资源信息：

- Reservation 剩余容量（GCP Compute API `count - inUseCount`）
- 各集群活跃 node pool 数量及拓扑
- 各 (cluster, topology) 队列深度和 running job 数

**暴露方式**: TBD——可能通过 HTTP/gRPC 接口供上层 API Server 采集，也可能直接作为 Prometheus metrics 暴露

**其他**:

- 结构化日志（任务生命周期事件）
- Prometheus 指标（下发延迟、任务耗时、失败率）
- 审计日志
- Grafana 面板

### Phase 4: 高级特性

- 优先级调度（跨集群任务优先级）
