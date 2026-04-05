# Beaver GitHub 集成规范 (GitHub Integration Specs)

Beaver 以 GitHub App 身份接入目标仓库，通过 Webhook 事件监听与 API 主动查询的双通道机制，实现 SRA（感知-推理-行动）框架中对 GitHub 数据的完整覆盖。本文档定义了 Beaver 与 GitHub 的集成架构、事件订阅、API 交互模式及数据模型映射。

## 1. GitHub App 身份与权限

Beaver 以 **GitHub App**（而非 OAuth App 或 Personal Access Token）身份运行。GitHub App 提供细粒度权限控制、独立 Bot 身份、Webhook 自动配置及更高的 API 速率限额。

### 1.1 Repository 权限

| 权限 | 级别 | 用途 |
| :--- | :--- | :--- |
| **Issues** | Read & Write | 读取 Issue 详情；添加/移除标签；发表评论 |
| **Pull requests** | Read & Write | 读取 PR Diff/Review；发表评论；关联 Issue |
| **Checks** | Read & Write | 创建 Check Runs 实现合并门禁 |
| **Contents** | Read-only | 读取仓库文件（如 README 中的 Beaver 配置） |
| **Metadata** | Read-only | 基础仓库元数据（GitHub App 默认授予） |

### 1.2 Organization 权限

| 权限 | 级别 | 用途 |
| :--- | :--- | :--- |
| **Projects** | Read & Write | 读写 ProjectsV2 看板项、自定义字段 |
| **Members** | Read-only | 解析 @mention 和 assignee 对应的团队成员 |

### 1.3 Webhook 安全

所有 Webhook 请求必须通过 `X-Hub-Signature-256` 签名校验（HMAC-SHA256），拒绝任何签名不匹配的请求。

## 2. 感知层 — Webhook 事件订阅

Beaver 通过 Webhook 实时捕获 GitHub 事件，作为 SRA 感知层的核心输入。以下是需要订阅的事件及其与 Beaver 业务能力的映射。

### 2.1 Issue 相关事件

| 事件 | Action | 触发的 Beaver 能力 |
| :--- | :--- | :--- |
| `issues` | `opened` | LLM 初筛（建议 `size/`、`type/`、assignee）；加入晨间管理者调度池 |
| `issues` | `labeled` / `unlabeled` | 状态守门员校验（Status Guard）；SOP 流转触发 |
| `issues` | `assigned` / `unassigned` | 更新晨间 Focus 待办清单 |
| `issues` | `milestoned` / `demilestoned` | DDL 提取与进度风险梯度计算 |
| `issues` | `closed` / `reopened` | 完成度统计；迭代复盘数据采集 |

### 2.2 Pull Request 相关事件

| 事件 | Action | 触发的 Beaver 能力 |
| :--- | :--- | :--- |
| `pull_request` | `opened` / `synchronize` / `reopened` / `ready_for_review` | LOC 校验（规模超限检测）；关联 Issue 检查 |
| `pull_request` | `review_requested` | 实时 Review 邀请通知 |
| `pull_request` | `closed` (merged) | Cycle Time 计算；日报成果采集 |
| `pull_request_review` | `submitted` | 更新 Review 状态；检查是否可解除合并门禁 |
| `check_suite` / `check_run` | `completed` | CI 状态感知；CI 连续失败告警 |

### 2.3 Project 相关事件

| 事件 | Action | 触发的 Beaver 能力 |
| :--- | :--- | :--- |
| `projects_v2_item` | `created` / `edited` / `deleted` | 看板视图同步（Project 为视图层，非真理源） |

### 2.4 Milestone 事件

| 事件 | Action | 触发的 Beaver 能力 |
| :--- | :--- | :--- |
| `milestone` | `closed` | 触发迭代复盘自动化（Sprint Retrospective） |

### 2.5 Issue Comment 事件

| 事件 | Action | 触发的 Beaver 能力 |
| :--- | :--- | :--- |
| `issue_comment` | `created` / `edited` | 解析 `DDL: YYYY-MM-DD` 语义；解析 `Depends on #ID` 依赖关系；测试结论证据扫描 |

## 3. 查询层 — API 交互模式

当 Webhook 事件触发后，Beaver 通过 API 拉取更完整的上下文。定时任务（晨间提醒、日报、周报）也依赖 API 批量查询。

### 3.1 上下文补全（事件驱动）

当感知到关键事件时，Beaver 按需拉取关联上下文：

| 触发场景 | API 调用 | 协议 | 返回数据 |
| :--- | :--- | :--- | :--- |
| Issue 标签变更 | 获取 Issue 完整详情 + 标签列表 + 关联 PR | REST `GET /repos/{owner}/{repo}/issues/{number}` | 当前所有标签、assignee、milestone |
| PR 打开/同步 | 获取 PR Diff 统计 | REST `GET /repos/{owner}/{repo}/pulls/{number}/files` | 各文件变更行数，用于 LOC 校验 |
| PR 打开 | 检查 Issue 关联 | 解析 PR body 中的 `Closes #N` / `Fixes #N` | 关联的 Issue 编号 |
| Issue 评论新增 | 获取评论内容 | Webhook payload 自带 | `DDL: YYYY-MM-DD` 或 `Depends on #N` 语义 |

### 3.2 批量查询（定时驱动）

定时任务通过 GraphQL 高效批量获取数据：

| 定时场景 | 查询内容 | 协议 |
| :--- | :--- | :--- |
| 晨间 Focus | 按 assignee 查询所有 `status/in-progress` 和 `status/ready-to-develop` 的 Issue | GraphQL（按标签 + assignee 过滤） |
| 晨间管理者池 | 查询所有 `status/triage` 或无 assignee 的 Issue | GraphQL |
| 日报 | 过去 24h 内状态变更的 Issue/PR 列表 | GraphQL（按 `updatedAt` 过滤） |
| 周报 | 本周 Milestone 进度、PR Cycle Time 统计 | GraphQL（聚合查询） |
| Stale 检测 | 查询在 `in-progress`/`review-needed` 状态停留超过阈值的 Issue | GraphQL（按标签 + `updatedAt` 过滤） |

### 3.3 GraphQL vs REST 选用原则

| 场景 | 推荐协议 | 原因 |
| :--- | :--- | :--- |
| 单个实体详情获取 | REST | 简单直接，缓存友好 |
| 批量多字段查询 | GraphQL | 一次请求获取多维数据，减少网络开销 |
| Project V2 操作 | GraphQL（推荐） | 支持 GraphQL 与 REST，GraphQL 查询更灵活 |
| Check Runs 创建/更新 | REST（推荐） | 支持 GraphQL 与 REST，REST 调用更简洁 |

## 4. 行动层 — 写入操作清单

Beaver 对 GitHub 的写入操作分为三类：**标签管理**、**评论输出**、**合并门禁**。

### 4.1 标签管理

| 操作 | API | 触发条件 | 示例 |
| :--- | :--- | :--- | :--- |
| 添加 `beaver/*` 标签 | REST `POST /repos/{owner}/{repo}/issues/{number}/labels` | 检测到合规问题或异常状态 | 添加 `beaver/stale`、`beaver/needs-split`、`beaver/missing-test` 等 |
| 移除 `beaver/*` 标签 | REST `DELETE /repos/{owner}/{repo}/issues/{number}/labels/{name}` | 合规问题已修复 | 开发者补充测试证据后移除 `beaver/missing-test` |
| 修改 `status/*` 标签 | 移除旧标签 + 添加新标签 | 特定自动化流转条件满足 | Stale 超阈值后自动从 `status/in-progress` 切换到 `status/blocked` |

### 4.2 评论输出

| 场景 | API | 评论内容 |
| :--- | :--- | :--- |
| 合规性指引 | REST `POST /repos/{owner}/{repo}/issues/{number}/comments` | 违规项说明 + 修正建议 + 相关文档链接 |
| 拆解审计反馈 | 同上 | LLM 对子任务覆盖度、原子性、测试定义的评估结果 |
| 上下文补全要求 | 同上 | 要求开发者补充关联 Issue 编号或 PRD/RFC 链接 |
| DDL 风险提醒 | 同上 | 临近 DDL 或已逾期的进度询问 |

### 4.3 合并门禁（Check Runs）

| 操作 | API | 说明 |
| :--- | :--- | :--- |
| 创建 Check Run | REST `POST /repos/{owner}/{repo}/check-runs` | PR 打开时创建 "Beaver Compliance" Check |
| 更新 Check Run 状态 | REST `PATCH /repos/{owner}/{repo}/check-runs/{check_run_id}` | 校验通过标记 `conclusion: success`；不通过标记 `conclusion: failure`，附带不通过原因 |

**Check Run 校验项**：

- PR 关联的 Issue 必须具备 `type/` 和 `size/` 标签
- PR 关联 Issue 的 `status/` 必须处于 `review-needed`
- 核心目录代码变更 LOC ≤ 200（排除测试、文档、自动生成文件）
- PR 描述中包含关联的 Issue 编号

## 5. 数据模型映射

定义 GitHub 实体与 Beaver 内部概念的对应关系，确保团队对术语和数据流有统一理解。

### 5.1 核心实体映射

| GitHub 实体 | Beaver 概念 | 映射关系 |
| :--- | :--- | :--- |
| Repository | 项目 (Project) | 1:1，一个 Repo 对应一个 Beaver 项目实例 |
| Issue | 任务 (Task) | 1:1，Issue 是 Beaver 追踪的最小可管理单元 |
| Issue (with Task List) | 父任务 (Parent Task) | 包含 `- [ ] #N` 子任务列表的 Issue 视为父任务 |
| Pull Request | 代码交付 (Delivery) | N:1，一个 Issue 可关联多个 PR |
| Label (`status/*`) | 生命周期状态 | 标签为状态真理源，驱动 SRA 全流程 |
| Label (`type/*`) | 任务类型 | 不可变分类属性 |
| Label (`size/*`) | 任务规模 | 决定适用的 SOP 流程（标准/快速路径） |
| Label (`p*/*`) | 优先级 | 决定推理权重与通知紧急度 |
| Label (`beaver/*`) | 系统标记 | Beaver 自动挂载的合规/风险标记 |
| Milestone | 迭代 (Sprint) | 1:1，继承截止日期作为 DDL |
| ProjectsV2 | 看板视图 (Board View) | 仅作为视图层，不作为状态真理源 |
| Issue Comment | 事件上下文 | 用于语义解析（DDL、依赖关系、测试证据） |
| Check Run | 合规门禁 | Beaver 创建的自动化校验结果 |

### 5.2 关键语义约定

| 语义 | 在 GitHub 中的表达 | Beaver 解析方式 |
| :--- | :--- | :--- |
| 任务 DDL | Issue 描述或评论中的 `DDL: YYYY-MM-DD` | 正则匹配 + Milestone 截止日期继承 |
| 任务依赖 | Issue 描述中的 `Depends on #N` | 正则匹配，构建有向依赖图 |
| PR-Issue 关联 | PR body 中的 `Closes #N` / `Fixes #N` / `Resolves #N` | GitHub 原生解析 + 正则匹配兜底 |
| 测试证据 | Issue 评论或 PR 中的测试报告链接、截图、`Test Passed` 日志 | LLM 语义识别 |
| 子任务列表 | Issue body 中的 GitHub Task List `- [ ] #N` | 正则解析 checkbox 条目 |
