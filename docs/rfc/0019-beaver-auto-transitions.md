---
title: "RFC-0019: Beaver 自动迁移基础设施"
status: draft
author: sii-xinglong
date: 2026-04-24
reviewers: []
---

# RFC-0019: Beaver 自动迁移基础设施

## 概述

在 Cloudflare Worker 内实现 4 条系统触发的 Project V2 #14 `Status` 字段自动迁移规则——Iteration 加入、Design Doc PR 合并、SubTask 全关、assignee + Start date 触达——使现行项目管理流程不再依赖团队成员手动切换 Status。

本 RFC 同时为后续 Goal「删除/重构历史标签写 transition」做铺垫：建立 Project V2 字段写的共享 helper、admin replay 端点与测试基线，使下一步把 `iteration-design.ts`（原 `milestone-design.ts`，见 §「命名」）与 `design-doc-merged.ts` 中的 `status/*` 标签写改为字段写、并删除标签写路径，无需再次新增基础设施。

## 背景

### 技术环境

- **运行载体**：Cloudflare Worker，TypeScript（`package.json` 声明 `wrangler ^4.14.0` / `vitest ^3.1.1` / `zod ^4.3.6`）。
- **入口与分发**：`src/index.ts` 暴露 `fetch`/`scheduled`/`queue`；`src/router.ts` 验签后通过 `src/sensing/webhook.ts::classifyRoute` 决定走 fast-path 还是异步队列；`src/fast-path.ts::handleFastPath` 是 fast-path 总分发器。
- **既有 2 条 Status 自动迁移**作为本次实现的精确先例（**均为待删除/重构的历史遗留**——本 RFC 范围内不删除，仅作为模式参考；删除/重构由后续 Goal 承接，本 RFC 在 §「系统边界」明确该 follow-up 动机）：
  - `src/acting/transitions/milestone-design.ts:30`（**拟改名为 `iteration-design.ts`**，见 §「命名」）在 `issues.milestoned` 上把 `size/L + status/triage` 迁到 `status/design-pending`（写**标签**）。
  - `src/acting/transitions/design-doc-merged.ts:26` 在 `pull_request.closed`（merged=true，仓库 = `primatrix/wiki`）上解析 PR body 中的 `owner/repo#N`（`src/acting/transitions/parse-issue-refs.ts`），跨仓把 `status/design-pending` 迁到 `status/ready-to-develop`（写**标签**）。
- **Project V2 GraphQL 既有读路径**：`src/queue/handlers/sync-project.ts:28-64` 用 `ProjectV2ItemFieldSingleSelectValue` 内联片段读 Status；当前仓库无字段**写**路径。
- **跨仓查找**：`design-doc-merged.ts:49` 通过 `findRepositoryByFullName` 在 D1 `repositories` 表里取 `github_installation_id`，该机制在本设计中复用。
- **GitHub App 订阅事件**（`README.md:140`）已包含 `Issues / Pull requests / Projects v2 items`；本设计无需新增订阅。
- **Worker 子请求上限**：50/请求（`design-doc-merged.ts:24` 的 `MAX_REFS_PER_RUN = 10` 即为该上限的体现）。

### 系统边界

- **In-scope 代码面**：`src/sensing/events.ts`（`FAST_PATH_ACTIONS`）、`src/fast-path.ts`（dispatcher）、新增 `src/acting/transitions/{iteration-set,subtask-closed,issue-assigned}.ts`、新增 `src/acting/github/project-v2.ts`、`src/admin/router.ts`（admin replay 端点）以及对应 `test/` 测试。
- **Out-of-scope 代码面**：`plugins/beaver/scripts/beaver-lib.sh`（命令侧，已有 set_status）；现有 `milestone-design.ts`（拟改名为 `iteration-design.ts`）与 `design-doc-merged.ts` 写标签的行为本设计**不删除**，留待后续 Goal 处理（理由见下条「未来 Goal 动机」）。
- **未来 Goal 动机（本 RFC 是其铺垫）**：本 RFC 引入的 `setProjectStatus` helper、`getProjectFields` helper、KV 缓存策略、admin replay 端点、GraphQL fixture 测试基线，**目的之一**是为后续「删除历史标签写 transition、把 `iteration-design.ts` 与 `design-doc-merged.ts` 改为只写 Project V2 字段、清理所有 `status/*` 标签存量」的 Goal 提前铺好基础设施。届时只需替换两个 handler 的写入面、删除 `addLabels/removeLabel` 调用、运行一次性回填脚本，即可完成 RFC-0013 §1 声明的「字段为 canonical store」终态。该 follow-up Goal 不在本 RFC 内执行。

### 客观背景事实

- 来源 spec：`~/Code/wiki/docs/onboarding/project-management.md` L161（a-1）/ L195（a-2）/ L62、L83、L227（a-3）/ L175（b）。
- 上游 RFC：`~/Code/wiki/docs/rfc/0013-beaver-commands-realignment.md` L44（(a) out-of-scope）/ L45（(b) out-of-scope）。本 RFC 是 0013 显式承诺的「后续 Goal」。
- §7 Q&A 已确认：venue = Worker fast-path；Status 写入面 = Project V2 字段；(a-1) trigger = `projects_v2_item.edited`；`Start date` 字段已在 Project #14 上配置。
- `events.ts:20` 已声明 `IssuesAction` 含 `assigned / closed`，但 `events.ts:52-55` 的 `FAST_PATH_ACTIONS` 未路由这两个 action；`projects_v2_item` 同样在 `WEBHOOK_EVENT_TYPES`（`events.ts:8`）中声明但未在 fast-path 出现——三者都需要由本 RFC 加入 `FAST_PATH_ACTIONS`。

## 方案

### 系统上下文

```text
┌──────────────────────────────────────────────────────────────────────┐
│ GitHub (primatrix/projects, primatrix/wiki, target repos)            │
│                                                                      │
│ Webhook events:                                                      │
│   issues.assigned ─────────┐                                         │
│   issues.closed ───────────┤                                         │
│   pull_request.closed (wiki) ─┤  (already wired in fast-path.ts;     │
│   projects_v2_item.edited ─┘    handler extended for a-2 field write)│
└─────────────────┬────────────────────────────────────────────────────┘
                  │ HMAC-verified webhook (router.ts)
                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Cloudflare Worker — fast-path                                        │
│                                                                      │
│  src/sensing/events.ts                                               │
│    FAST_PATH_ACTIONS += issues.assigned, issues.closed,              │
│                         projects_v2_item.edited                      │
│                                                                      │
│  src/fast-path.ts (dispatcher)                                       │
│    ├─► transitions/iteration-set.ts            (rule a-1)            │
│    ├─► transitions/design-doc-merged.ts        (rule a-2: existing   │
│    │     + NEW parallel field write)                                 │
│    ├─► transitions/subtask-closed.ts           (rule a-3)            │
│    └─► transitions/issue-assigned.ts           (rule b)              │
│                                                                      │
│  All four call:                                                      │
│    src/acting/github/project-v2.ts (NEW)                             │
│      ├─ getProjectFields(...) — Status / Size / Start date           │
│      └─ setProjectStatus(...) — updateProjectV2ItemFieldValue        │
│         IDs (project / item / field / option) cached in KV (1h TTL)  │
│                                                                      │
│  src/admin/router.ts                                                 │
│    POST /admin/transitions/replay                                    │
│      auth: ADMIN_API_TOKEN bearer                                    │
│      body: {rule, repo, issue_number}                                │
└──────────────────────────────────────────────────────────────────────┘
```

### 核心架构

四个 transition 处理器 + 一个共享字段写入 helper + 一个 admin 重放端点，全部沿用 `iteration-design.ts`（原 `milestone-design.ts`，本 RFC 一并改名，见 §「命名」）/ `design-doc-merged.ts` 设立的 fast-path 模式：纯决策函数 + side-effecting handler 分离。

### 命名

历史文件名 `src/acting/transitions/milestone-design.ts` 沿用「milestone」语义（GitHub 原生 Milestone 触发）已过时——Beaver 流程后续以 Project V2 `Iteration` 字段取代 GitHub Milestone 作为周期载体（参见 RFC-0013 §「Iteration 载体」与 spec `project-management.md`）。本 RFC 在实施计划 Phase 0 一并把该文件物理改名为 `iteration-design.ts`，同时更新 `src/sensing/events.ts` 与 `src/fast-path.ts` 中的 dispatch 引用、所有测试文件与 fixture 路径。本 RFC 文档全文已采用新名 `iteration-design.ts` 指代该 handler，描述其行为时仍保留「触发事件 = `issues.milestoned`」以反映其当前实现现状（该触发事件是否同步切换到 `projects_v2_item.edited` 由后续删除/重构 Goal 决定，不在本 RFC）。

### 接口与数据流

#### 共享字段读写 helper（src/acting/github/project-v2.ts，新增）

```typescript
export async function setProjectStatus(
  client: GitHubClient,
  env: Env,                          // for KV access
  owner: string,                     // primatrix
  projectNumber: number,             // 14
  repo: string,
  issueNumber: number,
  statusName: string,                // e.g. "Design Pending"
): Promise<{ written: boolean; reason: string }>;

export async function getProjectFields(
  client: GitHubClient, env: Env,
  owner: string, projectNumber: number,
  repo: string, issueNumber: number,
): Promise<{ status: string | null; size: string | null; startDate: string | null }>;
```

实现要点：

- **缓存分两层**（Issue 级 itemId 与项目级元数据生命周期不同，必须分键，否则不同 Issue 的 `itemId` 会在同一 key 下互相覆盖）：
  - 项目元数据：`projectId / Status fieldId / Size fieldId / Start date fieldId / 各 option id` 通过 `getProjectMetadata(...)` 解析后写入 KV，key = `pv2:project:{owner}/{projectNumber}`，TTL 1h。这些值与 Issue 无关，多 Issue 共享。
  - Item ID：`itemId` 通过 `getProjectItemId(repo, issueNumber, projectId)` 解析后写入 KV，key = `pv2:item:{owner}/{projectNumber}:{repo}#{issueNumber}`，TTL 1h。每个 Issue 独立缓存，避免覆盖。
- 写入用 `updateProjectV2ItemFieldValue` mutation；当 mutation 返回 `option_not_found` 类错误时，剔除项目元数据缓存（`pv2:project:*`）并重试一次；返回 `item not found` 类错误时，剔除对应 item 缓存（`pv2:item:*`）并重试一次。
- `Start date` 是 `ProjectV2ItemFieldDateValue`（不是 single-select），`getProjectFields` 在同一查询里同时使用 `ProjectV2ItemFieldSingleSelectValue` 与 `ProjectV2ItemFieldDateValue` 内联片段。

#### 各规则数据流

| 规则 | 触发事件 | 读 | 纯决策（pure function） | 写（决策为真时） |
|---|---|---|---|---|
| (a-1) | `projects_v2_item.edited`，且 `changes.field_value.field_node_id` 指向 `Iteration` 字段，且 `projects_v2_item.content_type = "Issue"` | `getProjectFields(...)` 取 `Status` / `Size` | `Size == "L" && Status ∈ {"Triage", "Ready to Claim"}` | `setProjectStatus(issue, "Design Pending")` |
| (a-2) | `pull_request.closed`，`merged=true`，仓库 = `primatrix/wiki`（已由 `fast-path.ts:278-286` 路由） | `parseIssueRefs(pr.body)`（已存在）+ 对每个 ref 现有的标签读取 + **新增** `getProjectFields(...)` 取 Status | per ref: `Status == "Design Pending"` | **现有标签写保留**；**新增** `setProjectStatus(ref, "Ready to Develop")` 并行写入 |
| (a-3) | `issues.closed` | `GET /repos/{owner}/{repo}/issues/{n}/parent`（sub-issues REST API，2025-09-11 GA，需附 `Accept: application/vnd.github+json` 与 `X-GitHub-Api-Version: 2022-11-28` 头）→ 若有 parent，`getProjectFields(parent)` 取 `Status` / `Size`；`GET /repos/{owner}/{repo}/issues/{parent.number}/sub_issues?per_page=100` 取所有子 Issue 的 `state` | `parent.Size == "L" && parent.Status == "In Progress" && all_subtasks.state == "closed"` | `setProjectStatus(parent, "Done")` |
| (b) | `issues.assigned` | `getProjectFields(...)` 取 `Status` / `Size` / `Start date` | `Size == "S" && Status == "Ready to Claim"` → 真；`Size == "L" && Status == "Ready to Develop" && Start date != null && today_utc >= Start date` → 真；其它一律假 | `setProjectStatus(issue, "In Progress")` |

#### 纯决策函数签名（每个 rule 一个）

```typescript
// src/acting/transitions/iteration-set.ts
export function shouldTransitionToDesignPendingByIteration(
  fields: { status: string | null; size: string | null },
  changedFieldName: string,
): boolean;

// src/acting/transitions/subtask-closed.ts
export function shouldTransitionParentToDone(
  parent: { status: string | null; size: string | null },
  subtaskStates: Array<"open" | "closed">,
): boolean;  // true iff parent.size === "L" && parent.status === "In Progress" && every subtask closed

// src/acting/transitions/issue-assigned.ts
export function shouldTransitionAssignedToInProgress(
  fields: { status: string | null; size: string | null; startDate: string | null },
  hasAssignee: boolean,
  todayUtc: string,                  // YYYY-MM-DD
): boolean;
// Truth table:
//   Size=S && Status==Ready to Claim                                   → true
//   Size=L && Status==Ready to Develop && startDate!=null && today>=startDate → true
//   anything else                                                       → false
```

#### Admin replay 端点

`src/admin/router.ts` 新增 `POST /admin/transitions/replay`：

- Auth：现有 `ADMIN_API_TOKEN` Bearer。
- Body：`{rule: "iteration-set" | "design-doc-merged" | "subtask-closed" | "issue-assigned", repo: string, issue_number: number}`。
- 行为：合成对应规则的最小 webhook payload（含 `repository`、`issue` / `pull_request` / `projects_v2_item` 字段），直接调用对应 transition handler；返回 `{written: boolean, reason: string}` 与执行日志。`reason` 取值枚举：`written`、`skipped:not_eligible`、`skipped:already_at_target`、`skipped:no_parent`（仅 a-3）、`skipped:subtasks_open`（仅 a-3）、`skipped:start_date_future`（仅 b）、`error:option_not_found_after_invalidation`、`error:graphql_failure`。

### 关键 trade-offs 与理由

1. **写 Project V2 字段而非 status/\* 标签**——新代码直接写 Project V2 #14 `Status` 字段（不写标签），与 `iteration-design.ts:44-45`（原 `milestone-design.ts`，本 RFC Phase 0 改名）和 `design-doc-merged.ts:74-75` 写标签的现有 2 条 transition 模式不一致。理由：RFC-0013 §1 已声明字段是 canonical store；新代码不应往老方向走。副作用：`sync-project.ts` 将检测到字段↔标签不一致并日志记录，**不会**自动修复——这是已知遗留，留给未来「删除/重构现有 2 条标签写 transition」的 Goal 处理（本 RFC §「系统边界 — 未来 Goal 动机」已说明该 follow-up 关系）。
2. **(a-2) 双写例外**——`design-doc-merged.ts` 的现有标签写**保留**；本 RFC 在同一 handler 中**追加**字段写。理由：避免破坏当前依赖 `status/ready-to-develop` 标签的下游消费方。这是过渡期例外，不作为模式推广。
3. **KV 缓存 project / field / option IDs（TTL 1h）**——Project schema 几乎不变；缓存命中时单次 transition 只需 1 次 GraphQL 写。`setProjectStatus` 在拿到 `option_not_found` 类错误时剔除缓存并重试一次，避免脏数据陷在缓存里。参考 `src/acting/github/auth.ts` 的 JWT KV 缓存模式。
4. **预读 Status 实现幂等**（不引入 KV idempotency key）——每个 handler 在写入前先 `getProjectFields` 读当前 Status；与目标值相同则 no-op。完全沿用 `design-doc-merged.ts:67` 的先例。(a-3) 的 parent Status 读本来就在数据流里，零额外成本。
5. **`Start date` 缺失即软失败**——若 `getProjectFields` 解析不到 `Start date` 字段（field id 缺失），日志一次 `field_unresolved=Start date` 警告并返回 `null`；(b) 的 size/L 分支直接判定为不满足，相当于过渡期内不触发。避免「字段未配置时 Worker 抛异常导致整个 fast-path 失败」。

### 测试策略

每条规则三个测试文件，统一放在 `test/acting/transitions/`：

| 文件 | 内容 |
|---|---|
| `<rule>.decision.test.ts` | 纯决策函数的真值表覆盖（按前置条件矩阵枚举） |
| `<rule>.handler.test.ts` | mock `GitHubClient`，断言 (a) 读调用次数与参数，(b) 写调用是否发生，(c) GraphQL mutation 变量结构 |
| `<rule>.fixture.test.ts` | 用 `test/fixtures/project-v2/` 下的真实 GraphQL 响应快照重放，捕获 schema 漂移 |

加上路由层测试：

- `test/sensing/webhook.test.ts`：断言 `issues.assigned / issues.closed / projects_v2_item.edited` 进入 fast-path。
- `test/fast-path.test.ts`：断言 `(event, action)` 元组分发到正确 handler。
- `test/acting/github/project-v2.test.ts`：KV 缓存命中/未命中、ID 解析、`option_not_found` 失效重试、`Start date` 缺失软失败。

执行命令：`npm test`（CI 已在 `.github/workflows/deploy.yml` 中运行）。

### 部署依赖

1. **Webhook 订阅**——`Issues / Pull requests / Projects v2 items` 已订阅（`README.md:140`），无新增。
2. **`Start date` 字段**——已在 Project #14 上配置（§7 Q&A 第 4 题确认）。缺失时软失败，过渡期内非阻塞。
3. **D1**——无 schema 迁移；复用现有 `repositories` 表。
4. **环境变量**——`ADMIN_API_TOKEN`（已存在）保护 admin replay 端点；`OWNER`（`primatrix`）与 `PROJECT_NUMBER`（`14`）作为常量写在 `src/acting/github/project-v2.ts`，配置漂移风险低，代码内注释说明。**单板假设**：本设计假定 Project #14 是组织内唯一的 Beaver canonical board，所有需要被自动迁移的 Issue（无论位于哪个 registered repo）都已通过 GitHub App 安装映射并被加入到 Project #14。未在 Project #14 上的 Issue 在 `getProjectFields` 阶段会拿到 `item not found`，handler 应日志 `skipped:not_on_project_14` 并 return，不抛异常。
5. **跨仓 sub-issue 关系**——(a-3) 依赖 GitHub 2025-09-11 GA 的 sub-issues REST API（含 `GET /repos/{owner}/{repo}/issues/{n}/parent`，docs.github.com/en/rest/issues/sub-issues）；该 API 已支持跨仓 sub-issue。所有调用必须附 `Accept: application/vnd.github+json` 与 `X-GitHub-Api-Version: 2022-11-28` 头。Worker 拿到 parent 引用后通过现有 `findRepositoryByFullName` 解析跨仓 installation。
6. **CI**——`.github/workflows/deploy.yml` 现有流水线（lint + typecheck + test + deploy on push to main）覆盖本次新增；无 workflow 改动。
7. **GitHub Webhooks payload schema 演进**——`projects_v2_item.edited` 仍处 GitHub 公开预览（曾于 2024-06-27 增加 `from`/`to` 字段）。Worker 解析 `payload.changes.field_value.field_node_id` 时必须 `?? null` 容错，缺失时按「非 Iteration 字段编辑」短路 return，避免预览 schema 再变时整个 fast-path 抛异常。

### 备选方案

#### Alt 1 — 全 Actions 实现（`.github/workflows/*.yml`）

- 方法：4 条规则各用一个 GitHub Actions workflow，由 `on: issues / pull_request / projects_v2_item` 触发，workflow 内用 `actions/github-script` 或 `gh project` CLI 读字段写 Status。
- 优点：Worker 零新代码；per-repo onboarding 只需启用 workflow；日志在 GitHub UI 中可见。
- 拒绝原因：
  - **逻辑分裂**——既有 2 条 transition 在 Worker，再分一半到 Actions 会造成 FSM 真源分裂、双日志存储、事件序列同时触发时优先级歧义。
  - **无 D1 访问**——Actions 拿不到 `repositories` 表的 `installation_id` 映射，(a-2) 的跨仓写需另起炉灶。
  - **Per-repo onboarding 负担**——每加一个新仓都要复制 workflow；Worker 模型靠 GitHub App 安装自动覆盖。
- 决策依据：§7 S1.Q1 用户选 Worker fast-path。

#### Alt 2 — 写 status/* 标签（不写字段）

- 方法：复用 `addLabels / removeLabel`，与 `iteration-design.ts:44-45`（原 `milestone-design.ts`）完全一致。
- 优点：风险与代码量最低；无需 KV 缓存；与现有 2 条 transition 模式统一。
- 拒绝原因：
  - **写入已废弃存储**——RFC-0013 §1 声明字段为 canonical；新代码继续写标签是反向移动。
  - **未来双重迁移**——本来只需迁 2 条现有 transition，现在变成 6 条。
- 决策依据：§7 S1.Q2 用户选字段写。
- 例外：(a-2) 保留标签写并**新增**字段写并行（见 trade-off #2）。

#### Alt 3 — 给 (a-1) 用 cron 周期性扫描

- 方法：扩展 `sync-project.ts`（`README.md:79` 现已 `30 3 * * *` 每天跑一次），让它在 `Iteration set ∧ Status ∈ {Triage, Ready to Claim} ∧ Size = L` 时直接写 `Status = Design Pending`。
- 优点：对 webhook 漏投鲁棒；复用已有 GraphQL 分页代码。
- 拒绝原因：
  - **cron 间隔的延迟下限**——目前每日 11:30 CST 一次，对用户主动设置 Iteration 后立即期望生效的场景体感差。
  - **职责混淆**——`sync-project.ts` 当前只检测+日志，不写。把它改成「检测+写」会改变它的契约，应另立设计。
  - **Webhook 已经可用**——`projects_v2_item` 已在 `WEBHOOK_EVENT_TYPES`（`events.ts:8`），仅缺 dispatch。
- 决策依据：§7 S1.Q3 用户选 webhook。

#### Alt 4 — 在 D1 缓存 child→parent 映射（用于 a-3）

- 方法：新增 D1 表 `sub_issue_parent (child_repo, child_number, parent_repo, parent_number)`，由 `issues.opened` / sub-issue 创建事件填充；(a-3) 直接 D1 查询。
- 优点：close path 上零 API 调用，最快、最省 subrequest。
- 拒绝原因：
  - **schema 迁移 + 回填负担**——必须把组织里既有所有 parent↔child 关系回填一次；漏一个就永久空洞。
  - **D1 单调增长**——需 TTL 或关闭级联清理；与 GitHub 真源一致性还是要周期性核对。
  - **一次 API 调用很便宜**——Worker subrequest cap = 50；(a-3) 总预算约 4 次（parent fetch + parent Status 读 + sub-issues 列表 + Status 写），余量大。
- 决策依据：§7 S3.Q2 用户选原生 sub-issues parent endpoint。

## 影响范围

- **新增代码面**：`src/acting/github/project-v2.ts`（新文件）；`src/acting/transitions/{iteration-set,subtask-closed,issue-assigned}.ts`（新文件）；`src/admin/router.ts`（新增 `POST /admin/transitions/replay` handler）。
- **修改代码面**：`src/sensing/events.ts`（`FAST_PATH_ACTIONS` 在 `issues` 集合里增加 `assigned` / `closed` 两个 action，同时**新增** `projects_v2_item` 顶层 key 包含 `edited` action——后者目前不存在该 key）；`src/fast-path.ts`（dispatcher 增三个分支 + 在已有的 wiki PR 合并分支上追加 `setProjectStatus` 调用 + 同步更新 `iteration-design.ts` 的 import 路径）；`src/acting/transitions/design-doc-merged.ts`（在循环内追加并行字段写）。
- **重命名**：`src/acting/transitions/milestone-design.ts` → `iteration-design.ts`（Phase 0；含同步更新 events.ts/fast-path.ts 引用、`test/acting/transitions/milestone-design.*.test.ts` 测试文件与 fixture 路径）。文件内部行为不变。
- **测试新增**：4 × 3 = 12 个 transition 测试 + 2 个路由测试更新 + 1 个 helper 测试 + fixtures 目录。
- **不影响**：`plugins/beaver/scripts/beaver-lib.sh`（命令侧）；`status/*` 标签流（继续由 `iteration-design.ts` / `design-doc-merged.ts` 现有逻辑维护）；D1 schema；wiki 仓库。
- **下游消费方**：依赖 Project V2 #14 `Status` 字段的 dashboard / `sync-project.ts` 的 mismatch 检测、用户的看板视图——本次后将在更多场景下看到字段被自动更新；依赖 `status/*` 标签的下游消费方对本次变更无感（除 (a-2) 因双写两路都更新）。

## 实施计划

| Phase | 内容 | 验收 |
|---|---|---|
| Phase 0 | `git mv src/acting/transitions/milestone-design.ts iteration-design.ts`，同步更新 `src/sensing/events.ts` / `src/fast-path.ts` 中的 import & dispatch 路径，重命名 `test/acting/transitions/milestone-design.*.test.ts` 与对应 fixture 目录 | `npm test` 通过；grep 仓库无残留 `milestone-design` 字面引用 |
| Phase 1 | `src/acting/github/project-v2.ts`（含 KV 缓存与软失败）+ 对应单元测试 | `npm test test/acting/github/project-v2.test.ts` 通过 |
| Phase 2 | (a-1) `iteration-set.ts` 决策 + handler + fixture + 路由 | 沙箱 Issue 设置 Iteration 字段后 Project V2 Status 切到 `Design Pending` |
| Phase 3 | (b) `issue-assigned.ts` 决策 + handler + fixture + 路由 | 沙箱 size/S Issue assign 后切到 `In Progress`；size/L 在 `Start date` 未到时不切，到达后切 |
| Phase 4 | (a-3) `subtask-closed.ts` 决策 + handler + fixture + 路由 | 沙箱父 size/L Task 的最后一个 SubTask 关闭后父切到 `Done` |
| Phase 5 | (a-2) `design-doc-merged.ts` 追加字段并行写 + 测试更新 | 沙箱 wiki RFC PR 合并后：(a) 关联 Issue 的 Project V2 `Status` 字段切到 `Ready to Develop`；(b) 现有 `status/ready-to-develop` 标签写保持不变。当现有标签前置条件失败（Issue 不在 `status/design-pending`）时，新字段写也跟随 skip——保持单一 skip 决策路径，不引入独立判断 |
| Phase 6 | `src/admin/router.ts` `POST /admin/transitions/replay` + 文档 | 用 `curl` 携带 `ADMIN_API_TOKEN` 重放上述 4 条规则成功 |

每个 Phase 一个独立 SubTask、独立 PR、独立合并；本 RFC 合并后由 `/beaver-decompose` 拆分为对应 SubTask。

## 风险

- **风险 R1：字段↔标签不一致扩大**。本 RFC 写字段不写标签，新代码触发的 transition 之后，Issue 的 Project V2 `Status` 与 `status/*` 标签将持续不同步。**缓解**：`sync-project.ts` 已能日志记录这种不一致；该不一致的最终归零依赖后续「删除/重构 `iteration-design.ts` 与 `design-doc-merged.ts` 标签写」Goal（见 §「系统边界 — 未来 Goal 动机」）。下游若仍依赖标签，需在该 follow-up Goal 启动前明确告知。可在监控里加一条 mismatch count 阈值告警。
- **风险 R2：`projects_v2_item.edited` 高频触发**。Project V2 任何字段编辑都触发同一事件（filter 必须靠 `changes.field_value.field_node_id`）。Worker 需在 dispatcher 入口快速短路非 Iteration 字段的事件，避免每次都走 `getProjectFields`。**缓解**：handler 第一步即从 payload 中提取 `changes` 与 `field_node_id` / 字段名，命中 Iteration 才往下走；其它字段直接 return。需在 fixture 测试中覆盖「非 Iteration 字段编辑」用例确保 no-op。
- **风险 R3：`option_not_found` 缓存失效循环**。若 Project 的 Status 选项重命名/删除，`setProjectStatus` 会拿到 `option_not_found` 错误；缓存失效重试一次仍然失败时必须停止重试，否则可能在每次事件上无限重试。**缓解**：单次事件最多一次缓存失效重试；二次失败直接日志 `error=option_not_found_after_invalidation`，return；不影响后续事件处理。
- **风险 R4：(a-3) 同一父 Task 多次重入**。同一父 Task 下若多个 SubTask 在短时间内关闭，每个 close 事件都会跑一次 sub-issues 列表查询并尝试写 Status；最后那个会真正切到 Done，其它 no-op（因 `parent.Status == Done` 已经被前一次写入）。**缓解**：`shouldTransitionParentToDone` 要求 `parent.Status == "In Progress"`，前一次写入后 parent 已是 `Done`，重入直接判假；预读 Status 进一步提供幂等。竞态窗口理论上存在但写 mutation 是 idempotent，影响仅限重复一次写入。
- **风险 R5：Worker subrequest 上限**。最重的路径是 (a-3)：`parent fetch (1) + getProjectFields(parent) (1 GraphQL with multiple fragments) + sub_issues list (1, may paginate) + setProjectStatus (1 GraphQL mutation) = ~4 calls`，余量充足；但若某父 Task 子 Issue 数量极多（>100）需要分页时可能超量。**缓解**：参考 `design-doc-merged.ts:24` 的 `MAX_REFS_PER_RUN = 10` 模式，给 sub_issues 分页设上限（首页 100 已覆盖绝大多数情形）；超出上限时降级为「不能确定全部已关闭」并 log + skip，留给下次 close 事件复算。
- **风险 R6：(b) 中 `Start date` 后置设置不触发**。AC #4 仅要求「`Start date` 仍在未来或未设时，`Status` 保持不变」，未要求事后 `Start date` 触达时主动迁移。当下序列「先 assign → 后设 Start date 到今天」会让 size/L Task 永远停在 `Ready to Develop`。**缓解**：先按 AC 字面执行，本 Goal 不为 Start date 字段编辑加 `projects_v2_item.edited` 第二个分支；明确该路径需用户在设完 Start date 后手动 unassign+reassign 触发，或留待后续「Start date 字段写入触发」Goal 处理。运营层面在 admin replay 端点上提供 `b` 规则的手动重放兜底。

### 范围说明：spec `M/L/XL` vs 本 RFC `size/L`

spec L161 的 (a-1) 与 L195 的 (a-2) 描述均覆盖 `Size = M / L / XL` 三档；本 RFC 与 Issue #130 显式收窄到 `size/L`——理由是 RFC-0013 §「Size 载体」决定命令侧仅实现 `S / L` 子集（详见 0013 §「与 spec 的差异」）。在 `size/M` 和 `size/XL` 的命令侧支持落地之前，本设计的纯决策函数硬编码 `Size == "L"`；未来扩展至 `Size ∈ {M, L, XL}` 只需修改决策函数返回条件，handler / helper / 路由层均无需变动。

<!-- provenance
- "Cloudflare Worker, TypeScript, wrangler/vitest/zod versions" ← Discovery D3 (package.json)
- "src/sensing/webhook.ts::classifyRoute" ← Read of src/sensing/webhook.ts:79-87
- "src/fast-path.ts::handleFastPath" ← Read of src/fast-path.ts:241
- "src/acting/transitions/milestone-design.ts:30 (issues.milestoned, size/L+status/triage → status/design-pending)" ← Read of src/acting/transitions/milestone-design.ts:30-46
- "src/acting/transitions/design-doc-merged.ts:26 (pull_request.closed for primatrix/wiki, parses owner/repo#N)" ← Read of src/acting/transitions/design-doc-merged.ts:26-83
- "MAX_REFS_PER_RUN=10 stays within Cloudflare Workers' 50-subrequest limit" ← Read of src/acting/transitions/design-doc-merged.ts:20-24
- "src/queue/handlers/sync-project.ts uses ProjectV2ItemFieldSingleSelectValue inline fragment to read Status" ← Read of src/queue/handlers/sync-project.ts:50-54, 92-94
- "design-doc-merged.ts:49 cross-repo lookup via findRepositoryByFullName" ← Read of src/acting/transitions/design-doc-merged.ts:49
- "GitHub App subscriptions include Issues / Pull requests / Projects v2 items" ← Discovery D3 (README.md:140)
- "src/sensing/events.ts FAST_PATH_ACTIONS does NOT include issues.assigned/closed or projects_v2_item.*" ← Read of src/sensing/events.ts:50-55
- "issues.assigned and issues.closed declared in IssuesAction" ← Read of src/sensing/events.ts:18-20
- "projects_v2_item is in WEBHOOK_EVENT_TYPES" ← Read of src/sensing/events.ts:8
- "spec L161 / L195 / L62 / L83 / L227 / L175" ← Read of ~/Code/wiki/docs/onboarding/project-management.md L161, L195, L62, L83, L227, L175
- "RFC-0013 §「范围」L44-L45 declares (a) (b) out-of-scope" ← Read of ~/Code/wiki/docs/rfc/0013-beaver-commands-realignment.md:44-45
- "Implementation venue = Worker fast-path" ← QA Section 1 Q1
- "Status write surface = Project V2 Status field only" ← QA Section 1 Q2
- "(a-1) trigger event = projects_v2_item.edited" ← QA Section 1 Q3
- "Start date already exists on Project V2 #14" ← QA Section 1 Q4
- "Latency = behavior-only (no SLO floor)" ← QA Section 2 Q1
- "Non-goals = NOT migrate existing transitions / NOT Blocked toggles / NOT write Start date" ← QA Section 2 Q2
- "Verification = unit + mocked + smoke + admin replay endpoint" ← QA Section 2 Q3
- "Field-write helper = standalone module + KV-cached ID resolution" ← QA Section 3 Q1
- "(a-3) child→parent via native sub-issues parent endpoint" ← QA Section 3 Q2
- "(b) preconditions via single GraphQL query" ← QA Section 3 Q3
- "Idempotency = pre-write Status check, no KV idempotency keys" ← QA Section 3 Q4
- "Test surface = pure decision + mocked handler + routing + GraphQL fixture replay" ← QA Section 3 Q5
- "Section 4 enumerates 4 main rejected alternatives" ← QA Section 4 Q1
- "ProjectV2ItemFieldDateValue inline fragment for Start date" ← GitHub Projects V2 GraphQL API documentation (external reference, not in repo)
- "GET /repos/{owner}/{repo}/issues/{n}/parent for sub-issue parent lookup" ← GitHub sub-issues REST API GA 2025-09-11 (docs.github.com/en/rest/issues/sub-issues; verified during PR review)
-->
