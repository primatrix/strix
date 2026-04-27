---
title: "RFC-0021: Beaver 自动指派 Reviewer——按组负载均衡"
status: draft
author: sii-xinglong
date: 2026-04-26
reviewers: []
---

# RFC-0021: Beaver 自动指派 Reviewer——按组负载均衡

## 概述

当 PR 从 Draft 转为 Open 或直接以 Open 状态创建时，Beaver 从 `beaver-config` 中的 `reviewerGroups` 配置读取分组成员，按组各选一名近 7 天 review 工作量最少的成员，通过 GitHub API 指派为 Reviewer。

## 背景

### 技术现状

Beaver 是基于 Cloudflare Workers 的 GitHub App，采用 Sense→Reason→Act 架构处理 webhook 事件。当前 `pull_request.ready_for_review` 已注册在 `PullRequestAction` type union 中（`src/sensing/events.ts`），但尚未加入 `FAST_PATH_ACTIONS`，且 Reviewer 指派逻辑尚未落地到 main 分支。

### 系统边界

- `reviewerGroups` 配置存储在 Project V2 #14 README 的 `yaml beaver-config` 块中
- Reviewer 指派通过 `POST /repos/{owner}/{repo}/pulls/{n}/requested_reviewers` GitHub API 完成
- 处理流程采用异步 Queue（`assign_reviewer` task type），避免 webhook 响应超时

### 关键约束

- 不修改任何 Project V2 字段（Status / Size / Iteration 等）
- handler 不包含 Issue Status 迁移逻辑（由其他规则/手动完成）
- 不展开 GitHub Teams、不处理 @org/team 和 email owner

## 方案

### 核心架构

```text
Webhook (pull_request.ready_for_review / opened)
  → router.ts: 识别事件，fast-path 入队 assign_reviewer task
  → Queue consumer → queue/handlers/assign-reviewer.ts
    → 1. 从 repo_id 查询 DB 获取 owner/repo
    → 2. 读 beaver-config 获取 reviewerGroups
    → 3. 幂等检查：PR 已有 requested_reviewers → 跳过
    → 4. pickGroupedReviewers(groups, recentCounts, {exclude: author})
    → 5. POST /pulls/{n}/requested_reviewers → 指派
```

### 新增模块

| 文件 | 职责 |
|---|---|
| `src/acting/reviewer-assign.ts` | `pickGroupedReviewers`（按组 least-loaded 选择器）、`countRecentReviewsByUser`（7 天 review 计数） |
| `src/queue/handlers/assign-reviewer.ts` | Queue handler，编排 config 读取 + 幂等检查 + 选择器调用 + API 指派 |

### 变更模块

| 文件 | 变更 |
|---|---|
| `src/queue/types.ts` | 新增 `assign_reviewer` task type 及 payload：`{ pr_number: number }` |
| `src/queue/producer.ts` | 新增 `enqueueAssignReviewer` 函数 |
| `src/queue/router.ts` | 新增 `assign_reviewer` case |
| `src/fast-path.ts` | 新增 `ready_for_review` + `opened`（非 Draft）分支，入队 `assign_reviewer` |
| `src/sensing/events.ts` | `FAST_PATH_ACTIONS.pull_request` 增加 `ready_for_review` |

### reviewerGroups 配置 schema

在 Project V2 #14 README 的 `yaml beaver-config` 块中新增 `reviewerGroups` 字段：

```yaml
reviewerGroups:
  infra:
    - alice
    - bob
  ml:
    - charlie
    - dave
```

类型：`Record<string, string[]>`，组名为 key，成员 GitHub login 列表为 value。

解析逻辑：读取 Project V2 README → 提取 `yaml beaver-config` fenced block → YAML parse → 取 `reviewerGroups` 字段。缺失或格式错误时跳过指派并写 structured log。

### 接口设计

**`pickGroupedReviewers(groups, recentCounts, opts)`**

```typescript
interface PickGroupedOptions { exclude?: string }

function pickGroupedReviewers(
  groups: Record<string, string[]>,
  recentCounts: Map<string, number>,
  opts?: PickGroupedOptions,
): string[]
```

- 对每个组，排除 `opts.exclude`（PR 作者），按 `recentCounts` 升序排序；若计数相同，则通过随机或哈希平局逻辑取第一个
- 候选池为空的组跳过并 `console.log`
- 返回去重后的 reviewer login 数组

**`countRecentReviewsByUser(client, owner, repo, logins, now?)`**

```typescript
async function countRecentReviewsByUser(
  client: GitHubClient,
  owner: string,
  repo: string,
  logins: string[],
  now?: Date,
): Promise<Map<string, number>>
```

- 通过 GitHub Search API 批量查找 logins 成员近 7 天 review 的 PR（合并多个 `reviewer:login` 条件为单次查询）
- 对命中 PR 的 reviews 列表计数
- `now` 参数可注入，用于确定性测试

### 幂等机制

handler 入口通过 `GET /repos/{owner}/{repo}/pulls/{n}` 检查 `requested_reviewers` 数组：

- 非空 → 跳过本次指派，写日志 `[assign-reviewer] pr=#{n} skip=already-has-reviewers`
- 空 → 继续执行指派流程

### 关键 trade-off

| 决策 | 选择 | 理由 |
|---|---|---|
| 同步 vs 异步 | 异步 Queue | 避免 webhook 响应超时（多次 API 调用） |
| 专用 vs 复用 task type | 专用 `assign_reviewer` | 职责清晰、独立重试、不影响 LLM 分析 |
| CODEOWNERS vs reviewerGroups | 仅 reviewerGroups | 配置直接定义候选池，无需 gitignore pattern 解析的额外复杂度 |

### 测试策略

| 测试文件 | 覆盖范围 |
|---|---|
| `test/acting/reviewer-assign.test.ts` | 多组各选一人、空候选池跳过、排除作者、最少 review 优先+平局随机/哈希 |
| `test/queue/handlers/assign-reviewer.test.ts` | mock GitHubClient 的 handler 集成测试：幂等跳过、正常指派、reviewerGroups 缺失处理 |

### 部署

- 无新环境变量
- 无 D1 migration
- 需由管理员在 Project V2 #14 README 的 `beaver-config` 中配置 `reviewerGroups`

### 备选方案

1. **fast-path 同步处理**：否决——多次 GitHub API 调用可能超过 webhook 10s 响应窗口，导致 GitHub 重试。
2. **复用 analyze_event task type**：否决——LLM 分析与 reviewer 指派职责混合，且重试耦合（reviewer 指派失败不应重试 LLM 分析）。
3. **支持 CODEOWNERS @org/team owner**：否决——需展开 team membership、处理权限、处理嵌套 team，复杂度不值得，当前场景不需要。
4. **基于 CODEOWNERS 文件做候选池筛选**：否决——候选池由 `reviewerGroups` 配置直接定义即可，引入 CODEOWNERS 解析增加了不必要的复杂度（gitignore pattern 匹配、3 个候选路径查找）。
5. **reviewerGroups 缺失时回退到 CODEOWNERS**：否决——保持架构简洁，缺失时跳过指派并写日志。

## 影响范围

- **Beaver Worker 代码**：新增 2 个模块（reviewer-assign、assign-reviewer handler），变更 5 个模块（types、producer、router、fast-path、events）
- **Project V2 #14 README**：需新增 `reviewerGroups` 配置字段
- **所有使用 Beaver 的仓库**：PR 从 Draft→Open 或直接 Open 创建时将自动指派 Reviewer（前提是 `reviewerGroups` 已配置）
- **现有 PR 处理流程**：不影响——`opened`/`reopened`/`synchronize` 的现有 Check Run 和 Gate Check 逻辑不受影响

## 实施计划

| # | SubTask | 依赖 | 预期交付物 |
|---|---|---|---|
| 1 | 按组 Reviewer 选择器 | 无 | `src/acting/reviewer-assign.ts` + `test/acting/reviewer-assign.test.ts`（pickGroupedReviewers, countRecentReviewsByUser） |
| 2 | beaver-config reviewerGroups 读取 | 无 | 在现有 config 读取逻辑中新增 `reviewerGroups` 解析 + 测试 |
| 3 | Queue 基础设施 | 无 | `assign_reviewer` task type 注册（types.ts, producer.ts, router.ts） |
| 4 | assign-reviewer handler | 1, 2, 3 | `src/queue/handlers/assign-reviewer.ts` + `test/queue/handlers/assign-reviewer.test.ts`（编排逻辑 + 幂等 + 集成测试） |
| 5 | fast-path 接入 | 3 | `fast-path.ts` + `events.ts` 变更，`ready_for_review` / `opened` 入队 |

## 风险

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| GitHub Search API 速率限制 | 低 | `countRecentReviewsByUser` 合并多个 `reviewer:login` 条件为单次批量查询，大幅降低请求数 | 控制 `reviewerGroups` 成员总数在合理范围；handler 内 catch 错误写日志、不抛出 |
| Queue 处理延迟 | 低 | Cloudflare Queue 批处理间隔可能导致指派延迟数秒 | 可接受——异步指派不影响 PR 功能性 |
| `reviewerGroups` 配置错误 | 低 | 配置格式不符预期导致解析失败 | fallback 为跳过指派 + 写 structured log，不影响 PR 正常流程 |

<!-- provenance
- "Beaver 是基于 Cloudflare Workers 的 GitHub App" ← Discovery D3 README.md
- "Sense→Reason→Act 架构" ← Discovery D3 README.md
- "ready_for_review 已注册在 PullRequestAction type union 中，但尚未加入 FAST_PATH_ACTIONS" ← Discovery D2: src/sensing/events.ts line 24 (type) vs line 54 (FAST_PATH_ACTIONS)
- "现有 Queue 架构: types.ts, producer.ts, router.ts, consumer.ts" ← Discovery D2 file reads
- "现有 task types: analyze_event, gate_check 等 8 种" ← Discovery D2 src/queue/types.ts
- "gate_check payload 为 { pr_number, check_run_id? }，不含 owner/repo" ← Discovery D2 src/queue/types.ts line 34
- "reviewerGroups schema: Record<string, string[]>" ← QA round (4.3 Design)
- "不再需要 CODEOWNERS 解析器" ← QA round (4.5 Alternatives)
- "reviewerGroups 缺失时跳过指派" ← QA round (4.5 Alternatives)
- "异步 Queue 处理" ← QA round (4.1 Context & Scope)
- "触发事件包含 ready_for_review + opened" ← QA round (4.1 Context & Scope)
- "handler 仅做 Reviewer 指派，不处理 Status 迁移" ← QA round (4.1 Context & Scope)
-->
