---
title: "RFC-0020: 取消日报、新增 per-person commit 汇总与 per-repo Project V2 进展报告"
status: draft
author: sii-xinglong
date: 2026-04-25
reviewers: []
---

# RFC-0020: 取消日报、新增 per-person commit 汇总与 per-repo Project V2 进展报告

## 概述

取消现有 per-repo 日报（`daily` report_type），替换为两条独立发送的新报告：(1) org 级 per-person commit 汇总（`person_commit_summary` report_type），合并所有 primatrix org 仓库的 commits、merged PRs、closed issues，按人聚合为一条消息；(2) per-repo 项目进展报告（`repo_project_progress` report_type），基于 Project V2 #14 的 Status 和 Target Date 字段，每个仓库独立生成一条消息。同时将 morning_focus 投递渠道从 DingTalk direct message 改为 group channel。周报（`weekly` report_type）保持不变。

## 背景

**技术现状：** Beaver 现有报告系统通过 cron 触发，遍历 D1 中所有活跃仓库（`repositories` 表的 `full_name` 列标识仓库，如 `primatrix/Beaver`），每个仓库独立生成一条 daily/weekly 报告并发送到 DingTalk 群。报告内容包含 achievements（按类型分类的 PR 成果）、contributor_summary（LLM 从事件推断的贡献者统计）、milestone_progress、pending_items 和 summary。morning_focus 则为每个有活跃工作项的用户生成个性化简报并通过 DingTalk 私信发送。

**问题：** (1) 现有 contributor_summary 由 LLM 从 merged PRs 和 closed issues 推断，缺少 commits 维度，且无法独立于 LLM 验证其准确性。(2) 日报缺少 Project V2 #14 级别的项目进展视图（如 Target Date 预警）。(3) morning_focus 以私信发送，团队无法了解其他成员的工作安排。(4) 多仓库各自发送日报导致消息碎片化，缺少 org 级统一视图。(5) 日报与周报内容高度重叠（均基于 merged PRs / closed issues），信息密度低，收效不抵噪声。

## 设计目标

**Goals:**

- G1: 取消 per-repo 日报（`daily` report_type），减少群消息噪声
- G2: 新增 org 级 per-person commit 汇总报告（`person_commit_summary`），按 author 聚合 commits、merged PRs、closed issues，合并所有 primatrix 仓库为一条消息
- G3: 新增 per-repo 项目进展报告（`repo_project_progress`），基于 Project V2 #14 的 Status 和 Target Date 字段，每个仓库独立生成一条消息
- G4: morning_focus 报告从 DingTalk 私信改为群消息，让全组可见
- G5: LLM 调用复用现有 OpenAI-compatible 配置（`LLM_BASE_URL` / `LLM_MODEL` / `LLM_PROVIDER` 在 `src/env.ts` 中已定义），运维侧配置 mimo v2 pro 的 base_url 和 model 即可，无需代码改动

**Non-Goals:**

- 不修改现有周报（`weekly` report_type）——`handleDailyWeekly` 函数（仅 weekly 路径）、`ReportOutput` Zod schema、`formatWeeklyReport` 格式化函数均保持不变
- 不改变非 primatrix org 仓库的报告行为
- 不引入报告模板自定义系统或历史回看功能
- 不修改 LLM provider 基础设施代码——切换到 mimo v2 pro 仅需运维配置 `LLM_PROVIDER=openai-compatible`、`LLM_BASE_URL` 和 `LLM_MODEL` 环境变量

## 方案

### 系统上下文

```text
cron (工作日)
  → scheduler.ts
     ├─ 09:00 CST: morning_focus（现有，渠道从 direct 改为 group）
     ├─ 18:00 CST: 取消 per-repo daily，新增 ↓
     │   ├─ person_commit_summary（新 report_type，org 级一条消息）
     │   └─ repo_project_progress（新 report_type，每个 primatrix 仓库一条消息）
     └─ 周五 18:00 CST: weekly（不变）

generate-report.ts handler 分支:
  ├─ handleDailyWeekly()        ← 仅 weekly 路径，daily 路径移除
  ├─ handleMorningFocus()       ← 渠道从 direct 改为 group
  ├─ handlePersonCommitSummary() ← 新增
  └─ handleRepoProjectProgress() ← 新增
```

### 1. 数据采集（`src/queue/handlers/report-data.ts` 新增）

**1a. `collectOrgPerPersonData()` — per-person commit 汇总数据：**

- 从 D1 `repositories` 表筛选 `full_name LIKE 'primatrix/%'` 的活跃仓库（`full_name` 列由 `listActiveRepositories` 返回，见 `src/db/repository.ts`）
- 每个仓库调用 `GET /repos/{owner}/{repo}/commits?since={period.start}` 获取 commits 列表（不指定 `sha` 参数，默认返回所有分支的 commits），按 `author.login` 聚合 commit 计数
- 复用现有 GitHub API 调用获取 merged PRs 和 closed issues，按 author 聚合
- 所有 GitHub API 调用需支持分页（`per_page=100`，通过 `Link` header 获取下一页）和重试（指数退避，最多 3 次）
- 输出类型:

  ```typescript
  type OrgPerPersonData = Array<{
    author: string
    commits: number
    mergedPrs: number
    closedIssues: number
    commitMessages: string[]  // 每条 commit 的 message，供 LLM 生成上下文感知的 highlight
  }>
  ```

**1b. `collectRepoProjectProgress()` — per-repo 项目进展数据：**

- 使用 `BEAVER_PROJECT_NODE_ID` 环境变量（已在 `src/env.ts` 中定义）定位 Project #14
- 调用 GraphQL 查询 Project #14 所有 items，读取 `Status` 和 `Target Date` 自定义字段（GraphQL 查询需支持 cursor-based 分页，通过 `pageInfo.hasNextPage` 获取下一页）
- 按仓库分组：Project V2 item 直接关联的 Issue 通常是 project repo 中的 tracker issue，需通过 tracker issue 的关联 PR 所在仓库或 body 中的跨仓库引用来确定实际 repo 归属；若无关联 PR，则 fallback 到 tracker issue 自身所在仓库
- 筛选逻辑采用"变化驱动"策略，仅展示有实质进展的事项，避免信息过载（全面视图由周报覆盖）：
  - 今日/本周关闭事项：Status = Done 且在时间窗口内
  - 新提交 PR：关联 Issue 在 Project #14 上且有新 PR
  - 超过/接近 Target Date 的事项：`Target Date ≤ today + 3 天` 且 `Status ≠ Done`
  - 若事项未设置 Target Date，默认时间范围为开始时间后的一周（即 `Target Date = period.start + 7 天`）
- 输出类型:

  ```typescript
  type RepoProjectProgress = Array<{
    repo: string
    closedItems: Array<{ number: number; title: string }>
    newPrs: Array<{ number: number; title: string; author: string }>
    deadlineAlerts: Array<{ number: number; title: string; targetDate: string; daysRemaining: number }>
  }>
  ```

**1c. 周报数据来源（现有 collectWeeklyData()，不变）：**

周报数据由现有的 per-repo 数据采集函数提供，数据来源如下：

- `merged_prs`: 通过 GitHub REST API `GET /repos/{owner}/{repo}/pulls?state=closed&sort=updated&direction=desc` 获取时间窗口内合并的 PR
- `closed_issues`: 通过 GitHub REST API `GET /repos/{owner}/{repo}/issues?state=closed&sort=updated&direction=desc` 获取时间窗口内关闭的 Issue
- `status_changes`: 通过 D1 `events` 表查询时间窗口内的 `issues` 事件（`labeled` / `unlabeled` action），提取 `status/*` 标签变更
- milestones: 通过 GitHub REST API `GET /repos/{owner}/{repo}/milestones` 获取活跃 Milestone 及其进度

### 2. LLM Prompts（`src/reasoning/prompts/` 新增与修改）

所有 prompt 遵循统一的 `LLMRequest` 接口：`system` 字符串 + `messages` 数组（user message 传入 JSON 格式的原始数据），输出通过 Zod schema 校验。

#### 2a. per-person commit 汇总 prompt（`src/reasoning/prompts/person-commit-summary.ts` 新增）

**System prompt:**

```text
你是一个研发团队活动分析师。请根据以下 org 级仓库的原始数据，生成 per-person commit 汇总报告。

要求：
1. 使用中文输出
2. 按 commits 数量降序排列每位成员
3. 对每位成员，提供一行简短 highlight（如："专注 Feature X 开发，提交 3 个 PR 均已合并"）
4. 如果某成员 commits 多但 merged PRs 少，指出可能存在未合并的工作
5. 如果某成员无 commits 但有 merged PRs，说明其通过 review/merge 贡献
6. 末尾提供一段团队整体总结（2-3 句话）
7. 输出 Markdown 格式
```

**User message 格式:**

```json
{
  "period": { "start": "2026-04-24T18:00:00Z", "end": "2026-04-25T18:00:00Z", "type": "daily" },
  "per_person_data": [
    {
      "author": "alice", "commits": 8, "mergedPrs": 3, "closedIssues": 2,
      "commitMessages": ["feat: add webhook retry handler", "fix: resolve timeout in queue consumer", "..."]
    },
    {
      "author": "bob", "commits": 5, "mergedPrs": 1, "closedIssues": 0,
      "commitMessages": ["refactor: simplify auth flow", "..."]
    }
  ]
}
```

**输出遵循 `PersonCommitSummaryOutput` Zod schema**（见第 3 节）。

#### 2b. per-repo 项目进展 prompt（`src/reasoning/prompts/repo-project-progress.ts` 新增）

**System prompt:**

```text
你是一个项目管理分析师。请根据以下 per-repo 的 Project V2 进展数据，生成项目进展报告。

要求：
1. 使用中文输出
2. 按仓库分组展示，每个仓库一个段落
3. 每个仓库段落包含：
   - 已关闭事项列表（如有）
   - 新提交 PR 列表（如有）
   - Target Date 预警：列出距 Target Date ≤3 天或已逾期的事项，标注剩余天数和 urgency（overdue/critical/warning）
   - 如无上述事项，注明"本周期无进展更新"
4. 末尾提供整体项目健康度总结（关注逾期事项数量和趋势）
5. 输出 Markdown 格式
6. 如果某仓库所有事项均无 Target Date，仅展示已关闭和新 PR，不报错
```

**User message 格式:**

```json
{
  "period": { "start": "2026-04-24T18:00:00Z", "end": "2026-04-25T18:00:00Z", "type": "daily" },
  "repo_progress": [
    {
      "repo": "primatrix/Beaver",
      "closedItems": [{ "number": 42, "title": "Fix webhook retry logic" }],
      "newPrs": [{ "number": 101, "title": "Add org summary handler", "author": "alice" }],
      "deadlineAlerts": [{ "number": 38, "title": "Implement Project V2 sync", "targetDate": "2026-04-27", "daysRemaining": 2 }]
    }
  ]
}
```

**输出遵循 `RepoProjectProgressOutput` Zod schema**（见第 3 节）。

#### 2c. morning_focus prompt（`src/reasoning/prompts/report.ts` 现有，调整输出渠道）

**System prompt:**

```text
你是一个个人研发助手。请根据以下用户的工作项数据，生成今日晨间 Focus 简报。

要求：
1. 使用中文输出
2. 按优先级排序（p0 > p1 > p2 > p3），同优先级内按 DDL 临近度排序
3. 标注每个任务的 DDL 状态：已逾期（🔴）、<48h（🟡）、正常（🟢）、无 DDL（⚪）
4. 高亮"今日最值得关注"的前 3 个核心任务，简述选择理由
5. 如果存在阻塞项（status/blocked 或被其他任务依赖），置顶提醒
6. 输出 Markdown 格式
```

**User message 格式:**

```json
{
  "user": "alice",
  "date": "2026-04-25",
  "assigned_tasks": [
    {
      "number": 42,
      "repo": "primatrix/Beaver",
      "title": "Fix webhook retry logic",
      "status": "in-progress",
      "priority": "p1",
      "targetDate": "2026-04-26",
      "labels": ["type/bug", "priority/p1"],
      "linkedPrs": [{ "number": 100, "status": "open" }]
    }
  ]
}
```

**输出遵循 `MorningFocusOutput` Zod schema**（见第 3 节）。

#### 2d. weekly report prompt（`src/reasoning/prompts/report.ts` 现有，不变）

**System prompt:**

```text
你是一个研发团队项目管理分析师。请根据以下仓库的本周数据，生成周报。

要求：
1. 使用中文输出
2. 包含以下段落：
   - 里程碑达成度：分析本周工作对整体 Milestone 的贡献百分比及进度预测
   - 研发效能分析：吞吐量（完成的 size/L 与 size/S 任务分布）、响应时长（PR cycle time）
   - 缺陷与质量趋势：按优先级统计 type/bug 堆积情况
   - 深度风险诊断：识别停留 >3 天的任务、资源瓶颈（某成员承载过多 size/L）
   - 下周展望：根据 status/ready-to-develop 储备建议下周计划
3. 提供数据支撑的分析，避免空泛描述
4. 输出 Markdown 格式
```

**User message 格式:**

```json
{
  "period": { "start": "2026-04-18T18:00:00Z", "end": "2026-04-25T18:00:00Z", "type": "weekly" },
  "repo": "primatrix/Beaver",
  "merged_prs": [...],
  "closed_issues": [...],
  "status_changes": [...],
  "milestones": [...]
}
```

**输出遵循 `ReportOutput` Zod schema**（现有，不变）。

### 3. Zod schemas（`src/reasoning/schemas/report.ts` 新增）

```typescript
// 新增: per-person commit 汇总输出
const PersonCommitSummaryOutput = z.object({
  per_person: z.array(z.object({
    author: z.string(),
    commits: z.number(),
    merged_prs: z.number(),
    closed_issues: z.number(),
    highlight: z.string(),
  })),
  team_summary: z.string(),
})

// 新增: per-repo 项目进展输出
const RepoProjectProgressOutput = z.object({
  repos: z.array(z.object({
    repo: z.string(),
    closed_items: z.array(z.object({ number: z.number(), title: z.string() })),
    new_prs: z.array(z.object({ number: z.number(), title: z.string() })),
    deadline_alerts: z.array(z.object({
      number: z.number(),
      title: z.string(),
      target_date: z.string(),
      days_remaining: z.number(),
      urgency: z.enum(["overdue", "critical", "warning"]),
    })),
    repo_summary: z.string(),
  })),
  overall_health: z.string(),
})

// 新增: morning focus 输出（若现有 schema 不含结构化字段）
const MorningFocusOutput = z.object({
  top_tasks: z.array(z.object({
    number: z.number(),
    repo: z.string(),
    title: z.string(),
    priority: z.string(),
    deadline_status: z.enum(["overdue", "critical", "normal", "none"]),
    reason: z.string(),
  })),
  blockers: z.array(z.object({ number: z.number(), title: z.string() })),
  summary: z.string(),
})

// 现有: weekly report 输出（不变）
// ReportOutput = z.object({ achievements, contributor_summary, milestone_progress, pending_items, summary })
```

### 4. Formatters（`src/acting/notify/report-formatter.ts` 新增）

```typescript
// 新增: formatPersonCommitSummary()
// 输出 DingTalk Markdown，包含 per-person 列表和团队总结

// 新增: formatRepoProjectProgress()
// 输出 DingTalk Markdown，按仓库分组展示已关闭、新 PR、DDL 预警

// 现有: formatWeeklyReport() 不变
// 现有: formatMorningFocus() 不变（或微调以适配新 schema）
```

**消息长度限制与拆分策略：**

- 单条 DingTalk 消息长度限制为 **500 字**（中文字符计算）
- 若格式化后的消息超过 500 字，按以下策略拆分：
  - `person_commit_summary`: 按 author 分组拆分，每个 author 的 highlight 作为独立消息发送
  - `repo_project_progress`: 按 repo 分组拆分，每个 repo 的进展作为独立消息发送
  - `morning_focus`: 按 top_tasks 列表拆分，每条消息最多包含 3 个任务
- 拆分后的消息在标题中标注序号（如 `[1/3]`、`[2/3]`、`[3/3]`）

### 5. Handler 与调度变更

**`src/queue/handlers/generate-report.ts`:**

- `handleDailyWeekly()`: 移除 `daily` 路径，仅保留 `weekly` 路径
- `handleMorningFocus()`: `channel: "direct"` 改为 `channel: "group"`
- 新增 `handlePersonCommitSummary()`: 调用 `collectOrgPerPersonData()` → LLM → `formatPersonCommitSummary()` → group 消息
- 新增 `handleRepoProjectProgress()`: 调用 `collectRepoProjectProgress()` → LLM → `formatRepoProjectProgress()` → group 消息

**`src/sensing/scheduler.ts`:**

- 移除: 18:00 CST 的 per-repo `generate_report(daily)` 任务
- 新增: 18:00 CST 的 `generate_report(person_commit_summary)` 任务（一条，org 级）
- 新增: 18:00 CST 的 `generate_report(repo_project_progress)` 任务（每个 primatrix 仓库一条）
- 保持: 周五 18:00 CST 的 per-repo `generate_report(weekly)` 任务
- 保持: 09:00 CST 的 `generate_report(morning_focus)` 任务

**`src/queue/types.ts`:**

- `report_type` 枚举: 移除 `"daily"`，新增 `"person_commit_summary"` 和 `"repo_project_progress"`

**`src/acting/notify/router.ts`:**

- `ROUTE_TABLE.morning_focus.channels` 从 `["direct"]` 改为 `["group"]`
- 新增 `ROUTE_TABLE.person_commit_summary.channels = ["group"]`
- 新增 `ROUTE_TABLE.repo_project_progress.channels = ["group"]`

### 6. morning_focus 预期流程

```text
cron: 0 1 * * 1-5 (09:00 CST, 工作日)
  ↓
scheduler.ts: 遍历 D1 中所有活跃仓库，对每个仓库发送
  TaskMessage { type: "generate_report", payload: { report_type: "morning_focus" } }
  ↓
generate-report.ts::handleMorningFocus():
  ├─ 1. 查询 D1: 获取所有有活跃工作项的用户列表
  │     （当前 iteration 内有 assign 给该用户的 Issue/PR 的 assignee）
  ├─ 2. 对每个用户:
  │     ├─ collectMorningFocusData(userId)
  │     │   ├─ 筛选条件：所有 assign 给该用户且属于当前 iteration 的 Issue/PR
  │     │   ├─ 读取 status 标签（in-progress / ready-to-develop 等）
  │     │   ├─ 读取 priority 标签（p0/p1/p2/p3）
  │     │   ├─ 读取 Target Date 字段
  │     │   └─ 读取关联的 PR 状态
  │     ├─ buildMorningFocusRequest(data)  ← 2c prompt
  │     ├─ callAndValidate()               ← LLM 调用 + Zod 校验 + retry
  │     └─ formatMorningFocus(output)      ← DingTalk Markdown
  ├─ 3. 合并所有用户的简报为一条消息（或分别发送）
  └─ 4. 发送到 DingTalk 群（channel: "group"，从 "direct" 改为 group）
```

**渠道变更说明:** 将 morning_focus 从私信改为群消息，目的是让全组可见每个人的工作安排，促进团队对齐。若涉及敏感信息（如个人绩效相关），可由 LLM prompt 控制不输出具体内容，或由用户自行配置是否参与群内推送。

### 7. 备选方案

| 替代方案 | 否决理由 |
|---------|---------|
| 保留日报，额外新增两条报告 | 日报与周报内容重叠，保留只会增加噪声；取消日报后群消息数从 N+1（N 个仓库日报 + 1 条 morning_focus）降为 N+2（per-person + per-repo progress），但 per-person 是 org 级一条，实际为 1 + M + 1（M 为 primatrix 仓库数）|
| 将 per-person 和 per-repo 合并为一条消息 | 两条消息职责不同（人员活跃度 vs 项目进展），分开发送便于不同角色关注不同消息；合并会导致单条消息过长 |
| 程序直接格式化报告而不用 LLM | LLM 能提供智能分析（趋势洞察、highlight），且与现有报告流程保持架构一致性 |
| morning_focus 保持 DingTalk direct message | morning_focus 信息对全组有价值（让团队了解每个人的工作安排），且减少对 DingTalk App 私信接口的依赖 |

## 影响范围

- **`src/queue/handlers/report-data.ts`**: 新增 `collectOrgPerPersonData()` 和 `collectRepoProjectProgress()` 函数
- **`src/queue/handlers/generate-report.ts`**: 新增 `handlePersonCommitSummary()` 和 `handleRepoProjectProgress()` 分支；`handleDailyWeekly()` 移除 daily 路径；`handleMorningFocus()` 渠道改为 group
- **`src/reasoning/prompts/person-commit-summary.ts`** (新文件): per-person commit 汇总 LLM prompt
- **`src/reasoning/prompts/repo-project-progress.ts`** (新文件): per-repo 项目进展 LLM prompt
- **`src/reasoning/schemas/report.ts`**: 新增 `PersonCommitSummaryOutput`、`RepoProjectProgressOutput`、`MorningFocusOutput` Zod schema
- **`src/acting/notify/report-formatter.ts`**: 新增 `formatPersonCommitSummary()` 和 `formatRepoProjectProgress()` 格式化函数
- **`src/acting/notify/router.ts`**: morning_focus 路由从 direct 改为 group；新增 person_commit_summary 和 repo_project_progress 路由
- **`src/sensing/scheduler.ts`**: 移除 daily 调度，新增 person_commit_summary 和 repo_project_progress 调度
- **`src/queue/types.ts`**: `report_type` 枚举移除 `"daily"`，新增 `"person_commit_summary"` 和 `"repo_project_progress"`
- **DingTalk 群聊**: 取消 per-repo 日报；新增 per-person commit 汇总消息和 per-repo 项目进展消息；morning_focus 从私信改为群消息
- **现有周报**: 不受影响（`handleDailyWeekly` weekly 路径、`ReportOutput`、`formatWeeklyReport` 均不变）
- **非 primatrix org 仓库**: 不受影响

## 实施计划

| # | SubTask | 依赖 | 预期交付物 |
|---|---------|------|-----------|
| S1 | 新增 org 级 per-person 数据采集函数 | 无 | `collectOrgPerPersonData()` in `src/queue/handlers/report-data.ts` + 单元测试 |
| S2 | 新增 per-repo Project V2 进展数据采集函数 | 无 | `collectRepoProjectProgress()` in `src/queue/handlers/report-data.ts` + 单元测试 |
| S3 | 新增 per-person commit 汇总 prompt + Zod schema | S1 | `src/reasoning/prompts/person-commit-summary.ts` + `PersonCommitSummaryOutput` in `src/reasoning/schemas/report.ts` + 测试 |
| S4 | 新增 per-repo 项目进展 prompt + Zod schema | S2 | `src/reasoning/prompts/repo-project-progress.ts` + `RepoProjectProgressOutput` in `src/reasoning/schemas/report.ts` + 测试 |
| S5 | 新增 per-person commit 汇总 formatter | S3 | `formatPersonCommitSummary()` in `src/acting/notify/report-formatter.ts` + 快照测试 |
| S6 | 新增 per-repo 项目进展 formatter | S4 | `formatRepoProjectProgress()` in `src/acting/notify/report-formatter.ts` + 快照测试 |
| S7 | 串联 handlePersonCommitSummary handler + 调度 | S1, S3, S5 | `handlePersonCommitSummary()` in `src/queue/handlers/generate-report.ts` + scheduler 变更 + 集成测试 |
| S8 | 串联 handleRepoProjectProgress handler + 调度 | S2, S4, S6 | `handleRepoProjectProgress()` in `src/queue/handlers/generate-report.ts` + scheduler 变更 + 集成测试 |
| S9 | 移除 daily 路径 + 更新 report_type 枚举 | S7, S8 | `handleDailyWeekly()` 移除 daily 分支；`types.ts` 更新枚举；router 新增路由 |
| S10 | morning_focus 渠道从 direct 改为 group | 无 | `src/acting/notify/router.ts` + `src/queue/handlers/generate-report.ts` + 测试更新 |

- S1 和 S2 可并行开发
- S3 和 S4 可并行开发（分别依赖 S1、S2）
- S5 和 S6 可并行开发（分别依赖 S3、S4）
- S7 和 S8 可并行开发（分别依赖 S1+S3+S5、S2+S4+S6）
- S9 依赖 S7、S8
- S10 独立于 S1-S9，可在任意时间实施

## 风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| GitHub API 速率限制：遍历多仓库 commits + GraphQL 查询可能触发限流 | 报告生成失败或延迟 | 使用现有 GitHubClient 的 App token（5000 req/hr）；必要时添加 pagination 和 retry |
| LLM 输出不稳定：数据量大时 JSON 输出可能不符合 schema | 报告生成失败 | 复用现有 `callAndValidate` 的 retry 机制（失败后重试一次） |
| Target Date 字段缺失：部分 Issue 可能未设置 Target Date | 预警段落数据不完整 | 仅展示有 Target Date 的事项，缺失时跳过，不报错 |
| DingTalk 消息长度限制：per-person 汇总合并多仓库数据后消息可能超长 | 消息发送失败或被截断 | LLM prompt 中限制输出长度；必要时按仓库拆分 |
| 取消日报后团队不适应：部分成员依赖日报了解每日进展 | 信息断层 | per-person commit 汇总 + per-repo 项目进展覆盖日报原有信息；周报保留深度分析 |

<!-- provenance
- "现有 contributor_summary 由 LLM 从 merged PRs 和 closed issues 推断" ← Discovery D2: src/reasoning/prompts/report.ts 行 3-4 (LLM prompt 指令)
- "morning_focus 通过 DingTalk 私信发送" ← Discovery D2: src/acting/notify/router.ts 行 9-11 (ROUTE_TABLE.morning_focus.channels = ["direct"])
- "reports 发送到 DingTalk 群" ← Discovery D2: src/acting/notify/router.ts 行 13-15 (ROUTE_TABLE.report.channels = ["group"])
- "repositories 表的 full_name 列" ← Discovery D2: src/db/repository.ts (listActiveRepositories 返回 full_name); src/queue/handlers/report-data.ts 行 167 (const [owner, repo] = fullName.split("/"))
- "BEAVER_PROJECT_NODE_ID 环境变量" ← Discovery D2: src/env.ts 行 34 (BEAVER_PROJECT_NODE_ID?: string)
- "commits 数据来源 = GitHub REST API" ← QA round 4.1-1
- "Target Date = Project V2 #14 上的自定义 Date 字段" ← QA round 4.1-2
- "新增段落仅 primatrix org 仓库" ← QA round 4.1-3
- "morning_focus 复用现有 DINGTALK_ROBOT_WEBHOOK" ← QA round 4.1-4
- "合并所有 primatrix 仓库为一条消息" ← QA round 4.2-1
- "嵌入现有 cron 时间" ← QA round 4.2-2
- "现有 per-repo 报告继续发送" ← QA (reviewer BLOCK fix round 1: 用户确认并行发送两条报告)
- "成功指标 = 功能正确性" ← QA round 4.2-4
- "在 generate-report.ts 中新增 handler 分支" ← QA round 4.3-1
- "LLM 生成报告" ← QA round 4.3-2
- "单元 + 集成测试" ← QA round 4.3-3
- "LLM_PROVIDER/LLM_BASE_URL/LLM_MODEL 已支持配置化" ← Discovery D3: README.md 行 166-169; src/env.ts 行 23-26
- "现有 callAndValidate retry 机制" ← Discovery D2: src/queue/handlers/generate-report.ts 行 21-50
- "sync-project.ts 仅做 label↔status 不一致检测" ← Discovery D2: src/queue/handlers/sync-project.ts 行 161-201 (extractMismatches)
- "handleDailyWeekly / ReportOutput / formatDailyReport / formatWeeklyReport 保持不变" ← QA (reviewer BLOCK fix round 1: non-goal 显式声明; Discovery D2 确认这些函数存在)
- "取消日报改为 per-person commit 汇总 + per-repo 项目进展" ← 用户指示 v2 修改
- "四个 prompt 详细描述" ← 用户指示 v2 修改
- "morning_focus 预期流程" ← 用户指示 v2 修改
-->
