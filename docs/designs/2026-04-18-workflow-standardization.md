---
issue: primatrix/Beaver#115
title: 工作流程标准化:5 月底完成标准 skills、GitHub 自动化与 onboarding 文档
date: 2026-04-18
status: design-pending
---

# 工作流程标准化:5 月底完成标准 skills、GitHub 自动化与 onboarding 文档 Design Document

## 1. Context & Scope

Beaver 团队当前同时维护三套与"工作流标准化"相关的资产:

**a) Skills 库 — `primatrix/skills`**

Claude Code plugin marketplace,目前发布 8 个 plugin:`beaver`、`exec-remote`、`gke-tpu`、`lint-fix`、`session-recorder`、`superpowers`、`tpu-perf-model`、`xprof-profiling-analysis`。其中 `beaver` 套件已实现 7 个 skill:`beaver-issue`、`beaver-pr`、`beaver-design-doc`、`beaver-audit`、`beaver-focus`、`beaver-report`、`beaver-engine`,覆盖"开发者认领 issue → 写设计文档 → 提交 PR → 项目状态"主链路。**多人 Doc Review 环节没有对应 skill 或自动化。**

**b) GitHub 自动化 — `primatrix/Beaver`(Cloudflare Worker)**

已上线能力:

- Webhook 处理:9 类 GitHub 事件(issues / PRs / reviews / check suites / comments / project items / milestones / installation / installation_repositories)
- FSM:`triage → ready → in-progress → review → done`,fast-path 同步驱动状态变化
- Gatekeeper:LOC 限制、必需 label、issue 关联校验、design-approval guard
- Cron:daily/weekly report、stale detection、project sync、repo discovery
- 用户身份:已有 `user_mapping`(GitHub ↔ DingTalk),**但无分组信息**
- Sub-issue 处理:仅 `audit-split.ts` 解析 issue body markdown checklist 用于 LLM 拆分质量审核;**不监听 GitHub 原生 `sub_issues` webhook,无父子状态联动**
- Beaver GitHub App 已安装到 org 下全部仓库(包括 `primatrix/wiki`)

**c) Onboarding 文档 — `primatrix/wiki`**

已有 `docs/onboarding/` 目录,内含 7 篇领域型文档:`cloudflare-setup.md`、`dev-environment.md`、`index.md`、`pallas-kernel-guide.md`、`performance-overview.md`、`performance-tpu.md`、`profiling-hbm-guide.md`。**没有面向"端到端工作流"的入门文档。**

**本次交付的目标工作流:**

```text
TL 设目标 → 沟通对齐会 → 拆 Task 分配
     ↓
开发者认领 Issue → 写 Design Doc → 多人 Doc Review
     ↓
拆 Sub-Task → 开发 + 测试 → 每个 Sub-Task 关联 PR → Code Review
```

交付截止:**2026-05-31**(issue#115 原文要求)。

## 2. Design Goals

### 2.1 Goals

**G1 — Beaver 自动指派 PR Reviewer(全仓库)**
Beaver Worker 监听任意仓库 `pull_request.opened` 事件,基于 CODEOWNERS + 团队分组 + 工作量数据,自动选取 reviewer 并通过 GitHub API 指派。覆盖代码 PR 与 design doc PR。

**G2 — Doc Review 状态联动**
`primatrix/wiki` 仓库的 design-doc PR merge 时,Beaver Worker 解析 PR body 中的 `Closes primatrix/<repo>#<N>` 关键字,把对应 Beaver issue 从 `status/design-pending` 自动转 `status/ready-to-develop`。

**G3 — Sub-Task ↔ 父 Task 联动**
Beaver Worker 监听 GitHub 原生 `sub_issues` webhook;当某父 issue 的全部 sub-issues 都已 closed 且父 issue 当前处于 `in-progress` 或 `review` 状态时,自动转 `status/done`。

**G4 — 用户身份与工作量数据模型**
在 Beaver D1 内建立:

- `user_group`:GitHub login → 团队分组(1 / 2)+ active 状态
- `reviewer_workload`:GitHub login → 当前 in-flight 工作量(authored open PRs + pending reviews)

维护机制:webhook 增量 + 每日 cron 全量重算。

**G5 — Onboarding 工作流文档**
在 `primatrix/wiki/docs/onboarding/` 新增 6 篇工作流文档(总入口 + 5 篇环节文档),覆盖从 Issue 认领到 PR merge 的完整路径。新成员仅依赖此文档即可独立完成首个任务。

### 2.2 Non-Goals

- **NG1** 不做 TL 设目标 / 沟通对齐会 / Task 拆分 / Sub-Task 拆分 skill —— 这些环节本次保持人工
- **NG2** 不做 onboarding FAQ / 故障排查文档
- **NG3** 不做评审者选择算法的复杂化升级 —— 仅按 in-review 数量最小挑人,出现并列按 GitHub 用户名字典序;不引入历史评审次数、技能匹配模型、轮换策略(技能维度通过复用 CODEOWNERS 兜底实现,无需新模型)
- **NG4** 不提供本地 skill 介入 G1 reviewer 指派 —— G1 完全由 Worker 驱动;手动覆盖通过 PR 评论 / 直接修改 PR 的 reviewer 字段实现,不做 skill
- **NG5** 不做跨仓库的 sub-issue 联动 —— G3 仅处理同仓库内的父子 issue
- **NG6** 不重写已有 7 个 `beaver-*` skill —— 只在 onboarding 文档中串联介绍
- **NG7** 不做用户身份 / 分组的自助绑定 UI —— `user_group` 通过现有 admin REST API 维护
- **NG8** 不做 onboarding 文档英文版 —— 本次仅出中文

### 2.3 Success Metrics

| #     | 指标                  | 度量方式                                                                                                       |
| ----- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| **SM1** | G1 自动指派覆盖率     | 任意仓库新开 PR,Beaver 在 5 分钟内完成 reviewer 指派的比例 ≥ 95%                                              |
| **SM2** | G2 状态转换正确率     | wiki design PR merge 后,对应 Beaver issue 在 5 分钟内完成 `design-pending → ready-to-develop` 的比例 = 100%(0 漏、0 误转) |
| **SM3** | G3 父任务 done 触发正确率 | 父 issue 的所有原生 sub-issue 全部 close 后,父 issue 在 5 分钟内转 `status/done` 的比例 = 100%                |
| **SM4** | G4 数据完整性         | issue#115 完成时,所有当前活跃团队成员均已录入 `user_group`;`reviewer_workload` 与 GitHub 实际值抽样 10 次差异 = 0 |
| **SM5** | G5 onboarding 有效性  | 至少 1 名新成员仅依赖 `docs/onboarding/` 文档,1 个工作日内独立完成首个 Issue → PR → Done 全流程               |
| **SM6** | 整体交付时间          | 所有产物在 2026-05-31 前合入各自仓库 main 分支                                                                 |

## 3. The Design

### 3.1 System Context Diagram

```text
┌──────────────────── Cloudflare Worker (现有,扩展) ────────────────────┐
│                                                                       │
│  fetch()  ←  PR webhook  ─────────────┬──→ G1 reviewer 指派           │
│              (所有仓库)                ├──→ G2 wiki PR merged 状态转换 │
│                                       └──→ T4 reviewer_workload 更新  │
│                                                                       │
│           ←  sub_issues webhook  ─────→ G3 父任务 done 联动           │
│                                                                       │
│  scheduled() ← daily cron ────────────→ T5 reviewer_workload 全量重算 │
│                                                                       │
│  D1 数据库:                                                          │
│   现有: user_mapping, issues, prs, ...                                │
│   新增: user_group, reviewer_workload                                 │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ Beaver GitHub App(已装到全部仓库)
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
         primatrix/wiki                  primatrix/<其他仓库>
         ├ docs/onboarding/ 新增 6 篇    └ G1 / G2 / G3 状态联动目标
         └ G2 状态转换的 PR 来源
```

### 3.2 Core Architecture

本次设计**不引入新服务**,完全在现有 Beaver Worker 内扩展:

| 组件                                  | 类型     | 职责                                                                                       |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| **`acting/reviewer/assign.ts`** (新增) | 业务模块 | G1 核心逻辑:CODEOWNERS 解析 → 候选人筛选 → 工作量比较 → 调 GitHub `requestReviewers` API |
| **`acting/state/doc-merged.ts`** (新增) | 业务模块 | G2:解析 `Closes` 关键字 → 校验状态 → 转 label                                            |
| **`acting/state/sub-issue-done.ts`** (新增) | 业务模块 | G3:查父 issue 全部 sub-issues → 全 closed 时转父 done                                     |
| **`db/reviewer-workload.ts`** (新增)  | 数据层   | `reviewer_workload` 表的增量 / 全量更新                                                   |
| **`db/user-group.ts`** (新增)         | 数据层   | `user_group` 表 CRUD                                                                       |
| **`router.ts` / `fast-path.ts`** (扩展) | 现有     | 新增 webhook 路由:`pull_request`、`sub_issues`                                            |
| **`scheduled()`** (扩展)              | 现有     | 新增 cron 任务:`reviewer_workload` 全量重算                                              |
| **`admin/` API** (扩展)               | 现有     | 新增端点:`POST /admin/user-group`、`GET/DELETE /admin/user-group/:login`                  |

**技术选型与理由:**

- **不引入新依赖**:CODEOWNERS 解析直接手写正则(GitHub 官方语法简单);D1 / Workers 队列 / GitHub Client 全部复用现有抽象
- **沿用现有 LLM 不参与本次自动化**:G1/G2/G3 全部为确定性规则,无需 LLM,降低成本与不可预测性

### 3.3 Interfaces & Data Flow

**新增 D1 表:**

```sql
-- migrations/NNN_user_group.sql
CREATE TABLE user_group (
  github_login TEXT PRIMARY KEY,
  group_id    INTEGER NOT NULL CHECK (group_id IN (1, 2)),
  active      INTEGER NOT NULL DEFAULT 1,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- migrations/NNN_reviewer_workload.sql
CREATE TABLE reviewer_workload (
  github_login       TEXT PRIMARY KEY,
  authored_open_prs  INTEGER NOT NULL DEFAULT 0,
  pending_reviews    INTEGER NOT NULL DEFAULT 0,
  workload           INTEGER GENERATED ALWAYS AS
                       (authored_open_prs + pending_reviews) STORED,
  last_synced_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**G1 数据流(reviewer 指派):**

```text
pull_request.opened webhook
   ↓
1. 加载 PR 元数据 + repo 的 .github/CODEOWNERS
   ↓
2. 计算"已有 reviewer 集合" = CODEOWNERS 命中的人 ∪ PR 作者已 @ 的人
   ↓
3. 若 |已有 reviewer| ≥ 2 → 指派全部命中的 owner,完成
   ↓
4. 否则,补差额:
   a. 按 user_group 分组,排除 PR 作者、排除已选人、排除 active=false
   b. 对每组,按 reviewer_workload.workload ASC, github_login ASC 取第 1 人
   c. 优先从"未覆盖的组"补人,直到凑齐 ≥ 2 人
   ↓
5. 调 PATCH /repos/{owner}/{repo}/pulls/{n}/requested_reviewers
   ↓
6. 在 PR 上评论说明:"由 Beaver 自动指派 / 来源 (CODEOWNERS / 工作量最小)"
```

**G2 数据流(wiki PR merged → issue 状态转换):**

```text
pull_request.closed webhook (merged=true) on primatrix/wiki
   ↓
1. 解析 PR body,匹配 /(?:Close[sd]?|Fix(?:e[sd])?|Resolve[sd]?)\s+([\w-]+)\/([\w-]+)#(\d+)/gi
   ↓
2. 对每个匹配的 issue:
   a. 调 GET /repos/{o}/{r}/issues/{n} 拿当前 labels
   b. 若包含 status/design-pending → 移除该 label,加 status/ready-to-develop
   c. 若不包含 → 静默跳过(已被人工改过状态)
```

**G3 数据流(sub-issue 全 closed → 父 done):**

```text
issues webhook (action=closed):
   - 调 GET /repos/{o}/{r}/issues/{n}/parent 拿父 issue(若无父则跳过)
sub_issues webhook (action=parent_issue_added / sub_issue_added):
   - 直接拿 payload 中的父 issue
   ↓
1. 拿到父 issue ID
   ↓
2. 调 GET /repos/{o}/{r}/issues/{parent}/sub_issues 列全部子任务
   ↓
3. 全部 state=closed?
   NO  → 跳过
   YES → 检查父 issue 当前 label
          - 含 in-progress 或 review → 移除该 label,加 status/done
          - 其他状态 → 静默跳过(避免误把 triage / ready 直接 done)
```

**`reviewer_workload` 维护(T4 + T5):**

| Webhook 事件                                                   | 表更新                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `pull_request.opened`                                          | author.authored_open_prs += 1                                |
| `pull_request.closed`                                          | author.authored_open_prs -= 1;且该 PR 所有 requested_reviewers.pending_reviews -= 1(避免未评审即关闭导致评审者工作量虚高) |
| `pull_request.review_requested`                                | reviewer.pending_reviews += 1                                |
| `pull_request.review_request_removed`                          | reviewer.pending_reviews -= 1                                |
| `pull_request_review.submitted`(approved/changes_requested) | reviewer.pending_reviews -= 1                                |
| 每日 cron                                                      | 全表 truncate,调 GitHub Search API 重算所有人,校正 webhook 漏报 |

### 3.4 Trade-offs

| #       | 决策                       | 选择                                          | 放弃                       | 理由                                                                                          |
| ------- | -------------------------- | --------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| **TO1** | G1 执行位置                | Cloudflare Worker 自动                        | 本地 skill                 | 强制流程不应依赖人记得跑命令;与 G2/G3 同属 webhook 驱动模式,内聚性高                        |
| **TO2** | reviewer 选择策略          | in-review 数最小 + CODEOWNERS 兜底            | 技能匹配 / 历史轮换        | 复杂度爆炸;CODEOWNERS 已隐式表达"目录归属",直接复用即可解决专业度问题                       |
| **TO3** | 工作量数据来源             | D1 缓存 + webhook 增量 + 每日全量重算         | 实时 GitHub Search API     | Search API 30 req/min rate limit 易撞顶;缓存方案延迟 < 1s                                    |
| **TO4** | sub-issue 联动事件源       | 原生 `sub_issues` webhook                     | 解析 issue body checklist  | 与 GitHub 原生数据一致;无正则脆弱性                                                          |
| **TO5** | T2 PR ↔ issue 关联识别 | `Closes <owner>/<repo>#<N>` 关键字            | PR 文件路径反推            | 与 `beaver-design-doc` skill 现有 PR 模板一致                                                |

### 3.5 Test Strategy

**单元测试**(vitest,沿用现有基础设施):

- CODEOWNERS 解析的边界用例(团队 owner、glob 模式、多行规则)
- reviewer 选择算法(0/1/2 owner、作者已@、单组无人、全组无人等边界)
- `reviewer_workload` 增量更新逻辑
- G2 `Closes` 关键字解析(单/多 issue、跨仓库、大小写)
- G3 父 issue 全 closed 判定 + 状态转换约束

**集成测试**(模拟 webhook payload):

- PR opened 全流程:webhook → CODEOWNERS → workload → assign API
- wiki PR merged 全流程:webhook → Closes 解析 → label 转换
- sub-issue closed 全流程:webhook → 父查询 → 父 done

**不做:**

- E2E(真实 GitHub 仓库)—— 通过 admin API 手动触发即可
- reviewer_workload 并发一致性测试 —— webhook 单线程串行,无竞争

**Mock 策略**:GitHub API 全部 mock(沿用 `GitHubClient`);D1 用 vitest 内置 SQLite mock。

### 3.6 Deployment & Dependencies

**部署:**

- 沿用现有 wrangler 部署,无新基础设施
- 新增 2 个 D1 migration:`user_group`、`reviewer_workload`
- 首次上线后:用 admin API 录入团队成员 group 映射;触发一次全量 `reviewer_workload` 重算

**新增依赖:** 无 npm 包(CODEOWNERS 解析手写正则)

**GitHub App 配置变更:** 需在 GitHub App webhook 订阅中加入 `sub_issues` 事件(若未启用)

**Onboarding 文档产出物**(`primatrix/wiki/docs/onboarding/`):

| 文件                              | 内容                                                            |
| --------------------------------- | --------------------------------------------------------------- |
| `workflow-overview.md`            | 总入口:从 TL 设目标到 PR merge 的完整流程图 + 各环节链接       |
| `workflow-issue-claim.md`         | 认领 / 创建 Beaver issue(`/beaver:beaver-issue` 用法)          |
| `workflow-design-doc.md`          | 写 design doc + 多人 review 流程(`/beaver:beaver-design-doc` 用法) |
| `workflow-development.md`         | sub-issue 拆分约定 + 开发 + 测试                                |
| `workflow-pr-review.md`           | PR 提交 + 自动 reviewer 指派机制 + Code Review 流程             |
| `workflow-skills-reference.md`    | 7 个 `beaver-*` skill 速查                                      |

## 4. Alternatives Considered

### Alternative 1:G1 由本地 skill 执行

**做法:** 提供 `/beaver-doc-review` 等本地 skill,开发者手动跑;skill 调 GitHub API 选人并指派。

**取舍分析:**

- 优:实现简单,只需一个 skill 文件
- 优:开发者可在不同场景灵活控制
- 劣:**强制流程依赖人记忆**,漏跑即漏触发,SM1 ≥ 95% 覆盖率难以达成
- 劣:与 G2/G3 的"webhook 自动驱动"模式不一致,内聚性差
- 劣:G4 的 `reviewer_workload` 缓存本来就在 Worker 里,skill 反而要绕一圈拉数据

**拒绝理由:** 团队对该工作流的强制性要求与"依赖人记忆"的本质矛盾;Worker 实现的边际成本低,因为 G2/G3/G4 已经迫使 Worker 持有相关基础设施。

### Alternative 2:G1 仅覆盖 design-doc PR(方案 R)

**做法:** Beaver 仅对 `primatrix/wiki` 的 design-doc PR 自动指派 reviewer;代码 PR 仍由 CODEOWNERS / 手动 @ 处理。

**取舍分析:**

- 优:范围小,与现有代码评审约定零冲突
- 优:风险面小,代码 PR 评审仍由人主导
- 劣:代码评审仍可能因人脉而跳过工作量均衡
- 劣:团队希望"统一所有 PR 评审流程"的目标无法达成

**拒绝理由:** 与团队对齐后,明确希望 Beaver 自动指派覆盖代码 + 设计两类 PR;CODEOWNERS 兜底机制(见 3.3 G1 数据流)已能保证代码 PR 的专业度,扩展到全 PR 没有额外的"专业度漏配"风险。

### Alternative 3:reviewer 选择引入完整技能匹配模型

**做法:** 为每个 reviewer 打技能标签(TPU / 前端 / infra / ...),为每个 PR 做技能分类(LLM 或路径规则),按技能匹配 + 工作量加权选人。

**取舍分析:**

- 优:理论上选人最精准
- 劣:**技能标签体系本身就是 size/L 工程**:谁打、怎么维护、如何与 PR 自动匹配
- 劣:LLM 分类引入不可预测性 + 成本
- 劣:与 NG3「不做评审者算法升级」直接冲突

**拒绝理由:** CODEOWNERS 已隐式表达"目录 → 责任人"的领域映射,直接复用即可解决"前端的人不该 review TPU 内核"的具体痛点;无需新建技能模型。NG3 因此保留,设计复杂度控制在合理范围内。

### Alternative 4:工作量数据走 GitHub Search API 实时查询

**做法:** 每次 G1 触发时,实时调 `GET /search/issues?q=is:open+is:pr+review-requested:USER` 等 API 查询每个候选人的工作量,不建本地缓存。

**取舍分析:**

- 优:无需新表,无一致性维护成本
- 优:数据 100% 准确,无 webhook 漏报风险
- 劣:候选人 N 个 → N 次 API 调用,延迟随团队规模线性增长
- 劣:**Search API 30 req/min 独立 rate limit**,在多 PR 并发时极易撞顶
- 劣:每次 G1 触发都要等待 N 个 round-trip,SM1 的"5 分钟"窗口仍可能因外部 API 抖动而违规

**拒绝理由:** 缓存方案仅多 2 张 D1 表,延迟从 N × API roundtrip 降到一次本地查询(< 100ms);webhook 漏报风险通过每日 cron 全量重算补偿,工程上可接受。

### Alternative 5:G3 解析 issue body 的 `- [ ] #N` checklist

**做法:** 不用 GitHub 原生 `sub_issues` webhook,沿用类似 `audit-split.ts` 的 markdown checkbox 解析方式判定父子关系。

**取舍分析:**

- 优:不依赖新 webhook 订阅,与现有 `audit-split` 模式一致
- 劣:**与 GitHub 原生 sub-issue 数据脱节** —— 开发者通过 GitHub UI 拖拽建立 / 调整 sub-issue 关系时,issue body 不会自动更新
- 劣:正则脆弱,checkbox 格式漂移即失效
- 劣:无法实时响应 sub-issue 关闭事件,只能轮询

**拒绝理由:** GitHub 原生 `sub_issues` webhook 已 GA,直接订阅可获得精准、实时的父子关系事件;放弃 markdown 解析既消除脆弱性,也与团队已经使用原生 sub-issue 功能的现状对齐。
