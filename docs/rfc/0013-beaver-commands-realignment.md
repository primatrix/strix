---
title: "RFC-0013: Beaver Commands 与项目管理流转预期对齐"
status: draft
author: sii-xinglong
date: 2026-04-22
reviewers: []
---

# RFC-0013: Beaver Commands 与项目管理流转预期对齐

## 概述

将 `plugins/beaver` 的 9 个命令与 `beaver-engine` 重构为以 GitHub Projects V2 字段 + 原生 Issue Type 为单一事实源，对齐 [项目管理框架](../onboarding/project-management.md) 中描述的状态机、Status / Type / Size 枚举与跨命令交接预期。

## 背景

### 技术现状

Beaver 当前位于 `plugins/beaver/` (v3.2.0)：9 个命令 (`beaver-{create,claim,design,decompose,dev,pr,tracker,focus,setup}.md`) + 1 个内部 skill (`beaver-engine`) + 9 个配套 bash 脚本于 `scripts/`。后端使用 GitHub Projects V2 (`primatrix/projects` org-project #14)，已存在的自定义字段为 `Level / Status / Progress / Iteration`，并已通过 `beaver-setup` 创建了原生 Issue Type `Goal / Task / SubTask / Milestone`（spec 在 PR #106 合并后已不再使用 Milestone，需在本次重构中一并淘汰）。

### 与 spec 的差异

[项目管理框架](../onboarding/project-management.md) 是本次对齐的权威 spec。当前 Beaver 与 spec 在以下维度不一致：

- **Status 载体**：spec 描述为 Project V2 单选字段（7 个值），Beaver 用 `status/*` 标签实现。
- **Type 载体**：spec（wiki PR #106 已合并）使用原生 Issue Type，集合精简为 `Goal / Task / SubTask / Bug / Feature` 共 5 个值（不再包含 Milestone）；Beaver 用 `type/*` 标签 (`feat/bug/refactor/docs/chore`)，枚举集合与字段位置都不一致，且现有 `beaver-setup` 创建的 `Milestone` Type 在 spec 中已不再使用。
- **Size 载体**：spec 定义 `XS / S / M / L / XL` 五档，Beaver 仅有 `size/S` / `size/L` 两档。
- **系统触发的状态迁移**（Iteration 加入 → Design Pending；Design Doc PR 合并 → Ready to Develop；SubTask 关闭 → 父 Done）：spec 描述为系统行为，Beaver 当前完全没有实现。
- **Roadmap 载体**：spec（wiki PR #106 已合并）将 Roadmap 工具映射定为 Project V2 原生 **Iteration** 字段，明确不再使用 Milestone（既不是 GitHub Milestone API，也不是 Issue Type）；Beaver 的 9 个命令与 scripts 已完成 Milestone → Iteration 的迁移（参见 commits `1f0b6bc / 7638710 / 7bb5d5e / 11ea69f / c5a59e8`），但 `beaver-engine` 文档与 `beaver-setup` 的 Issue Type 创建清单仍残留 `Milestone`，需一并清理。
- **Project tracker issue**：spec 要求每个 Project 在当前 Iteration 内维护一个 tracker issue 集中追踪本周期任务；Beaver 已存在 `beaver-tracker` 命令承担该职责，本次重构需保证 (a) tracker 创建/更新走 `beaver-lib.sh` 字段写路径而非 `status/*` 标签，(b) tracker 的 sub-issue 列表与 Project V2 内「Iteration=当前周期 ∧ repo 归属」的 Task 集合保持一致。
- **Task 起止时间**：spec 要求通过 Project 原生的 **Start date** / **Target date** 字段记录，不再依赖 Milestone 截止日。Beaver 不主动写这两个字段（保持 spec 中「由 Senior 在 Roadmap 制定阶段约定」的语义），但 `beaver-engine` 文档需声明此约束并避免误导命令去覆写。
- **Bug 通道**：spec（wiki PR #104 已合并）已放宽 Bug 的 size 与 priority 强制，并要求 Bug 创建时自动加入其所属 repo 的最新 Iteration；Beaver 当前 G008 仍强制 `size/S`，需删除并以 G011（Iteration 强制）替代。

### 范围

In-scope：所有 9 个 Beaver 命令 + `beaver-engine`；Project V2 #14 字段读写；命令间交接 `create → tracker → claim → design → decompose → dev → pr` 与 Bug 快速通道；`beaver-tracker` 的 per-Project tracker issue 维护（包含 sub-issue 列表与 Iteration 字段双向同步）；移除残留的 Milestone Issue Type 引用。

Out-of-scope（明示延迟）：(a) 系统触发的自动状态迁移（GitHub Actions / 后台 bot 基础设施），作为后续 Goal 处理；(b) `beaver-pr` 的 2-Reviewer + CI-green 合并门禁，保持当前 Draft + warning 行为；(c) Beaver 不主动写 Project V2 的 **Start date** / **Target date** 字段——spec 将其归为 Senior 在 Roadmap 制定阶段的人工约定，本次重构不引入相关命令；`beaver-engine` 仅声明该约束以避免命令误覆写。

## 方案

### 系统上下文

```text
┌──────────────────────┐         ┌─────────────────────────────────┐
│ User in Claude Code  │  /cmd   │ Beaver commands (9 .md files)   │
│                      │ ──────► │ + beaver-engine (rules)         │
└──────────────────────┘         └────────────┬────────────────────┘
                                              │ source
                                              ▼
                                 ┌─────────────────────────────┐
                                 │ scripts/beaver-lib.sh (NEW) │
                                 │  set_status / set_type /    │
                                 │  set_size / get_iteration / │
                                 │  resolve_item_id / ...      │
                                 └────────────┬────────────────┘
                                              │ gh api graphql
                                              ▼
                          ┌────────────────────────────────────────────┐
                          │ GitHub Projects V2 #14 (primatrix/projects)│
                          │ Fields: Level, Status (7), Size (5, NEW),  │
                          │         Progress, Iteration                │
                          │ Native Issue Types: Goal/Task/SubTask/     │
                          │   Milestone/Bug/Feature                    │
                          └────────────────────────────────────────────┘
```

### 分阶段重构

重构分两个 Phase。Phase A 为 3 个有序 SubTask，必须全部合并后 Phase B 才能开始。

#### Phase A — Engine + 基础设施（3 SubTasks）

| # | SubTask | 依赖 | 交付物 |
|---|---|---|---|
| A1 | `scripts/beaver-lib.sh` 初始库 + `--self-test` 子命令 | — | 新文件，含下列 public API；`bash scripts/beaver-lib.sh self-test` 通过 |
| A2 | `beaver-engine` 重写（§1 删除标签行；§2 状态机改为字段语义；§3 guardrail 重写；§4 Label Ops 替换为 Field Ops） | A1 | 重写后的 SKILL.md |
| A3 | `beaver-setup` 迁移：通过 `beaver-lib.sh::set_type` 创建 `Bug` 与 `Feature` 原生 Issue Type；创建 Project V2 单选 `Size` 字段（`XS / S / M / L / XL`） | A1 | `beaver-setup.md` + `beaver-setup.sh` 更新 |

A1 / A2 / A3 可在独立 PR 中提交，但 Phase B 任一 SubTask 启动前必须 **全部三者合并**。这符合 Issue #111 验收标准 2「对每个 Diff 点创建 Sub Task」。

#### Phase A.2 中的 guardrail 改写

- G001 / G002 / G006 / G009：从读 `status/*` / `type/*` / `size/*` 标签改为通过 `beaver-lib.sh` 读 Project V2 字段 + 原生 Issue Type。
- G008（Bug 强制 size/S）：**删除**——作者将同步修订 wiki spec 放宽该约束。
- G011（新增）：在 `/beaver-create` 处理 `Type=Bug` 时强制设置 Iteration，由下文算法解析。算法返回 null 时 G011 失败并提示用户运行 `/beaver-tracker <repo>`。

#### `latest_iteration_for_repo <repo>` 解析算法（G011 使用）

Iteration 字段定义在 Project V2 #14 上（不是 repo 级别），每个 entry 含 `title / startDate / duration`。"为该 repo 找最新 Iteration" 的语义被解释为「项目当前会把这个 repo 的任务路由到哪个 Iteration」：

1. 通过 GraphQL 拉取 Project #14 上的所有 Iteration entry（`title / startDate / duration`）。
2. 对每个 entry 计算 `endDate = startDate + duration days`。今日日期按 **UTC** 评估，避免边界日歧义。
3. **Step A**：选 `startDate <= today < endDate` 的 entry（"当前" Iteration）。若恰好 1 个，返回。
4. **Step B**：若 Step A 为 0，选 `startDate > today` 中 `startDate` 最早的 entry（"下一个未来" Iteration）。返回。
5. **Step C**：若 A 和 B 均为空，返回 null —— G011 失败，错误信息为 `"No current or future Iteration found on Project #14 for <repo>. Run /beaver-tracker <repo> to create this month's Iteration entry."`

Step A 返回 > 1 个的歧义路径：报错 `"Ambiguous current Iteration for <repo>: <list>. Resolve overlap before retrying."`。月度节奏下不会出现，但防御性检测。

未上线 Project #14 的 repo：算法本身没有 repo 维度。`/beaver-create` 已有的 `add-to-project` 步骤会先把 Issue 加入 Project，G011 在 project item 上运行；因此「未上线 repo」实际等价于「Issue 尚未挂到 Project #14」，由 `add-to-project` 处理。

#### `scripts/beaver-lib.sh` Public API

| 函数 | 作用对象 | 底层 mutation/query |
|---|---|---|
| `resolve_item_id <issue_url>` | Project V2 #14 | GraphQL `repository.issue.projectItems` 过滤 project #14 |
| `get_field_id <field_name>` | Project V2 #14 | GraphQL `node.fields` 查找 |
| `get_option_id <field_name> <option_name>` | Project V2 #14 单选字段 | GraphQL field options 查找 |
| `set_status <issue_number> <status>` | Project V2 #14 Status 字段 | `updateProjectV2ItemFieldValue`（singleSelectOptionId） |
| `set_size <issue_number> <size>` | Project V2 #14 Size 字段 | `updateProjectV2ItemFieldValue`（singleSelectOptionId） |
| `set_type <issue_number> <type>` | **原生 GitHub Issue Type**（org 范围） | `updateIssueIssueType`；要求 `admin:org` scope（与现有 `beaver-setup` 一致） |
| `get_type <issue_number>` | 原生 Issue Type | `repository.issue.issueType` |
| `get_iteration <issue_number>` | Project V2 #14 Iteration 字段 | `projectV2Item.fieldValueByName(name: "Iteration")` |
| `set_iteration <issue_number> <iteration_title>` | Project V2 #14 Iteration 字段 | `updateProjectV2ItemFieldValue`（iterationId） |
| `latest_iteration_for_repo <repo>` | Project V2 #14 Iteration 字段 | 拉取所有 Iteration entry，应用 G011 算法 |

`set_type` 在调用界面上与 `set_status` / `set_size` 同形，但底层走 `updateIssueIssueType` 而非 `updateProjectV2ItemFieldValue`——Type 是仓库/组织级原生属性，不是 Project V2 字段。这种非对称性被 `beaver-lib.sh` 屏蔽在调用方之外。

A3 重写 `beaver-setup` 中的 Issue Type 创建逻辑，统一通过 `beaver-lib.sh::set_type`（而非现有的直接 `gh api /orgs/<org>/issue-types`），保证全仓单一调用路径。

#### Phase B — 命令逐个迁移（8 SubTasks，并行）

每个命令一个 SubTask，作用对象：`beaver-create / beaver-claim / beaver-design / beaver-decompose / beaver-dev / beaver-pr / beaver-tracker / beaver-focus`（`beaver-setup` 已在 A3 处理）。

每个 SubTask 工作量：

1. Frontmatter `allowed-tools`：移除 `gh label:*`，确保 `gh api graphql:*`。
2. 命令正文：替换所有 `status/*` / `type/*` / `size/S|L` 文本为字段 / 原生 Type 语义。
3. `scripts/<command>.sh`：`source scripts/beaver-lib.sh`，将 `gh api repos/.../labels` 调用替换为库函数调用。
4. 沙盒 smoke：在 `primatrix/projects` 上跑一遍命令的 lifecycle 步骤，PR 描述记录会话。

`beaver-tracker` 的 SubTask 额外承担 spec 中「每个 Project 在当前 Iteration 内维护一个 tracker issue」的语义实现：

- tracker issue 的 sub-issue 列表 = `{Project V2 #14 内 ∧ Iteration=当前周期 ∧ repo 归属=<repo> ∧ Type ∈ {Task, Bug, Feature}}`，每次命令运行需做一次差集同步（add 缺失 sub-issue，remove 已不属于当前 Iteration 的 sub-issue）。
- tracker issue 自身的 Iteration 字段通过 `beaver-lib.sh::set_iteration` 写入，标识其归属周期。
- 上述差集查询走 `beaver-lib.sh::get_iteration` + GraphQL `projectV2.items` 过滤，不依赖 `status/*` 标签。

### 接口与数据流

- **读路径**：command-script → `beaver-lib.sh` `get_*` → GraphQL `projectV2Item.fieldValueByName` → typed value（单选 option name / iteration title）。
- **写路径**：command-script → `beaver-lib.sh` `set_*` → GraphQL `updateProjectV2ItemFieldValue` 或 `updateIssueIssueType`。每字段原子。
- **跨命令交接**：形态不变，每个命令把 Status 切到下一个合法值（通过 `set_status`），并打印下一步提示（如 `/beaver-create` 之后输出 `/beaver-claim N`）。

### 关键 trade-off

| Trade-off | 决策 | 理由 |
|---|---|---|
| Status / Size 用标签 vs Project V2 字段 | 字段 | Wiki 为权威 spec |
| Type 用 `type/*` 标签 vs 原生 Issue Type | 原生 Issue Type | Wiki 权威；`beaver-lib.sh::set_type` 屏蔽 GraphQL 非对称性 |
| 大爆炸 vs 分阶段重写 | 分阶段（A1/A2/A3 → B 并行） | engine + 库必须先于命令；命令可由不同人/agent 并行推进 |
| 共享 lib vs 每个 script 内联 helper | 共享 `beaver-lib.sh` | 消除漂移；`resolve-iteration` 等 helper 已在 `beaver-create.sh` 与 `beaver-tracker.sh` 中重复出现 |
| 仓库侧标签定义 | 保留在 `primatrix/projects`；移除的是源代码侧引用 | 可逆性——错误回滚不会丢失历史标签数据 |
| 系统侧自动迁移 | 本次不实现 | Actions 基础设施扩展范围约 2 倍；作为后续 Goal |
| Bug size / priority 强制 | 移除（删除 G008；G011 强制 Iteration） | wiki PR #104 已合并到 main，spec 与本 RFC 同步 |

### 测试策略

- **A1 SubTask**：`bash scripts/beaver-lib.sh self-test` —— 一次性把每个 public 函数串起来跑一遍沙盒 issue（create → set_status → read → set_size → read → set_type → read → close），断言 round-trip 相等。
- **A2 SubTask**：`beaver-engine/SKILL.md` 渲染评审 + grep 确保 engine 文档（除「废弃说明」段外）无 `status/*` / `type/*` 引用。
- **A3 SubTask**：在 `primatrix/projects` 上重跑 `beaver-setup`，断言：(a) `gh project field-list primatrix 14 --format json | jq '.fields[] | select(.name=="Size") | .options[].name'` 返回 `XS/S/M/L/XL`；(b) `gh api /orgs/primatrix/issue-types` 返回包含 `Bug` 与 `Feature`。
- **Phase B SubTasks**：每命令 PR 含沙盒 smoke 表（Step | Command | Expected | Observed）。
- **父 Issue 关闭前的最终验证**：见下文「成功指标」全部 4 项。

### 备选方案

**Alt A — 反向：保留当前 Beaver 标签 taxonomy，修订 wiki 适配 Beaver。** 工程量最低（Phase A 几乎为零），仅做内部一致性整理。**否决**：Issue #111 明确把 wiki 列为对齐目标；反转契约方向只会让团队的参考文档继续与现实不符，仅是把矛盾换边。Wiki 是先行的团队约定，命令必须追上，不应改写约定。

**Alt B — 混合 taxonomy（Status 用字段；Type / Size 仍用标签）。** Status 是 lifecycle 关键枚举因此值得字段化；Type/Size 多为只读筛选条件，标签足够。迁移成本最低。**否决**：每个命令需同时学两套集成模式（`gh api .../labels` for Type/Size + `gh api graphql` for Status）；G001/G002/G006/G009 必须混合读取标签和字段，失败模式增加；`beaver-lib.sh` 的去重价值大幅缩水。

**Alt D — 把系统侧自动迁移纳入本次 Goal。** 一次性把 Iteration-add → Design Pending、Design-PR-merge → Ready to Develop、SubTask close → parent Done 三个迁移用 GitHub Actions 实现。**否决**：需要 (a) 每个 trigger 一个 workflow；(b) PAT/GITHUB_TOKEN scope 评审；(c) 半应用迁移的回滚策略；(d) 单独的沙盒 project 用于测试，避免污染 #14。SubTask 数量约翻倍。Section 1 已明确放在范围外；本次重构需保持自包含、可评审的体量。

## 影响范围

- **代码**：`plugins/beaver/` 下 9 个命令 + 1 个 engine SKILL + 9 个 scripts，新增 1 个共享 lib（`scripts/beaver-lib.sh`）。
- **GitHub Projects V2 #14**：新增 Project 单选字段 `Size`（`XS/S/M/L/XL`）。
- **GitHub 组织级 Issue Types**：新增 `Bug` 与 `Feature`；`Goal/Task/SubTask` 已存在；`Milestone` 标记为弃用（不再被任何命令引用，但历史已创建的实例不主动删除）。
- **`primatrix/projects` 仓库标签**：`status/*` / `type/*` / `size/*` 标签定义本次保留不删除（仅停止源代码引用）。
- **使用方**：所有用 Beaver 命令做项目管理流转的开发者（迁移期内既有 Issue 的字段需用 `beaver-setup` 重跑或手工补齐 Status/Size 字段）。

### 成功指标

1. **End-to-end smoke（Draft-PR 终态）**：一个新建 size/M Issue 走完 `create → tracker → claim → design → decompose → dev → pr` 后落在 `Status=In Progress` + 一个 Draft PR + Project V2 字段值与 spec 表对齐，过程中无任何手动标签 / 字段编辑、无 Beaver script 之外的 `gh api` 调用。Pass = 录制的 session 日志。
2. **字段形态**：`gh project field-list primatrix 14 --format json | jq '.fields[] | select(.name=="Status") | .options[].name'` 精确返回 7 个 wiki 值；同样查询 `Size` 返回 `XS / S / M / L / XL`；`gh api /orgs/primatrix/issue-types` 返回包含 5 个 wiki Type 值（`Goal / Task / SubTask / Bug / Feature`），且 `git grep -nE "issue-type.*Milestone|--issue-type ['\"]?Milestone" plugins/beaver/` 在源码侧零命中。
3. **源码侧标签清理**：`git grep -nE "(gh label (add|create|delete|remove))|gh api[^|]*labels.*(status/|type/|size/)" plugins/beaver/` 在命令正文 / scripts / engine 上返回零命中。仓库侧标签定义保留不动。
4. **Bug fast-path smoke**：新建 `Bug` Issue 且 `Priority = P0`，单次 `/beaver-create` 完成后即落在 `Status = In Progress` 且按 G011 算法解析的 Iteration 已分配。
5. **Project tracker 一致性**：在已存在若干 Iteration=当前周期 Task 的 repo 上跑 `/beaver-tracker <repo>`，运行后 (a) `primatrix/projects` 中存在标 `tracker/<YYYY-MM>` + `tracker/<repo>` 的 issue；(b) 该 issue 的 sub-issue 集合等于「Project V2 #14 内 Iteration=当前周期 ∧ repo 归属=`<repo>` ∧ Type ∈ {Task, Bug, Feature}」的 Task 集合（差集为空）；(c) 该 tracker issue 自身的 Iteration 字段已设为当前周期。

## 实施计划

| 阶段 | SubTask | 依赖 | 交付 |
|---|---|---|---|
| Phase A.1 | `scripts/beaver-lib.sh` 初始库 + `--self-test` | — | 新文件 + 沙盒 round-trip 通过 |
| Phase A.2 | `beaver-engine` 重写 | A.1 | 字段化 §1–§4，guardrail G008 删除 / G011 新增 |
| Phase A.3 | `beaver-setup` 迁移 | A.1 | `Bug`/`Feature` Issue Type + Project V2 `Size` 字段 |
| Phase B (×8) | 每命令一个 SubTask（`beaver-{create,claim,design,decompose,dev,pr,tracker,focus}`） | Phase A 全部合并 | 命令 + script 字段化 + 沙盒 smoke |
| 收尾 | 父 Issue #111 关闭 | Phase B 全部合并 | 5 项成功指标全部通过 |

Phase A 的三个 SubTask 之间可并行评审（仅 A.2 / A.3 实现上引用 A.1），但合并顺序仍为 A.1 → 然后 A.2 / A.3。Phase B 八个 SubTask 完全并行。

## 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 既有 Issue 的 Project V2 字段未填值 | 迁移完成后旧 Issue 在 dashboard 上显示为空 Status / Size | A.3 可附带一次性扫描脚本，把仍带 `status/*` / `size/*` 标签的 Issue 写入对应字段 |
| `set_type` 需要 `admin:org` scope | 个别 contributor 的 token 缺少 scope | 命令在 `set_type` 失败时打印明确的 `gh auth refresh -h github.com -s admin:org` 提示 |
| 系统侧自动迁移仍未实现 | 用户要手动跑 `/beaver-claim` 才能 `ready-to-claim → design-pending`，与 wiki 描述的 "system behavior" 不符 | 在本 RFC 末尾建立后续 Goal「Beaver 自动迁移基础设施」，并在 `beaver-engine` §2 注明哪些迁移目前是手工触发 |
| Bug 通道 wiki spec 修订未先于本次合并 | G011 已上线但 wiki 仍说 Bug 强制 `size/S` | 已解除：wiki PR #104（Bug 不强制 Size，自动加入 Iteration）与 PR #106（Roadmap 改用 Iteration 字段 + per-Project tracker）均已合并到 main，本 RFC 已基于该状态对齐。 |
| Phase B 八个 SubTask 并行触发对 `primatrix/projects` Project #14 的写竞争 | 沙盒 smoke 之间相互污染 | 每个 SubTask 创建独立的 `[smoke] <command>` Issue，跑完即 close；review 后由作者手动归档 |

<!-- provenance
- "Beaver v3.2.0 9 命令 + 1 engine + 9 scripts" ← Discovery D2 (Glob plugins/beaver/**/*) + plugin.json
- "Project V2 #14 已有字段 Level/Status/Progress/Iteration" ← Discovery D3 (gh project view 14 --jq .readme)
- "已有原生 Issue Type Goal/Task/SubTask/Milestone" ← beaver-setup.md §Create Issue Types
- "近期 commit 1f0b6bc / 7638710 / 7bb5d5e / 11ea69f / c5a59e8 已迁移 Milestone→Iteration" ← Discovery D1 (git log)
- "wiki spec 7-phase lifecycle / 7 Status / 5 Size / 6 Type / system-vs-user trigger" ← https://github.com/primatrix/wiki/blob/main/docs/onboarding/project-management.md (gh api fetch)
- "G001/G002/G004/G006/G007/G008/G009/G010 名单" ← beaver-engine/SKILL.md §3 (Read)
- "primatrix/projects 当前是 issueRepo" ← beaver-config block in Project #14 README
- "Wiki 为权威 spec" ← QA Section 1 Q1
- "Status 用 Project V2 字段，drop status/* 标签" ← QA Section 1 Q2
- "Type 用原生 Issue Type，drop type/* 标签" ← QA Section 1 Q3
- "Size 5 档 XS/S/M/L/XL" ← QA Section 1 Q4
- "Bug auto-Iteration 替代 size/priority 强制；wiki 由作者另行修订" ← QA Section 1 Q5
- "系统侧自动迁移本次 out-of-scope，作为后续 Goal" ← QA Section 1 Q6
- "Goal: end-to-end runnable lifecycle 无手动 fallback" ← QA Section 2 Q1
- "Non-goals: no automation / no wiki edits / no new deps" ← QA Section 2 Q2
- "Success metrics: 4-point smoke + grep + field-list" ← QA Section 2 Q3
- "Phased rollout: engine first then commands parallel" ← QA Section 3 Q1
- "scripts/beaver-lib.sh 共享库" ← QA Section 3 Q2
- "Guardrail policy: 重写 G001/G002/G006/G009；删 G008；加 G011" ← QA Section 3 Q3
- "Per-command sandbox smoke" ← QA Section 3 Q4
- "beaver-setup 添加新字段/类型，旧标签留在 repo 上不删" ← QA Section 3 Q5
- "G011 算法 + UTC 边界 + 多 Iteration 错误 + 未上线 repo 处理" ← Spec review round 2 fixes
- "Metric 1 落点改为 Draft PR 终态" ← Spec review round 2 fix
- "set_type 走 updateIssueIssueType 而非 updateProjectV2ItemFieldValue" ← Spec review round 2 fix
- "Phase A 拆为 A1/A2/A3 三个有序 SubTask" ← Spec review round 2 fix
- "Metric 3 grep scope 限定于源码侧调用" ← Spec review round 2 fix
- "Type 6 值在 Metric 2 中显式断言" ← Spec review round 2 minor fix
- "Roadmap 改用 Project V2 Iteration 字段 + per-Project tracker issue + Start/Target date 字段" ← wiki PR #106 (commit 63c2e06)
- "Bug 不强制 Size，自动加入当前 Iteration" ← wiki PR #104 (commit 38fcec1)
- "Type 集合精简为 5 个值（去掉 Milestone）" ← wiki PR #106 main (latest project-management.md line 66)
- "Metric 5 (tracker 一致性) + Phase B beaver-tracker 额外职责" ← 对齐 wiki PR #106 spec
-->
