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

Beaver 当前位于 `plugins/beaver/` (v3.2.0)：9 个命令 (`beaver-{create,claim,design,decompose,dev,pr,tracker,focus,setup}.md`) + 1 个内部 skill (`beaver-engine`) + 9 个配套 bash 脚本于 `plugins/beaver/scripts/`（即本 RFC 后文出现的 `scripts/beaver-lib.sh` 等路径均为 `plugins/beaver/scripts/` 的简写，Metric 3 的 grep 范围 `plugins/beaver/` 已覆盖该子目录）。后端使用 GitHub Projects V2 (`primatrix/projects` org-project #14)，已存在的自定义字段为 `Level / Status / Progress / Iteration`，并已通过 `beaver-setup` 创建了原生 Issue Type `Task / SubTask / Milestone`（spec 在 PR #106 合并后已不再使用 Milestone，需在本次重构中一并淘汰）。

### 与 spec 的差异

[项目管理框架](../onboarding/project-management.md) 是本次对齐的权威 spec。当前 Beaver 与 spec 在以下维度不一致：

- **Status 载体**：spec 描述为 Project V2 单选字段（7 个值），Beaver 用 `status/*` 标签实现。
- **Type 载体**：spec（wiki PR #106 已合并）使用原生 Issue Type，集合精简为 `Bug / Task / SubTask` 共 3 个值（不再包含 Goal、Feature、Milestone）；Beaver 用 `type/*` 标签 (`feat/bug/refactor/docs/chore`)，枚举集合与字段位置都不一致，且现有 `beaver-setup` 创建的 `Goal`、`Milestone` Type 在 spec 中已不再使用。
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
                          │ Native Issue Types: Bug/Task/SubTask       │
                          │                                            │
                          └────────────────────────────────────────────┘
```

### 命令规约（重构后预期 workflow）

以下小节描述 9 个 Beaver 命令在重构完成后**期望**的端到端工作流程，以及 1 个新增命令 `/beaver-fix`，合计 10 条命令规约，作为 Phase B 各 SubTask 的对齐基线与最终验收的可枚举依据。每个命令以「触发场景 → 用户与系统的逐步交互 → 期望终态」的叙事形式给出，并不假设当前 plugins/beaver 实现的具体形态——若现有实现与本节描述存在差异，应以本节为准、由 Phase B 中相应 SubTask 把实现对齐到本节。`/beaver-fix` 为本 RFC 新增命令，其实现 SubTask 不包含在 Phase B 的 8 个并行 SubTask 中，需另行拆解。

所有「字段」均指 Project V2 #14 的自定义字段；所有 Type 值均指原生 GitHub Issue Type；不再出现 `status/*` / `type/*` / `size/*` 标签。

跨命令的状态机主路径（size/L Task）：
`Triage → Ready to Claim → Design Pending → Ready to Develop → In Progress → Done`。
size/S Task 与 Bug 跳过 Design Pending / Ready to Develop。
若描述中出现「命令写入 Status」即指本次实现的命令侧迁移；「系统迁移」指由 out-of-scope 的 GitHub Actions 完成（本次不实现，命令侧不写，可能仍需人工触发）。

**语言约定**：所有命令的交互式问答（QA）与终端输出（提示、错误、下一步建议）均使用**中文**。

#### 1. `/beaver-create`

**触发场景**：用户希望把一项尚未存在的工作落库——可能是一个新的 Task/SubTask，或一个刚被发现的 Bug——并希望走团队约定的字段化、状态化流程，而不是直接 `gh issue create` 然后手工补字段。

**预期 workflow**：

1. **意图收集**：用户运行命令，并以自然语言描述要创建的 Issue 内容（一段话或若干要点均可）。命令根据描述内容推断 Issue Type（`bug` 或 `task`），向用户展示推断结果与依据，等待用户确认或纠正。用户也可在命令行直接附带 `--type bug|task` 跳过推断。确认后意图分流为两条子流程：Task 或 Bug。
2. **目标 repo 解析**：命令从 Project #14 README 中嵌入的 `beaver-config` 读出当前 `issueRepo`（即 Issue 创建的目标仓库，不存在「关联 repo」字段——`beaver-config` 中 `issueRepo` 若为 `all` 则默认 `primatrix/projects`），不要求用户输入。`beaver-config` 缺失或解析失败时命令立即中止，并提示用户先运行 `/beaver-setup`。
3. **代码调研（上下文补充）**：在进入结构化问答前，命令在 `issueRepo` 下根据用户在第 1 步描述的内容，检索并阅读相关代码文件（文件名关键词匹配、函数签名搜索等），将调研摘要作为后续 QA 的「已有上下文」展示给用户，并在 Issue body 中以 `<!-- context -->` 块附注关键发现。此步骤自动执行，无需用户操作，但若 `issueRepo` 为非代码仓库（如 `primatrix/projects`）则跳过。
4. **结构化问答（QA 循环）**：命令通过逐段问答收集 Issue 内容，每段一个问题、一个回答、一次回显确认，禁止用户在命令行一次性传入完整 body。两条子流程问答模板不同：
   - **Size=L Task**：4 段——(a) 层级与父 Issue（用以推导 `Level`）、(b) 客观 objective、(c) 验收标准、(d) 已知约束 / 风险。
   - **Size=S Task**：3 段最小问答——objective、验收标准、依赖项。
   - **Bug**：4 段——复现步骤、期望行为、实际行为、影响范围 + 环境。
5. **Size 推荐与确认**：QA 收集 objective 后，命令基于描述复杂度（行数、是否跨模块、是否引入新依赖）自动建议 `Size ∈ {XS, S, M, L, XL}`，并展示推理依据。用户确认或在五档内覆盖；这是用户唯一直接编辑的 Project V2 字段。
6. **Iteration 决策**：
   - Task 路径：命令询问用户「跳过 / 加入当前 Iteration / 选择某月 Iteration」，三选一。选择 `skip` 时 Issue 创建后 `Iteration` 字段留空，等待 `/beaver-tracker` 在后续周期把它纳入。**若用户选择加入某个 Iteration 且在第 4 步声明了父 Issue，命令额外把父 Issue 关联到该 Iteration 的 tracker issue（`[Iteration] <repo> <YYYY-MM>`）作为 sub-issue，使父卡在 tracker 视图中可见；若 tracker issue 不存在，命令打印提示「建议先运行 `/beaver-tracker <repo> <YYYY-MM>` 再创建子任务」，但不阻断本次创建。**
   - Bug 路径：用户不被询问；命令按 §「`latest_iteration_for_repo` 解析算法」自动解析「当前 Iteration 或下一个未来 Iteration」并写入。算法返回空 → G011 失败 → 命令中止并提示 `/beaver-tracker <repo>`。
7. **Issue 预览 HARD-GATE**：所有写操作之前，命令把即将创建的 Issue 完整渲染（标题、body、推导出的所有字段值、父 Issue 链接、是否会 `@CODEOWNERS`）一次性展示给用户，等待用户显式输入 `yes` 才进入第 8 步。任何 `no` / Ctrl-C 都使命令零副作用退出。
8. **落库**（按以下顺序，前一步失败即整体回滚或人工干预提示）：
   a. 在 `issueRepo` 创建 Issue，body 即第 4 步收集的内容；P0 Bug 在 body 中 `@CODEOWNERS` mention。
   b. 若用户在第 4 步声明了 parent，通过 Sub-Issues API 链接到 parent —— **必须先于** 把 child 加入 Project #14，否则父卡会出现 "1 sub-issue not in this project" 的不一致。
   c. 把 Issue 加入 Project #14。
   d. 写入 Project V2 字段：`Type`（推导自 `<issue-type>`）、`Level`（推导自父子结构）、`Size`（用户确认值）、`Status`（默认 `Triage`；P0/blocker Bug 直接 `In Progress`）、`Iteration`（Bug 必填、Task 按用户选择）。每个字段的写入相互独立，不存在「写 Status 同时改 Type」的复合操作。
   e. 若第 6 步 Task 路径触发了父 Issue 关联 tracker，把父 Issue 通过 Sub-Issues API 挂到该 tracker issue 下，并写入父 Issue 的 `Iteration` 字段为所选 `<YYYY-MM>`。
9. **下一步建议**：命令打印 Issue URL 与下一步指引——
   - Task 未挂 Iteration → `/beaver-tracker <repo>`；
   - Task 已挂 Iteration → 等待系统迁移 `Triage → Ready to Claim` 后由其他成员在 GitHub UI assign 自己并手动切 Status（系统迁移本次 out-of-scope，可能仍需人工触发）；
   - 常规 Bug（P1/P2）→ 在 GitHub UI assign 自己并手动将 Status 切为 `In Progress`；
   - P0 Bug → 已为 `In Progress`，可直接 `/beaver-dev <number>` 或等待被 mention 的负责人响应。

**Guardrail**：G002（Type 必填，第 1 步推断后用户拒绝确认即中止）、G011（Bug 路径下 Iteration 解析失败即拒绝创建）；G008（Bug 强制 size/S）已删除。

**期望终态**：`issueRepo` 中存在新 Issue；该 Issue 已加入 Project #14；该 Project 项的 `Type / Level / Size / Status / Iteration` 字段值与上文 8.d 一致；若有父 Issue 且用户选择了 Iteration，父 Issue 已关联到对应 tracker；终端给出明确的 next-step 指引；本命令未触碰任何 `status/* / type/* / size/*` 标签。

#### 2. `/beaver-tracker`

**触发场景**：每个月初，团队需要为某个 repo 在 Project #14 上"开一个新本月的工作面"——把本月要推进的 Task 集中挂到一张 tracker Issue 下，把上月没做完的 open sub-issue 滚进本月，并从 backlog 中拉取本月新承接的项目。或者月中需要补一个之前忘了拉进 Iteration 的 Task。

**预期 workflow**：

1. **入参解析**：用户运行命令并给出 `<repo>`（必填，限定 `primatrix/<repo>`）与可选 `[YYYY-MM]`（默认本地当前年月）。命令立即校验 Project #14 上是否存在标题以 `<YYYY-MM>` 开头的 Iteration entry——缺失则中止并提示 `/beaver-setup`。
2. **唯一性检查**：命令在 `primatrix/projects` 上搜索是否已存在标题为 `[Iteration] <repo> <YYYY-MM>` 的 Issue。
   - 不存在：进入第 3 步创建。
   - 存在恰好 1 个：复用，进入第 4 步同步。
   - 存在多个：中止并打印冲突列表，要求人工合并。
3. **tracker Issue 创建**：在 `primatrix/projects` 创建标题为 `[Iteration] <repo> <YYYY-MM>` 的 Issue，body 含一段固定模板（说明该 tracker 的用途、所属周期、所属 repo），并打上仓库级标签 `tracker / tracker/<repo> / tracker/<YYYY-MM>`。这三个标签是 Beaver 自身的元数据通道，独立于被淘汰的 `status/* / type/* / size/*` taxonomy。tracker Issue 自身被加入 Project #14，并通过字段写入 `Iteration=<YYYY-MM>`。
4. **上月 carry-over**：命令查上一个 `YYYY-MM-1` 的 tracker（若有）；将其下所有 `state=open` 的 sub-issue 收集为 carry-over 候选集，并向用户展示「这些是上月未完成的，是否全部带入本月？」。用户可全选、全拒、或逐项勾选。被选中的 sub-issue 在第 6 步统一处理。
5. **Backlog 拉取**：命令从 Project #14 中查询「`Iteration` 字段未填 ∧ `Status=Triage` ∧ repo 归属=`<repo>` ∧ Type ∈ {Task, Bug}」的 Issue 集合作为 backlog 候选；以列表形式展示，让用户逐项决定是否纳入本周期。
6. **批量挂载（写差集）**：对第 4、5 两步选中的每个 sub-issue（先加入 sub-issue、再 add-to-project，与 `/beaver-decompose` 的落库顺序一致，避免父卡出现 "sub-issue not in this project" 的不一致）：
   a. 通过 Sub-Issues API 把 sub-issue 挂到本月 tracker 下；
   b. 若尚未在 Project #14 中，`add-to-project`；
   c. 写 Project V2 字段 `Iteration=<YYYY-MM>`。
7. **解挂同步**：命令查询本月 tracker 当前已有的 sub-issue 集合，识别出「已挂在 tracker 下，但 `Iteration` 字段不再是 `<YYYY-MM>` 或 repo 归属不再是 `<repo>`」的过期项，逐个从 tracker 下解除挂载（不删 Issue、不改其它字段）。这是把 tracker 视图变成 Project V2 字段的"投影"，确保两侧一致。
8. **完成回执**：命令打印 tracker Issue URL、本周期 sub-issue 数量、carry-over 数量、新拉取数量、解挂数量四项统计，便于 Senior 在周会上对账。

**Guardrail**：唯一性检查（第 2 步）；不再涉及任何 `status/*` 标签校验。

**期望终态**：`primatrix/projects` 下当月恰好存在 1 个 `[Iteration] <repo> <YYYY-MM>` Issue；该 Issue 的 sub-issue 集合等于 `{Project V2 #14 内 ∧ Iteration=<YYYY-MM> ∧ repo 归属=<repo> ∧ Type ∈ {Task, Bug}}`；tracker Issue 自身的 `Iteration` 字段已写入；所有被纳入的 sub-issue 的 `Iteration` 字段已对齐到 `<YYYY-MM>`（成功指标 5）。

#### 3. `/beaver-claim` ~~（已删除）~~

> **⚠️ 本命令在本次重构中删除。**
>
> 原设计中，`/beaver-claim` 同时承担「写 GitHub Assignee」与「迁移 Project V2 Status」两个职责。重新梳理后，两者应解耦：
>
> - **写 Assignee**：用户直接在 GitHub UI 或 `gh issue edit --add-assignee @me` 操作，无需专用命令。
> - **状态迁移**（`Ready to Claim → Design Pending / In Progress`）：预期由**系统根据 assign 事件自动触发**（GitHub Actions webhook），属于系统行为，不应由用户命令驱动。该自动化与其他系统迁移一同纳入 out-of-scope 的后续 Goal「Beaver 自动迁移基础设施」。
>
> **过渡期**（自动化上线前）：用户认领时，直接在 GitHub UI assign 自己，再手动在 Project #14 将 Status 切换到对应值（Size=L → `Design Pending`；Size=S / Bug → `In Progress`）。`/beaver-engine` §2 会列明哪些迁移目前仍需手工触发。

#### 4. `/beaver-design`

**触发场景**：Size=L 的 Task 在被认领后处于 `Design Pending`，意味着团队约定需要先沉淀一份可评审的 Design Doc 再开工。命令的职责是把"我有一个想法"逐步打磨成一份结构化的 RFC，并以 Draft PR 的形式提交到 wiki 仓库等待评审，而不是把设计直接散落在 Issue 评论中。

**预期 workflow**：

1. **入参与前置校验**：用户运行命令并给出 `<issue-number>`。命令读取该 Issue 在 Project #14 中的字段，校验 `Type=Task ∧ Size=L ∧ Status=Design Pending` 且当前用户为 assignee；任一不满足即中止并解释原因。
2. **wiki 工作树准备**：命令检查本地是否已有 `~/Code/wiki` clone；若没有则 clone，若已有则在 `main` 上 fetch + reset 到最新远端，并新开一个 `design/<n>-<slug>` 分支。`<slug>` 由 Issue 标题派生。
3. **设计资料采集**：命令读取 Issue body 作为设计意图的主要来源，并据此在当前仓库中主动搜索与阅读相关代码（涉及的模块、接口定义、现有实现等），将 Issue body 内容与代码阅读结果合并作为后续问答的"已有上下文"。
4. **结构化问答**：命令按四个维度逐一与用户进行 QA；每个维度先由命令列出当前上下文中尚不清晰或存在歧义的内容，再逐项与用户确认，直到命令判断该维度的信息已足够完整为止，再进入下一个维度；全程禁止跨维度跳问：
   - **Context & Scope**：技术现状、系统边界、客观背景事实、与现有模块的关系；
   - **Design Goals**：可量化的目标、明确不做的非目标、成功指标；
   - **The Design**：架构、组件、接口、数据流、技术选型理由、关键 trade-offs、测试策略、部署依赖；
   - **Alternatives Considered**：命令从已有上下文中识别出当前设计核心决策点的主要替代方案，逐一呈现给用户并询问为什么不采用该方案；用户的回答即为"被否决的可行方案与否决理由"，直到命令认为所有重要替代方案均已覆盖为止。
5. **草稿生成与逐段确认**：命令把收集到的回答拼装为完整 RFC 文档（遵循本 wiki 的 RFC 格式约定，含末尾 `<!-- provenance -->` 块标注每条事实的出处），逐段展示给用户审批。用户可对任一段落要求修改，命令即重新生成该段，直到用户全部满意。
6. **spec-document-reviewer 自检循环**：在 push 前命令调度 `spec-document-reviewer` subagent 对草稿进行最多 5 轮迭代评审；每一轮 reviewer 提出的问题由命令转给用户回答、并把回答合入草稿。直到 reviewer 给出 PASS，命令才允许进入第 7 步。任意一轮 reviewer 给出 BLOCK，循环继续；5 轮仍未通过则中止并要求人工介入。`spec-document-reviewer` 是一个 markdown 模板片段（类似 `subagent-driven-development/spec-reviewer-prompt.md`），**当前尚不存在，需单独创建**；其目标是审计设计文档的质量与内容，包括：每条事实是否可追溯到来源（anti-hallucination）、四个维度是否覆盖完整、逻辑是否自洽。
7. **PR 提交**：命令在 wiki 仓库写入 `docs/rfc/NNNN-<slug>.md`、在 `docs/rfc/index.md` 追加一行索引、commit、push、用 `gh pr create --draft` 打开 Draft PR；PR body 含「Closes #`<n>`」之外的内容（不能在合并时关 Task Issue，因为 Task 还要继续开发）。
8. **回写原 Issue**：命令在原 Task Issue 上评论 Design Doc PR 链接，便于其他人从 Issue 快速跳到 RFC。
9. **下一步指引**：命令打印 PR URL 与提示——「请自审 Draft → 转 Open → 等 Reviewer 通过 → 合并；PR 合并后 Status 由系统迁移到 `Ready to Develop`（本次 RFC 范围内系统迁移可能仍需人工触发）；之后用 `/beaver-decompose <n> --design-doc <pr-url>` 拆解 SubTask」。

**写字段**：本命令本身**不修改**任何 Project V2 字段。Status 保持 `Design Pending`；推进到 `Ready to Develop` 是 Design Doc PR 合并后的系统迁移职责（out-of-scope）。

**Guardrail**：spec-document-reviewer 必须通过（当前需先创建该模板片段）；前置校验失败立即中止。

**期望终态**：wiki 仓库上存在 Draft PR，文件位于 `docs/rfc/NNNN-<slug>.md`、`docs/rfc/index.md` 已追加；原 Task Issue 上有指向该 PR 的评论；Issue 在 Project #14 上的 Status 仍为 `Design Pending`，等待合并后的系统迁移。

#### 5. `/beaver-decompose`

**触发场景**：Size=L Task 的 Design Doc PR 合并后，认领人需要把"一份大的设计"切成若干"可以由不同人同时推进的最小可交付单元"，落到 GitHub 上即为：父 Task 下挂 N 个 SubTask Issue。命令负责把这一拆解过程结构化、可审计，并自动做基本的拆分质量审计。

**预期 workflow**：

1. **入参解析**：用户运行命令并给出 `<issue-number>` 与 `--design-doc <url-or-path>`（PR URL / blob URL / 本地文件路径三选一，必填）。命令读取父 Issue 的字段并校验 `Type=Task ∧ Status=Ready to Develop`（即设计已合并）；不满足则中止。
2. **Design Doc 摄取**：命令把 `--design-doc` 指向的 markdown 全文读入，并提取「方案 / 实施计划 / 测试策略」三段作为拆解依据。
3. **初步拆解建议**：命令基于 Design Doc 内容生成一份 SubTask 候选清单（每项含：建议标题、对应设计章节、预计 Size、依赖关系、是否需要 test 文件）。
4. **per-child QA 审批**：对每个候选 child，命令按 §7 流程逐项展示，用户可：保留、修改标题/描述、拆得更细、合并相邻两项、删除整项、或追加新项。直到用户确认整张列表。
5. **自动 audit**：命令对最终列表运行三类审计——
   - **Coverage**：Design Doc §方案 中列出的每个组件 / 接口是否至少被一个 child 覆盖；
   - **Atomicity**：是否存在 child 描述跨多模块且无法独立 review；
   - **Tests**：每个 child 是否在描述中显式说明了测试策略。
   未通过的项**不阻断创建**，仅在 child Issue 上贴 `beaver/missing-test / beaver/needs-split / beaver/missing-context` 三类 Beaver agent 元数据标签（独立于被淘汰的 taxonomy）。
6. **批量落库**（每个 child 重复以下顺序，先链接再加入 Project，避免父卡 sub-issue 不一致）：
   a. 在父 Issue 所在 repo 创建 child Issue；
   b. 通过 Sub-Issues API 链接到父 Issue；
   c. 把 child 加入 Project #14；
   d. 写 Project V2 字段：`Type`（Task→SubTask 拆得 `SubTask` + `Size=S`）、`Level`、`Size`、`Status=Triage`；**不**写 `Iteration`（留给后续 `/beaver-tracker` 或 `/beaver-create` 流程显式处理）。
   e. 若该 child 在第 5 步审计未通过，按结果贴对应 `beaver/*` 标签。
7. **父 Issue 总结**：命令在父 Issue 上发一条评论，列出所有 child Issue 编号 + 审计结果摘要，便于 Reviewer 快速对照设计与拆解。
8. **下一步指引**：终端提示「N 个 child 处于 `Status=Triage`；将这些 child 加入 Iteration 后由系统迁移转为 `Ready to Claim`；之后由开发者在 GitHub UI assign 自己并手动切 Status 认领（`/beaver-claim` 已删除，见 §3）」。

**Guardrail**：自动 audit（仅打 Beaver agent 标签，不阻断创建）；前置校验失败即中止。

**期望终态**：父 Issue 在 GitHub Sub-Issues 视图下持有 N 个 child；每个 child 在 Project #14 中具备 `Type / Level / Size / Status=Triage` 字段值；audit 失败的 child 已贴 `beaver/*` 标签；父 Issue 上有一条 audit summary 评论。

#### 6. `/beaver-dev`

**触发场景**：开发者已认领某个 Issue，并准备开始实际编码。命令把"开始写代码"形式化为「在隔离 worktree 中走 TDD 闭环 → 通过两段 code-review → 全量测试通过」三件事的强制串联，避免开发跳过 RED test、跳过 review 直接 push 的常见反模式。

**预期 workflow**：

1. **入参与前置校验**：用户运行命令并给出 `<issue-number>`。命令读取该 Issue 字段并校验：
   - 当前用户为 assignee；
   - Size=S → `Status=In Progress`（claim 阶段已迁移）；
   - Size=L → `Status ∈ {Ready to Develop, In Progress}` 且至少 1 个 sub-issue（G009）。
   不满足即中止。
2. **状态推进**：仅当 Size=L 且 `Status=Ready to Develop` 时，命令把 `Status` 推进到 `In Progress`，标识"实质开发已开始"。Size=S 在 claim 阶段已经 `In Progress`，本步跳过。
3. **worktree 创建**：命令在仓库根目录的 worktree 池中创建一个新分支 `<type>/<n>-<short_desc>`（`<type>` 来自 Issue Type，`<short_desc>` 由 Issue 标题派生为 kebab-case），并 `git worktree add` 到隔离目录。所有后续编码操作发生在此 worktree 内，主工作树不受污染。
4. **TDD 子代理调度**：在 worktree 内命令调度一个 TDD subagent，强制执行 Red-Green-Refactor 循环：
   - **Red**：subagent 先写一个会失败的测试，必须先看到失败输出才能进入 Green；
   - **Green**：写最少代码让该测试通过；
   - **Refactor**：清理重复 / 命名 / 结构，每一次 refactor 后重跑测试。
   TDD Iron Law：任何跳过 Red 直接写实现的尝试都会被 subagent 拒绝。
5. **debugging 兜底**：测试或运行期出现失败时，命令调度 `systematic-debugging` subagent 接管——先复现、再二分定位、再写一个能复现该 bug 的测试、再修。debugging 完成后回到第 4 步继续 TDD。
6. **两段 code-review**：开发者声明"功能完成"后，命令依次调度两个 review subagent：
   - **spec-compliance**：对照原 Issue 的验收标准 + Design Doc（若 Size=L），逐条勾选；
   - **code-quality**：对照仓库 lint/format/约定，标出可改进点。
   两段 review 各产出一份反馈清单；命令把反馈展示给用户，用户可逐项接受或拒绝并说明理由。被接受的反馈进入下一轮编码循环。
7. **Verification Iron Law**：命令在结束前强制运行项目的全量测试套件（项目根目录下的 `make test` / `pnpm test` 等），并要求 0 failures；任何 1 例失败都阻止命令进入"完成"分支，必须回到第 5 步 debugging。
8. **下一步指引**：所有循环结束后，命令打印 worktree 路径、当前分支名、新增 commits 数量与下一步「`/beaver-pr <n>` 进入 PR 阶段」。`Status` 保持 `In Progress`，直到 PR 合并后由系统迁移转 `Done`。

**Guardrail**：G009（Size=L 必须有 sub-issue）、TDD Iron Law、Verification Iron Law。

**期望终态**：worktree 内含通过的实现 + 测试 + 一组 conventional commits；项目全量测试 0 failures；Issue Status 为 `In Progress`；终端给出 `/beaver-pr <n>` 提示。

#### 7. `/beaver-pr`

**触发场景**：开发者在 worktree 内 `/beaver-dev` 走完后，需要把 commits 推到远端、开 Draft PR、走 Review。命令把"开 PR"形式化为：自动推断 Issue 编号、强制使用 conventional commit、强制以 Draft 起步、提供 4 个 finishing-options 让作者明确决定 PR 后续走向。

**预期 workflow**：

1. **Issue 编号推断**：用户运行命令，可显式给 `[issue-number]`，也可省略；省略时命令依次尝试：(a) 当前分支名 `<type>/<n>-<desc>` 中提取 `<n>`；(b) 最近一条 commit message 中匹配 `#<n>`；两者都失败即提示用户显式输入。
2. **本地状态校验**：命令检查当前 worktree 是否存在未提交修改或未推送 commit；若什么都没有则提示「无变更可提交」并退出。`gh auth status` 校验通过。
3. **commit 与 push**：
   a. 若有未提交修改，命令把它们组织为一次或多次 conventional commit（`<type>(<scope>): <subject>`），按文件变更范围自动建议但仍由用户确认 commit message。
   b. 若当前分支无远端 tracking，命令把分支 push 到 `origin` 并设置 upstream。
4. **PR 起草**：命令以 Draft 状态打开 PR——
   - 标题取最近一条非 fixup commit 的 subject；
   - body 含 Summary（自动汇总所有 commits 的 subject + 关键 file）+ Test Plan（取自 Issue 验收标准 + 本次 worktree 内执行的测试命令清单）+ `Closes #<n>`，便于 PR 合并时由 GitHub 自动 close 关联 Issue。
5. **Beaver agent 审计**：
   - **G004**：检查本次 PR 涉及的 commits 中是否有 test 文件改动；无则在 Issue 上贴 `beaver/missing-test`（Beaver agent 元数据标签，独立于淘汰 taxonomy）。
   - **G006**：检查 Issue 自身是否已具备 `Type / Size` 字段；缺失则贴 `beaver/missing-context`。
   两类审计均仅警告、不阻断 PR 创建。
6. **finishing-options**：命令向用户展示 4 个互斥选项，等待选择：
   - **保持 Draft**（默认）：PR 留在 Draft，作者自审后再手动转 Open；
   - **mark-ready**：自审 OK，命令立刻 `gh pr ready` 转 Open，触发系统按 CODEOWNERS 自动指派 Reviewer；
   - **保留分支**：保留 worktree 与分支，方便后续追加 commits；
   - **discard**：放弃本次工作，命令依次 `gh pr close --delete-branch` 并 `git worktree remove`，整体回到干净状态（仅在用户二次确认后执行，避免误删）。
7. **下一步指引**：命令打印 PR URL、所选 finishing-option、贴上的 audit 标签清单、以及合并预期：「PR 合并后 Issue 通过 `Closes #<n>` 自动 close；系统迁移把 Project Status 转为 `Done`（系统迁移本次 out-of-scope，可能仍需人工触发）；Size=L Task 须等所有 sub-issue 关闭后由系统汇总到 `Done`」。

**写字段**：本命令本身**不修改** Project V2 `Status` 字段，仅依赖 PR 合并触发的系统迁移。

**Guardrail**：G004、G006（仅警告）；finishing-options 中的 `discard` 需二次确认。

**期望终态**：远端存在 Draft（或 Open）PR；PR body 含 `Closes #<n>`；Issue 上按需贴有 `beaver/missing-test` / `beaver/missing-context`；终端给出 PR URL 与下一步说明；除 PR 自动 close 触发的系统迁移外，本命令未改任何 Project V2 字段。

#### 8. `/beaver-focus`

**触发场景**：开发者上班开机、午休回来、或卡在某个任务时希望快速看到「我此刻应该做什么、什么在挡我、什么 DDL 临近」。命令是个人 dashboard 的入口，**严格只读**，不修改任何远端状态——这一点是设计上的硬约束，避免"看一眼"误触发状态迁移。

**预期 workflow**：

1. **入参**：无。命令读取当前 `gh` 用户身份。
2. **数据汇聚**：命令并行查询 Project #14 上多个切片，按字段过滤生成一份个人视图：
   - 当前用户为 assignee 且 `Status=In Progress` 的所有 Issue；
   - 当前用户的 `Type=Bug ∧ Priority=P0` 的所有 Issue（不论 Status，用于置顶警示）；
   - 当前用户在所有 repo 上待 review 的 PR（通过 `gh search prs --review-requested=@me`）；
   - 当前 Iteration 内 `Status=Ready to Develop` 与 `Ready to Claim` 的可承接项；
   - `Status=Blocked` 且 assignee 为当前用户的项；
   - 当前 Iteration 的 `endDate ≤ today + 48h` 的所有 assigned 项。
3. **优先级排序与高亮**：
   - P0 Blocker 在分组中按 issue 持续时间排序，开放超过 24h 的项追加 ⚠️ 警示；
   - In Progress 按最后 commit / 评论时间倒序，便于回到最近的工作流；
   - DDL Warning 按 `endDate` 升序。
4. **LLM 推荐**：命令调度一个轻量 LLM 调用，把上述清单作为输入，让模型综合「DDL 紧急度 / blocker 严重度 / 当前 In Progress 进度」三维，输出 `Today's Top 3 Priorities` —— 三条具体可执行的下一步建议（不只是 issue 列表，而是「先解掉 #234 的阻塞 / 把 #145 的 RED test 补上 / Review @colleague 的 #178」这种粒度）。
5. **终端渲染**：命令把所有信息以 markdown 形式打印为一份分组 dashboard：
   - **P0 Blockers**（含 ⚠️ 持续时间提示）
   - **In Progress**
   - **Bugs**
   - **Ready to Develop**
   - **Ready to Claim**
   - **Awaiting My Review**
   - **My Blockers**（被自己挡住别人的项）
   - **DDL Warnings**（Iteration 临近结束）
   - **Today's Top 3 Priorities**（LLM 推荐）
6. **零写入保证**：整个流程结束后，命令断言自身未发起任何 mutation 性质的 GraphQL / REST 调用——这一点在测试中由"调用追踪 + grep 仅命中 query/get"来验证，是 `/beaver-focus` 与其他命令的根本区别。

**写字段**：无。

**Guardrail**：无（只读，不触发任何 mutation）。

**期望终态**：当前用户在终端获得一份按优先级 + DDL 排序的 markdown 工作视图；不修改任何远端状态、不创建任何 Issue / PR / 评论 / label。

#### 9. `/beaver-setup`

**触发场景**：(a) 一次性 bootstrap：第一次接入 Beaver 的项目需要在 Project #14 上把字段 / Issue Type / 标签 / `beaver-config` 全部建好；(b) 每次本 RFC 或后续 RFC 改动了字段定义后跑一次以补差。命令必须**幂等**——重跑只补缺失项，不会覆盖人工调整。

**预期 workflow**：

1. **入参**：无（所有 `org` / `project number` 硬编码为 `primatrix` + Project #14，避免被误指向其他 project）。
2. **scope 自检**：命令检查 `gh auth status` 与 token scope，需含 `project` 与 `admin:org`；任一缺失立即提示 `gh auth refresh -h github.com -s project,admin:org` 并退出，避免后续中途失败留下半完成状态。
3. **HARD-GATE 用户审批**：命令把"将要做的全部变更"作为一份清单展示给用户（要新建 / 补齐哪些字段、要创建哪些 Issue Type、要写哪些 README 段），等待用户显式输入 `yes`。任何 `no` / Ctrl-C 都使命令零副作用退出。
4. **Project V2 字段 ensure**（按字段逐个 ensure 存在 + 选项匹配，不一致则补差）：
   - `Level`：SINGLE_SELECT，选项 `Task / SubTask / Bug`；
   - `Status`：SINGLE_SELECT，**全量替换**为 wiki spec 的 7 个值（`Triage / Ready to Claim / Design Pending / Ready to Develop / In Progress / Blocked / Done`）——若现存选项数量或名称与之不符，命令打印 diff 并要求用户二次确认后替换；
   - `Size`：SINGLE_SELECT，选项 `XS / S / M / L / XL`（**新增字段**）；
   - `Progress`：NUMBER；
   - `Iteration`：ITERATION，自当前月起填到当年 12 月。
5. **原生 Issue Type ensure**（在组织级别）：
   - 通过 `gh api /orgs/primatrix/issue-types`（GET）列出已存在 Type；
   - 缺失的 `Bug / Task / SubTask` 通过 POST 补齐；
   - **不再创建 `Milestone`**（已弃用）；若历史已存在 `Milestone`，命令打印「检测到弃用 Type，建议人工迁移已存在的实例后再删」但不主动删，避免破坏历史数据。
6. **`beaver-config` 写入**：在 `primatrix/projects` README 中维护一个 YAML 块，含本 RFC 约定的 `issueRepo / projectNumber / fieldNames` 等配置，供其他命令读取。已存在则按 key 合并，不已有覆盖。
7. **执行总结**：命令打印一份 setup summary：本次新增 / 已存在 / 跳过 的字段、Type 数量；并附一个针对成功指标 2 的自检 grep 结果（确保源码侧无 `Milestone` Issue Type 引用）。

**Guardrail**：scope 缺失时立即退出；Status 选项数量不为 7 时给出 diff 警告并要求二次确认；任何写操作前必须经过第 3 步 HARD-GATE。命令在源码侧不再调用 `Milestone` Issue Type 的创建路径（成功指标 2 的 grep 断言）。

**期望终态**：Project #14 上 `Level / Status (7 项) / Size (5 项) / Progress / Iteration` 五个字段形态正确；组织级 Issue Type 含 `Bug / Task / SubTask` 三个值；`beaver-config` 在 README 中可被其他命令读取；终端给出 setup summary。

#### 10. `/beaver-fix`

**触发场景**：开发者在 `/beaver-pr` 开出 Draft 或 Open PR 后，收到了 Reviewer 的 review comments；希望由命令统一收集所有 open comments、给出修复建议、经用户 QA 确认后自动应用修改，并在所有 fix 完成后批量 resolve 所有 comments 并推送变更。

**入参**：`<pr-number>`（必填，且必须是当前用户自己发起的 PR）。

**预期 workflow**：

1. **入参与权限校验**：命令读取 `<pr-number>`，通过 `gh pr view` 确认 (a) 该 PR 确实存在于当前 repo；(b) PR author 与当前 `gh` 用户身份一致——若不一致立即中止并提示「只能对自己发起的 PR 运行 /beaver-fix」。
2. **评论收集**：命令通过 `gh api` 拉取该 PR 的所有 **review comments**（line-level）与 **issue comments**（PR-level 评论）；过滤掉 `state=RESOLVED` 的 review threads，仅保留 open / unresolved 的条目；若无 open comments，打印「无待处理评论」并退出。
3. **逐条修复建议**：对每个 open comment，命令：
   a. 在终端渲染评论内容（路径 + 行号 + 评论原文），并读取对应代码片段作为上下文；
   b. 生成一条具体的修复建议（diff 或自然语言描述），标注修复依据（遵循原评论意图 / 与 CLAUDE.md 约定一致 / 与 spec 一致 / 存疑需用户裁断）；
   c. 向用户发问（每次一条）：`[接受修复] / [修改建议] / [跳过，不修复] / [标为已知，仅 resolve comment]`，等待用户选择后再处理下一条；
   d. 若用户选择「接受修复」，命令立即把对应代码变更写入文件（不提前批量写，避免相互冲突）；
   e. 若用户选择「修改建议」，命令展示当前建议并允许用户补充意图，重新生成后再回到第 3.c 步。
4. **全局 QA 确认（HARD-GATE）**：所有 open comments 逐条处理完毕后，命令把本轮「已修复 / 跳过 / 仅 resolve」的汇总展示给用户，并等待用户输入 `yes` 确认继续；`no` / Ctrl-C 回滚所有文件修改并退出（不推送、不 resolve）。
5. **resolve comments**：用户确认后，命令通过 `gh api graphql` 对步骤 3 中「接受修复」与「仅 resolve」的每一条 review thread 调用 `resolveReviewThread`（mutation），批量标记为 resolved；「跳过」的 thread 保持 open 不触碰。
6. **commit 与 push**：命令把本轮所有文件修改组成一次 conventional commit（`fix(<scope>): address review comments`），commit message body 中以无序列表列出各条 comment 的简要描述与处置方式（修复 / resolve），push 到 PR 的远端分支。若无任何文件修改（所有 open comments 均选「仅 resolve」或「跳过」），跳过 commit 步骤，仅做 resolve。
7. **下一步指引**：命令打印 PR URL、本轮 resolved 数、剩余 open 数、push commit SHA（如有），并提示：
   - 若剩余 open 数 > 0：「仍有 N 条 open comment，可再次运行 `/beaver-fix <pr-number>` 处理」；
   - 若剩余 open 数 = 0 且 PR 仍为 Draft：「所有评论已 resolved；如已完成自审，可运行 `gh pr ready <pr-number>` 转 Open」；
   - 若剩余 open 数 = 0 且 PR 已为 Open：「所有评论已 resolved；等待 Reviewer 重新审核或批准」。

**写字段**：本命令**不修改** Project V2 字段。Push 到 PR 分支后由 GitHub 自动重新请求 review（若 CODEOWNERS 配置了 `dismiss stale reviews`），不需命令额外操作。

**Guardrail**：PR author 一致性检查（步骤 1，强制）；HARD-GATE（步骤 4，强制）；`discard` 路径（HARD-GATE 拒绝）回滚文件修改不推送；`resolveReviewThread` 仅对「接受修复」或「仅 resolve」的 thread 执行，「跳过」的不触碰。

**期望终态**：PR 对应分支上存在一次新 commit（如有文件修改）；「接受修复」与「仅 resolve」的 review threads 均处于 resolved 状态；「跳过」的 threads 保持 open；终端给出剩余 open comment 数与下一步指引；Project V2 字段未被修改。

### 分阶段重构

重构分两个 Phase。Phase A 为 3 个有序 SubTask，必须全部合并后 Phase B 才能开始。

#### Phase A — Engine + 基础设施（3 SubTasks）

| # | SubTask | 依赖 | 交付物 |
|---|---|---|---|
| A1 | `scripts/beaver-lib.sh` 初始库 + `--self-test` 子命令 | — | 新文件，含下列 public API；`bash scripts/beaver-lib.sh self-test` 通过 |
| A2 | `beaver-engine` 重写（§1 删除标签行；§2 状态机改为字段语义；§3 guardrail 重写；§4 Label Ops 替换为 Field Ops） | A1 | 重写后的 SKILL.md |
| A3 | `beaver-setup` 迁移：通过 `gh api /orgs/primatrix/issue-types`（POST）确保 `Bug / Task / SubTask` 原生 Issue Type 定义存在；通过 `beaver-lib.sh::set_type` 完成所有后续实例赋值；创建 Project V2 单选 `Size` 字段（`XS / S / M / L / XL`） | A1 | `beaver-setup.md` + `beaver-setup.sh` 更新 |

A1 / A2 / A3 可在独立 PR 中提交，但 Phase B 任一 SubTask 启动前必须 **全部三者合并**。这符合 Issue #111 验收标准 2「对每个 Diff 点创建 Sub Task」。

#### Phase A.2 中的 guardrail 改写

- G001 / G002 / G006 / G009：从读 `status/*` / `type/*` / `size/*` 标签改为通过 `beaver-lib.sh` 读 Project V2 字段 + 原生 Issue Type。
- G008（Bug 强制 size/S）：**删除**——作者将同步修订 wiki spec 放宽该约束。
- G011（新增）：在 `/beaver-create` 处理 `Type=Bug` 时强制设置 Iteration，由下文算法解析。算法返回 null 时 G011 失败并提示用户运行 `/beaver-tracker <repo>`。

#### `latest_iteration_for_repo <repo>` 解析算法（G011 使用）

`<repo>` 参数的角色：Iteration 候选集本身是 Project #14 全局的，但**挂载动作**有 repo 维度——`/beaver-create` 把 Bug Issue 创建在某个 repo 后，需要先把该 issue 加入 Project #14、再把解析出的 Iteration 写到该 issue 的 project item 上。`<repo>` 在算法层用于：(a) 失败时拼出 `/beaver-tracker <repo>` 错误提示；(b) 与上游 `add-to-project` 步骤的 caller 上下文对齐。算法本体（下列 5 步）不读 `<repo>`，仅返回当前/未来 Iteration 候选；caller 负责把结果写到属于该 repo 的 issue 上。

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
| `set_type <issue_number> <type>` | **原生 GitHub Issue Type**（给单个 issue 赋类型） | `updateIssueIssueType`（GraphQL，需 header `GraphQL-Features: issue_types`，issue_types 在 public preview）；要求 `admin:org` scope（与现有 `beaver-setup` 一致） |
| `get_type <issue_number>` | 原生 Issue Type | `repository.issue.issueType` |
| `get_iteration <issue_number>` | Project V2 #14 Iteration 字段 | `projectV2Item.fieldValueByName(name: "Iteration")` |
| `set_iteration <issue_number> <iteration_title>` | Project V2 #14 Iteration 字段 | `updateProjectV2ItemFieldValue`（iterationId） |
| `latest_iteration_for_repo <repo>` | Project V2 #14 Iteration 字段 | 拉取所有 Iteration entry，应用 G011 算法 |

`set_type` 在调用界面上与 `set_status` / `set_size` 同形，但底层走 `updateIssueIssueType` 而非 `updateProjectV2ItemFieldValue`——Type 是仓库/组织级原生属性，不是 Project V2 字段。这种非对称性被 `beaver-lib.sh` 屏蔽在调用方之外。

`set_type` 仅做**实例赋值**（给单个 issue 选一个已存在的 Type）。**创建组织级 Type 定义**走另一条 REST 接口 `POST /orgs/{org}/issue-types`，不属于 `beaver-lib.sh` 的 public API；A.3 中创建 Type 的逻辑保留在 `beaver-setup.sh` 内联的 `gh api /orgs/primatrix/issue-types` 调用中（与现有实现一致），仅 Type 的**赋值**收敛到 `set_type`。

A3 重写 `beaver-setup` 中的 Issue Type **赋值**逻辑（如初始化时给样例 issue 设 Type）统一通过 `beaver-lib.sh::set_type`；Type **定义**的创建仍走 `gh api /orgs/<org>/issue-types`，因为它是组织级一次性元数据操作，不需要库屏蔽。

#### Phase B — 命令逐个迁移（8 SubTasks，并行）

每个命令一个 SubTask，作用对象：`beaver-create / beaver-design / beaver-decompose / beaver-dev / beaver-pr / beaver-tracker / beaver-focus`（`beaver-setup` 已在 A3 处理；`beaver-claim` 已删除，见 §3）。

每个 SubTask 工作量：

1. Frontmatter `allowed-tools`：移除 `gh label:*`，确保 `gh api graphql:*`。
2. 命令正文：替换所有 `status/*` / `type/*` / `size/S|L` 文本为字段 / 原生 Type 语义。
3. `scripts/<command>.sh`：`source scripts/beaver-lib.sh`，将 `gh api repos/.../labels` 调用替换为库函数调用。
4. 沙盒 smoke：在 `primatrix/projects` 上跑一遍命令的 lifecycle 步骤，PR 描述记录会话。

`beaver-tracker` 的 SubTask 额外承担 spec 中「每个 Project 在当前 Iteration 内维护一个 tracker issue」的语义实现：

- tracker issue 的 sub-issue 列表 = `{Project V2 #14 内 ∧ Iteration=当前周期 ∧ repo 归属=<repo> ∧ Type ∈ {Task, Bug}}`，每次命令运行需做一次差集同步（add 缺失 sub-issue，remove 已不属于当前 Iteration 的 sub-issue）。
- tracker issue 自身的 Iteration 字段通过 `beaver-lib.sh::set_iteration` 写入，标识其归属周期。
- 上述差集查询走 `beaver-lib.sh::get_iteration` + GraphQL `projectV2.items` 过滤，不依赖 `status/*` 标签。

### 接口与数据流

- **读路径**：command-script → `beaver-lib.sh` `get_*` → GraphQL `projectV2Item.fieldValueByName` → typed value（单选 option name / iteration title）。
- **写路径**：command-script → `beaver-lib.sh` `set_*` → GraphQL `updateProjectV2ItemFieldValue` 或 `updateIssueIssueType`。每字段原子。
- **跨命令交接**：形态不变，每个命令把 Status 切到下一个合法值（通过 `set_status`），并打印下一步提示（如 `/beaver-create` 之后输出「请在 GitHub UI assign 自己后手动将 Status 切换」）。

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
- **A3 SubTask**：在 `primatrix/projects` 上重跑 `beaver-setup`，断言：(a) `gh project field-list primatrix 14 --format json | jq '.fields[] | select(.name=="Size") | .options[].name'` 返回 `XS/S/M/L/XL`；(b) `gh api /orgs/primatrix/issue-types` 返回包含 `Bug / Task / SubTask`。
- **Phase B SubTasks**：每命令 PR 含沙盒 smoke 表（Step | Command | Expected | Observed）。
- **父 Issue 关闭前的最终验证**：见下文「成功指标」全部 4 项。

### 备选方案

**Alt A — 反向：保留当前 Beaver 标签 taxonomy，修订 wiki 适配 Beaver。** 工程量最低（Phase A 几乎为零），仅做内部一致性整理。**否决**：Issue #111 明确把 wiki 列为对齐目标；反转契约方向只会让团队的参考文档继续与现实不符，仅是把矛盾换边。Wiki 是先行的团队约定，命令必须追上，不应改写约定。

**Alt B — 混合 taxonomy（Status 用字段；Type / Size 仍用标签）。** Status 是 lifecycle 关键枚举因此值得字段化；Type/Size 多为只读筛选条件，标签足够。迁移成本最低。**否决**：每个命令需同时学两套集成模式（`gh api .../labels` for Type/Size + `gh api graphql` for Status）；G001/G002/G006/G009 必须混合读取标签和字段，失败模式增加；`beaver-lib.sh` 的去重价值大幅缩水。

**Alt D — 把系统侧自动迁移纳入本次 Goal。** 一次性把 Iteration-add → Design Pending、Design-PR-merge → Ready to Develop、SubTask close → parent Done 三个迁移用 GitHub Actions 实现。**否决**：需要 (a) 每个 trigger 一个 workflow；(b) PAT/GITHUB_TOKEN scope 评审；(c) 半应用迁移的回滚策略；(d) 单独的沙盒 project 用于测试，避免污染 #14。SubTask 数量约翻倍。Section 1 已明确放在范围外；本次重构需保持自包含、可评审的体量。

## 影响范围

- **代码**：`plugins/beaver/` 下 9 个命令 + 1 个 engine SKILL + 9 个 scripts，新增 1 个共享 lib（`scripts/beaver-lib.sh`）。
- **GitHub Projects V2 #14**：新增 Project 单选字段 `Size`（`XS/S/M/L/XL`）。
- **GitHub 组织级 Issue Types**：保留 `Bug / Task / SubTask` 三个值；`Goal`、`Feature`、`Milestone` 标记为弃用（不再被任何命令引用，但历史已创建的实例不主动删除）。
- **`primatrix/projects` 仓库标签**：`status/*` / `type/*` / `size/*` 标签定义本次保留不删除（仅停止源代码引用）。
- **使用方**：所有用 Beaver 命令做项目管理流转的开发者（迁移期内既有 Issue 的字段需用 `beaver-setup` 重跑或手工补齐 Status/Size 字段）。

### 成功指标

1. **End-to-end smoke（Draft-PR 终态）**：一个新建 size/M Issue 走完 `create → tracker → claim → design → decompose → dev → pr` 后落在 `Status=In Progress` + 一个 Draft PR + Project V2 字段值与 spec 表对齐，过程中无任何手动标签 / 字段编辑、无 Beaver script 之外的 `gh api` 调用。Pass = 录制的 session 日志。
2. **字段形态**：`gh project field-list primatrix 14 --format json | jq '.fields[] | select(.name=="Status") | .options[].name'` 精确返回 7 个 wiki 值；同样查询 `Size` 返回 `XS / S / M / L / XL`；`gh api /orgs/primatrix/issue-types` 返回包含 3 个 wiki Type 值（`Bug / Task / SubTask`），且 `git grep -nE "issue-type.*Milestone|--issue-type ['\"]?Milestone" plugins/beaver/` 在源码侧零命中。
3. **源码侧标签清理**：`git grep -nE "(gh label (add|create|delete|remove))|gh api[^|]*labels.*(status/|type/|size/)" plugins/beaver/` 在命令正文 / scripts / engine 上返回零命中。仓库侧标签定义保留不动。
4. **Bug fast-path smoke**：新建 `Bug` Issue 且 `Priority = P0`，单次 `/beaver-create` 完成后即落在 `Status = In Progress` 且按 G011 算法解析的 Iteration 已分配。
5. **Project tracker 一致性**：在已存在若干 Iteration=当前周期 Task 的 repo 上跑 `/beaver-tracker <repo>`，运行后 (a) `primatrix/projects` 中存在标 `tracker/<YYYY-MM>` + `tracker/<repo>` 的 issue；(b) 该 issue 的 sub-issue 集合等于「Project V2 #14 内 Iteration=当前周期 ∧ repo 归属=`<repo>` ∧ Type ∈ {Task, Bug}」的 Task 集合（差集为空）；(c) 该 tracker issue 自身的 Iteration 字段已设为当前周期。

## 实施计划

| 阶段 | SubTask | 依赖 | 交付 |
|---|---|---|---|
| Phase A.1 | `scripts/beaver-lib.sh` 初始库 + `--self-test` | — | 新文件 + 沙盒 round-trip 通过 |
| Phase A.2 | `beaver-engine` 重写 | A.1 | 字段化 §1–§4，guardrail G008 删除 / G011 新增 |
| Phase A.3 | `beaver-setup` 迁移 | A.1 | `Bug / Task / SubTask` Issue Type + Project V2 `Size` 字段 |
| Phase B (×8) | 每命令一个 SubTask（`beaver-{create,claim,design,decompose,dev,pr,tracker,focus}`） | Phase A 全部合并 | 命令 + script 字段化 + 沙盒 smoke |
| 收尾 | 父 Issue #111 关闭 | Phase B 全部合并 | 5 项成功指标全部通过 |

Phase A 的三个 SubTask 之间可并行评审（仅 A.2 / A.3 实现上引用 A.1），但合并顺序仍为 A.1 → 然后 A.2 / A.3。Phase B 八个 SubTask 完全并行。

## 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 既有 Issue 的 Project V2 字段未填值 | 迁移完成后旧 Issue 在 dashboard 上显示为空 Status / Size | A.3 可附带一次性扫描脚本，把仍带 `status/*` / `size/*` 标签的 Issue 写入对应字段 |
| `set_type` 需要 `admin:org` scope | 个别 contributor 的 token 缺少 scope | 命令在 `set_type` 失败时打印明确的 `gh auth refresh -h github.com -s admin:org` 提示 |
| 系统侧自动迁移仍未实现 | 用户 assign 后需手动在 Project #14 切换 Status，与 wiki 描述的 "system behavior" 不符 | 在 `beaver-engine` §2 注明哪些迁移目前仍需手工触发；后续 Goal「Beaver 自动迁移基础设施」跟进 |
| Bug 通道 wiki spec 修订未先于本次合并 | G011 已上线但 wiki 仍说 Bug 强制 `size/S` | 已解除：wiki PR #104（Bug 不强制 Size，自动加入 Iteration）与 PR #106（Roadmap 改用 Iteration 字段 + per-Project tracker）均已合并到 main，本 RFC 已基于该状态对齐。 |
| Phase B 八个 SubTask 并行触发对 `primatrix/projects` Project #14 的写竞争 | 沙盒 smoke 之间相互污染 | 每个 SubTask 创建独立的 `[smoke] <command>` Issue，跑完即 close；review 后由作者手动归档 |

<!-- provenance
- "Beaver v3.2.0 9 命令 + 1 engine + 9 scripts" ← Discovery D2 (Glob plugins/beaver/**/*) + plugin.json
- "Project V2 #14 已有字段 Level/Status/Progress/Iteration" ← Discovery D3 (gh project view 14 --jq .readme)
- "已有原生 Issue Type Task/SubTask/Milestone" ← beaver-setup.md §Create Issue Types
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
- "Type 集合精简为 3 个值（Bug / Task / SubTask，去掉 Goal、Feature、Milestone）" ← wiki PR #106 main (latest project-management.md line 66)
- "Metric 5 (tracker 一致性) + Phase B beaver-tracker 额外职责" ← 对齐 wiki PR #106 spec
- "命令规约 9 条目改为完整 workflow 叙事（触发场景 / 步骤化用户系统交互 / 期望终态）而非契约式字段表" ← 对齐 PR #105 review feedback：描述期望 workflow，不假设当前实现
- "/beaver-tracker 与 /beaver-setup 中保留 tracker-* / Control-By-Beaver / beaver-* 等仓库级标签" ← 这些是 Beaver 自身元数据，不属于 Status/Type/Size taxonomy，本次重构不淘汰；Metric 3 的 grep 仅断言 `status/|type/|size/` 三类前缀
- "所有命令的 QA 与终端输出均使用中文" ← wiki PR #105 声明（design/111-beaver-commands-realignment）
-->
