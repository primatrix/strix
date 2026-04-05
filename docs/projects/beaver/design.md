# Beaver 产品设计：GitHub 智能通知与跟进 (Phase 1)

## 1. 设计目标

针对现代软件开发中 GitHub 信息过载、上下文同步碎片化等问题，Beaver (Phase 1) 致力于通过 AI 代理能力构建一套**感知-推理-行动 (Sense-Reason-Act)** 的项目跟进体系，将 GitHub 的 Issue、PR 和 Project 动态转化为具有业务深度的智能通知。

## 2. 信息载体与集成方案

Beaver 所有的工作完全基于 GitHub 展开，其核心数据源包括：

- **GitHub Project (ProjectsV2)**：作为任务看板和状态流转的真理之源。
- **GitHub Issues**：作为功能需求描述、Bug 反馈及团队讨论的载体。
- **GitHub Pull Requests (PRs)**：作为代码变更、代码评审（Reviews）和 CI 状态的汇集点。

## 3. 核心能力架构：SRA 框架

### 3.1 感知 (Sensing)

Beaver 实时监听并捕获 GitHub 上的事件流：

- **事件监听**：通过 GitHub Webhooks 接收 `issues`, `pull_request`, `project_card` 等事件。
- **定期轮询**：利用 GitHub GraphQL/REST API 批量同步 Project 看板状态及 PR 的评论详情。
- **上下文拉取**：当检测到重要事件时，自动拉取相关的 PR Diff、Issue 讨论历史及关联的 Milestone。

### 3.2 推理 (Reasoning)

AI 代理对感知到的原始数据进行多维度语义分析：

- **重要性评估**：判断该事件是否位于核心路径上（如：主分支 PR、带有 `bug/blocker` 标签的任务）。
- **风险识别**：识别潜在风险（如：PR 长时间无回复、CI 连续失败、Project 任务延期）。
- **受众匹配**：根据事件性质自动匹配最适合的通知对象（对应负责人）。

### 3.3 行动 (Acting)

将推理结果转化为可执行的智能触达，区分不同频次的报告深度：

#### A. 实时告警 (Real-time Alerts)

- **触发条件**：新增 `p0/blocker` 标签、CI 核心流程失败、PR 冲突等。
- **形式**：即时推送高信号卡片，包含风险预警及直接上下文链接。

#### B. 晨间 Focus 提醒 (Morning Focus) - **新增**

- **触发时间**：工作日 09:00 (本地时间)。
- **推送策略**：
  - **面向开发者 (Personalized)**：通过私信推送“今日待办清单”，包含其名下所有处于 `status/in-progress` 或 `status/ready-to-develop` 的 Issue/PR。
  - **面向管理者 (Leadership)**：在管理频道推送“未分配任务提醒”，识别所有 `type/triage` 或已过初筛但无 `assignee` 的任务，协助管理者进行资源调度。
  - **任务优先级建议**：利用 LLM 根据 `p/` 标签和 DDL 自动排序，标注“今日最值得关注”的 3 个核心任务。

#### C. 智能日报 (Daily Progress Sync)

...

#### C. 智能周报 (Weekly Deep Analysis)

- **定位**：项目健康的“诊断书”，侧重**深度分析与风险预警**。
- **核心内容**：
  - **里程碑达成度**：分析本周工作对整体 Milestone 的贡献百分比及进度预测。
  - **研发效能分析**：
    - **吞吐量 (Velocity)**：本周完成的 `size/L` 与 `size/S` 任务分布。
    - **响应时长**：PR 从创建到合并的平均停留时长（Cycle Time）分析。
  - **缺陷与质量趋势**：
    - **存量 Bug 分析**：按优先级统计 `type/bug` 的堆积情况。
    - **代码质量回溯**：汇总本周 SonarQube 扫描出的高频问题或遗留债务。
  - **深度风险诊断**：
    - **长尾任务**：识别停留在某个状态超过 3 天的任务，分析潜在根因。
    - **资源瓶颈**：识别是否存在某位成员承载过多 `size/L` 任务导致的进度风险。
  - **下周展望**：根据当前 `status/ready-to-develop` 的储备情况，自动建议下周计划。

#### D. 流程守门员 (Workflow Gatekeeper) - **核心增强**

Beaver 作为 GitHub App/Bot，具备主动拦截与合规性校验能力：

- **状态流转校验 (Status Guard)**：
  - **规则限制**：禁止不符合逻辑的标签跳转（例如：未经过 `status/review-needed` 直接标记为 `status/done`）。
  - **发表合规评论/通知而不执行回滚**：当检测到非法操作时，Beaver 将发表合规评论/通知而不执行回滚，并在评论区给出合规性指引（如：“此任务需先经过方案评审，请补充 `docs/` 链接”）。
- **合并门禁 (Merge Gate)**：
  - **必需项检查**：PR 合并前必须具备 `size/`、`type/` 标签，且对应的 `status/` 必须处于 `review-needed` 且已获批准。
  - **自动 Block**：利用 GitHub `Commit Status API` 或 `Check Runs`，在 PR 合规性校验未通过时，将合并按钮标记为不可用。
  - **上下文补全**：如果 PR 描述中缺失关联的 Issue 编号，Beaver 会自动发表评论要求开发者关联。

#### E. 任务拆解、审计与跨任务依赖 (Decomposition, Audit & Dependencies) - **核心增强**

Beaver 不代劳拆分，但作为“质量审计员”确保拆解的科学性，并监控网络化的任务依赖：

- **拆解完整性审计 (Split Review)**：当 `size/L` 任务挂载子任务列表时，LLM 介入评估：
  - **覆盖度**：子任务是否覆盖了父任务 PRD/RFC 中的所有核心模块。
  - **原子性**：**强制项**。子任务必须足够小，预期代码变更量应控制在 **200 行 (LOC)** 以内，以确保 Review 质量与速度。
  - **测试定义**：**强制项**。每个子任务描述中必须包含“测试方法 (How to Test)”。若缺失，Beaver 将在评论区 Block 状态流转。
- **跨任务依赖追踪 (Cross-Dependencies)**：**新增**
  - **依赖识别**：自动解析 Issue 描述中的 `Depends on #ID` 语义，建立任务间的无向图依赖网络。
  - **阻塞传播 (Blocked Propagation)**：如果下游任务（如前端接口对接）所依赖的上游任务（如后端 API）被标记为延迟或阻塞，Beaver 自动为下游任务挂载“上游阻塞风险 (Upstream Blocked)”，并在日报的风险链条中明确呈现。
- **测试结论验证 (Test Evidence Verification)**：
  - **置信度检查**：当任务请求标记为 `status/done` 时，Beaver 检查 Issue 评论或关联 PR 中是否包含“测试通过结论”（如：测试报告链接、截图、或具体的 `Test Passed` 日志）。
  - **规模校验 (LOC Guard)**：**合并门禁强制项**。对于 `size/S` 任务关联的 PR，若特定核心目录的代码变更（排除文档、测试用例、自动生成文件等）超过 200 行，Beaver 将自动标记为 `Changes too large` 并 Block 合并，建议进一步拆分。
  - **合规预警**：发现无置信测试结论或规模超限的任务进入 `status/done` 时，Beaver 不会自动回滚状态，而是在评论区要求补充证据或二次拆分，并触发风险通知。

  #### F. 时间与进度管理 (Time & Progress) - **新增**

    Beaver 引入 DDL 机制，将“进度”与“时间”双向锚定：

  - **DDL 来源**：
  - **GitHub Milestone**：继承里程碑的截止日期。
  - **Issue 语义识别**：自动从 Issue 描述或评论中的 `DDL: YYYY-MM-DD` 字段提取。
  - **进度风险梯度 (Risk Grading)**：
  - **Normal**：当前进度符合预期，距离 DDL 充裕。
  - **At Risk**：距离 DDL < 48h 且 `status/in-progress`。
  - **Overdue**：已过 DDL 但状态非 `status/done`。
  - **进度可视化 (Progress vs. Time)**：
  - **燃尽趋势**：在周报中对比“时间消耗率”与“任务完成率”。
  - **延期预测**：利用 LLM 根据历史交付速度，自动计算当前任务是否存在“逻辑延期”风险。

  #### G. 迭代复盘自动化 (Sprint Retrospective) - **新增**

  - **触发条件**：当 GitHub Milestone 状态被标记为 `Closed` 时自动触发。
  - **核心内容**：自动生成该迭代周期的闭环总结报告。
  - **The Good**：表彰本周期内合并最快、零阻塞、代码质量最高的典范任务或 PR。
  - **The Bad**：复盘耗时最长的阻塞链、被打回（被回滚标签）次数最多的任务，深挖其根因。
  - **Action Item**：LLM 基于本次周期的痛点数据，自动提出下一迭代（Sprint）的流程优化与资源调配建议。

  ## 4. 技术栈建议

  - **Compute & Backend**: Cloudflare Workers，利用边缘计算能力接收 Webhooks、处理定时任务与执行核心逻辑。
  - **Async Tasks**: Cloudflare Queues，用于异步可靠地处理 GitHub 大量的事件流。
  - **AI Engine**: 第三方大模型 API（支持 Anthropic Claude 或 Google Gemini 等），通过 REST 或轻量级 SDK 调用，进行深度意图推理与风险分析。
  - **Data Store**:
    - **Cloudflare D1 (SQL)**: 存储持久化的任务快照与项目配置。
    - **Cloudflare KV**: 作为高频的会话和事件缓存，减轻上游 API 压力。
  - **GitHub Integration**: 在 Workers 中通过 `fetch` 调用 GitHub API，或使用兼容 Web Standard 的 `octokit` 核心库。

  ## 5. 成功标准

  - **信噪比提升**：过滤掉 80% 以上不必要的 GitHub 基础活动通知。
  - **响应时效性**：关键阻塞项（Blockers）的平均响应时间缩短 30% 以上。
  - **信息透明度**：团队成员无需手动刷新 GitHub 看板即可掌握真实进度。
