# Beaver: GitHub-Centric Intelligent Notification & Follow-up

Beaver 是一个专注于软件项目**智能通知与跟进**的 AI 代理系统。在当前阶段，其核心目标是消除 GitHub 工作流中的信息噪音，通过感知、推理与行动（Sense-Reason-Act）框架，将碎片化的 GitHub 活动（Projects, Issues, PRs）转化为高信号的项目洞察。

## 核心设计理念

- **GitHub 为中心**：以 GitHub Issue 为任务管理与状态流转的真理之源，Pull Request 为代码交付载体，Project 看板为视图层。
- **高信号驱动**：通过 LLM 过滤无关噪音，仅针对关键风险、阻塞项和里程碑进展进行智能触达。
- **感知-推理-行动 (SRA)**：构建从 GitHub 事件监听、深度语义分析到多渠道精准通知的完整闭环。

## 关键能力

1. **感知 (Sensing)**：实时监听 GitHub Webhooks 和 REST API，捕获任务状态变更、代码审查评论及看板进度。
2. **推理 (Reasoning)**：利用 LLM 分析事件对项目核心路径的影响，评估风险等级与优先级。
3. **行动 (Acting)**：向不同干系人推送定制化通知，并自动生成基于 GitHub 动态的项目日报。

## 文档目录

- [产品设计文档 (Design Document)](./design.md)
- [标签体系规范 (Label System)](./label-system.md)
- [通知机制规范 (Notification System)](./notification-system.md)
- [GitHub 集成规范 (Integration Specs)](./github-integration.md)
- [Cloudflare 基础架构设计 (Infrastructure Design)](./cloudflare-infrastructure.md)
- 实施路径 (Roadmap)
