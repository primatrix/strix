---
title: RFC
---

# RFC（技术决策记录）

RFC (Request for Comments) 用于记录重要的技术决策和架构变更提案。

## 如何提交 RFC

1. 复制 [RFC 模板](./template.md)
2. 创建新文件 `docs/rfc/NNNN-your-title.md`（NNNN 为递增编号）
3. 填写模板内容
4. 提交 PR，指定架构组为 Reviewer

## RFC 列表

| 编号 | 标题 | 状态 | 作者 | 日期 |
|------|------|------|------|------|
| [0001](./0001-example-rfc.md) | 示例 RFC | accepted | team | 2026-03-22 |
| [0002](./0002-ling2-config-extension.md) | Ling2 配置扩展 (PR1) | draft | @Garrybest | 2026-03-26 |
| [0002](./0002-al-model-decoder-integration.md) | Ling2 Decoder 集成 | implemented (Phase 1) | @ClawSeven | 2026-03-31 |
| [0003](./0003-ling2-ckpt-conversion.md) | Ling2 检查点转换扩展 | draft | @aftersnow | 2026-03-31 |
| [0004](./0004-gla-rfc.md) | GLA Refactor | implemented | - | 2026-04-02 |
| [0007](./0007-ling3-config-extension.md) | Ling3 配置扩展 (PR1) | draft | @QiaoTong | 2026-04-15 |
| [0008](./0008-kda-maxtext-implementation.md) | KDA (Kimi Delta Attention) MaxText 实现 | draft | @qiaotonggg | 2026-04-21 |
| [0011](./0011-training-flow-alignment.md) | 训练主流程对齐 (PR10) | draft | @Garrybest | 2026-03-30 |
| [0012](./0012-ling3-model-integration.md) | Ling3 模型集成实现方案 | draft | @Garrybest | 2026-04-15 |
| [0013](./0013-beaver-commands-realignment.md) | Beaver Commands 与项目管理流转预期对齐 | draft | @sii-xinglong | 2026-04-22 |
| [0018](./0018-kimi-linear-model.md) | Kimi-Linear 模型框架实现 | draft | @zhengkezhou1 | 2026-04-22 |
| [0019](./0019-beaver-auto-transitions.md) | Beaver 自动迁移基础设施 | draft | @sii-xinglong | 2026-04-24 |
| [0020](./0020-org-activity-summary-project-v2-progress.md) | 取消日报、新增 per-person commit 汇总与 per-repo Project V2 进展报告 | draft | @sii-xinglong | 2026-04-25 |
| [0021](./0021-beaver-auto-assign-reviewer.md) | Beaver 自动指派 Reviewer——按组负载均衡 | draft | @sii-xinglong | 2026-04-26 |
