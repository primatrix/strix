---
title: "RFC-0001: 建立内部知识库"
status: accepted
author: team
date: 2026-03-22
reviewers:
  - architecture-team
---

# RFC-0001: 建立内部知识库

## 概述

使用 VitePress 构建内部技术知识库，部署在 Cloudflare Pages 上，通过 Cloudflare Access 限制仅组织成员可访问。

## 背景

团队知识分散在各处（Slack、个人笔记、代码注释），缺乏统一的技术文档平台。新成员入职需要花费大量时间寻找和理解既有知识。

## 方案

- **静态站点生成器**：VitePress — 构建快、中文支持好、配置简单
- **托管**：Cloudflare Pages — 自动构建部署、CDN 加速
- **访问控制**：Cloudflare Access + GitHub OAuth — 零代码认证
- **协作**：GitHub PR 工作流 — 代码审查机制保证内容质量

### 备选方案

1. **MkDocs Material** — 插件更丰富但需要 Python 环境，构建较慢
2. **Docusaurus** — 功能全面但配置复杂，更适合开源项目文档

## 影响范围

所有工程团队。每个项目需要指定文档负责人维护各自的文档目录。

## 实施计划

1. 搭建 VitePress 站点框架
2. 配置 Cloudflare Pages 自动部署
3. 配置 Cloudflare Access 认证
4. 各团队迁移/编写文档

## 风险

- 文档维护可能被忽视 → 通过 CODEOWNERS 和 PR Review 机制保证
- 搜索体验可能不如专用工具 → 内置 minisearch 对中文做了分词优化
