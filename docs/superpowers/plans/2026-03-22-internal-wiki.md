# Internal Wiki Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up a VitePress-based internal knowledge base deployed on Cloudflare Pages with GitHub OAuth access control via Cloudflare Access.

**Architecture:** Static VitePress site with Chinese locale, multi-project directory structure, CJK-aware search, Mermaid diagram support. Cloudflare Pages handles hosting and auto-deployment from GitHub. Cloudflare Access provides zero-trust authentication gating via GitHub OAuth restricted to org members.

**Tech Stack:** VitePress 1.x, Node.js 20+, vitepress-plugin-mermaid, markdownlint-cli2, Cloudflare Pages, Cloudflare Access

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `package.json` | Project metadata, dependencies, build scripts |
| `docs/.vitepress/config.ts` | VitePress configuration: locale, nav, sidebar, search, mermaid |
| `docs/index.md` | Homepage with hero section and feature cards |
| `docs/projects/project-alpha/index.md` | Example project overview page |
| `docs/projects/project-alpha/guide/index.md` | Guide section landing |
| `docs/projects/project-alpha/guide/getting-started.md` | Example getting-started guide |
| `docs/projects/project-alpha/reference/index.md` | Reference section landing |
| `docs/projects/project-alpha/reference/api.md` | Example API reference page |
| `docs/projects/project-alpha/troubleshooting/index.md` | Troubleshooting section landing |
| `docs/projects/project-beta/index.md` | Project Beta placeholder overview |
| `docs/projects/project-beta/guide/index.md` | Project Beta guide landing |
| `docs/projects/project-beta/reference/index.md` | Project Beta reference landing |
| `docs/rfc/index.md` | RFC listing page |
| `docs/rfc/template.md` | RFC template for new proposals |
| `docs/rfc/0001-example-rfc.md` | Example RFC document |
| `docs/best-practices/index.md` | Best practices landing page |
| `docs/best-practices/code-review.md` | Example best practice: code review |
| `docs/best-practices/coding-standards.md` | Example best practice: coding standards |
| `docs/onboarding/index.md` | Onboarding landing page |
| `docs/onboarding/dev-environment.md` | Example onboarding: dev environment setup |
| `.github/CODEOWNERS` | PR review assignment rules |
| `.github/workflows/lint.yml` | Markdown lint + build check CI |
| `.markdownlint.yaml` | Markdownlint configuration (ignore false positives) |
| `.gitignore` | Exclude node_modules, build output, cache |
| `docs/onboarding/cloudflare-setup.md` | Cloudflare Pages + Access setup guide |

### Files to Modify

| File | Change |
|------|--------|
| `README.md` | Rewrite with project description, setup instructions, contributing guide |

---

## Chunk 1: Project Initialization & VitePress Setup

### Task 1: Initialize npm project and install dependencies

**Files:**
- Create: `package.json`

- [ ] **Step 1: Initialize npm project**

Run:
```bash
cd /Users/xl/Code/wiki/.claude/worktrees/wise-chasing-deer
npm init -y
```

- [ ] **Step 2: Update package.json with correct metadata and scripts**

Edit `package.json` to set:
```json
{
  "name": "wiki",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "docs:dev": "vitepress dev docs",
    "docs:build": "vitepress build docs",
    "docs:preview": "vitepress preview docs",
    "lint": "markdownlint-cli2 \"docs/**/*.md\""
  }
}
```

- [ ] **Step 3: Install VitePress and plugins**

Run:
```bash
npm install -D vitepress vitepress-plugin-mermaid mermaid markdownlint-cli2
```

- [ ] **Step 4: Verify installation**

Run:
```bash
npx vitepress --version
```
Expected: Version number printed (1.x.x)

- [ ] **Step 5: Create .gitignore**

Create `.gitignore`:
```
node_modules/
docs/.vitepress/dist/
docs/.vitepress/cache/
```

- [ ] **Step 6: Commit**

```bash
git add package.json package-lock.json .gitignore
git commit -m "chore: initialize npm project with VitePress and dependencies"
```

---

### Task 2: Create VitePress configuration

**Files:**
- Create: `docs/.vitepress/config.ts`

- [ ] **Step 1: Create config directory**

Run:
```bash
mkdir -p docs/.vitepress
```

- [ ] **Step 2: Write config.ts**

Create `docs/.vitepress/config.ts` with the full configuration from the spec:

```ts
import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'Internal Wiki',
    description: '内部技术知识库',

    lastUpdated: true,

    themeConfig: {
      // Chinese UI localization
      lastUpdatedText: '最后更新',
      returnToTopLabel: '返回顶部',
      sidebarMenuLabel: '菜单',
      darkModeSwitchLabel: '主题',
      outlineTitle: '本页目录',
      docFooter: { prev: '上一页', next: '下一页' },

      // Navigation bar
      nav: [
        { text: '首页', link: '/' },
        {
          text: '项目',
          items: [
            { text: 'Project Alpha', link: '/projects/project-alpha/' },
            { text: 'Project Beta', link: '/projects/project-beta/' },
          ],
        },
        { text: 'RFC', link: '/rfc/' },
        { text: '最佳实践', link: '/best-practices/' },
        { text: '新人入门', link: '/onboarding/' },
      ],

      // Sidebar (manual configuration per section)
      sidebar: {
        '/projects/project-alpha/': [
          {
            text: 'Project Alpha',
            items: [
              { text: '概览', link: '/projects/project-alpha/' },
              {
                text: '教程指南',
                collapsed: false,
                items: [
                  { text: '快速开始', link: '/projects/project-alpha/guide/getting-started' },
                ],
              },
              {
                text: '技术参考',
                collapsed: true,
                items: [
                  { text: 'API 文档', link: '/projects/project-alpha/reference/api' },
                ],
              },
              {
                text: '故障排查',
                collapsed: true,
                items: [
                  { text: '概览', link: '/projects/project-alpha/troubleshooting/' },
                ],
              },
            ],
          },
        ],
        '/projects/project-beta/': [
          {
            text: 'Project Beta',
            items: [
              { text: '概览', link: '/projects/project-beta/' },
              {
                text: '教程指南',
                collapsed: false,
                items: [
                  { text: '概览', link: '/projects/project-beta/guide/' },
                ],
              },
              {
                text: '技术参考',
                collapsed: true,
                items: [
                  { text: '概览', link: '/projects/project-beta/reference/' },
                ],
              },
            ],
          },
        ],
        '/rfc/': [
          {
            text: 'RFC',
            items: [
              { text: '概览', link: '/rfc/' },
              { text: 'RFC 模板', link: '/rfc/template' },
              { text: 'RFC-0001: 示例', link: '/rfc/0001-example-rfc' },
            ],
          },
        ],
        '/best-practices/': [
          {
            text: '最佳实践',
            items: [
              { text: '概览', link: '/best-practices/' },
              { text: '代码审查', link: '/best-practices/code-review' },
              { text: '编码规范', link: '/best-practices/coding-standards' },
            ],
          },
        ],
        '/onboarding/': [
          {
            text: '新人入门',
            items: [
              { text: '概览', link: '/onboarding/' },
              { text: '开发环境搭建', link: '/onboarding/dev-environment' },
            ],
          },
        ],
      },

      // Edit link
      editLink: {
        pattern: 'https://github.com/primatrix/wiki/edit/main/docs/:path',
        text: '在 GitHub 上编辑此页',
      },

      // Local search with Chinese tokenization
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
            modal: {
              noResultsText: '未找到相关结果',
              resetButtonTitle: '清除',
              footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
            },
          },
          miniSearch: {
            options: {
              tokenize: (text) => {
                const segmenter = new Intl.Segmenter('zh-CN', { granularity: 'word' })
                const segments = [...segmenter.segment(text)]
                return segments
                  .filter((s) => s.isWordLike)
                  .map((s) => s.segment)
              },
            },
            searchOptions: {
              combineWith: 'AND',
              fuzzy: 0.2,
            },
          },
        },
      },

      // Social links
      socialLinks: [
        { icon: 'github', link: 'https://github.com/primatrix/wiki' },
      ],
    },

    // Mermaid plugin configuration
    mermaid: {},
  })
)
```

- [ ] **Step 3: Create minimal homepage to verify build**

Create `docs/index.md`:
```markdown
---
title: 首页
layout: home
hero:
  name: Internal Wiki
  text: 内部技术知识库
  tagline: 团队知识共享与技术文档中心
  actions:
    - theme: brand
      text: 新人入门
      link: /onboarding/
    - theme: alt
      text: 浏览项目
      link: /projects/project-alpha/
features:
  - title: 项目文档
    details: 按项目组织的技术文档，包括教程、API 参考和故障排查
  - title: RFC
    details: 技术决策记录，追踪重要架构变更和技术选型
  - title: 最佳实践
    details: 团队工程规范和编码标准
  - title: 新人入门
    details: 快速了解团队工作流程和开发环境搭建
---
```

- [ ] **Step 4: Run dev server to verify configuration**

Run:
```bash
npm run docs:dev
```
Expected: Dev server starts without errors, accessible at http://localhost:5173

- [ ] **Step 5: Run build to verify production build works**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds, output in `docs/.vitepress/dist/`

- [ ] **Step 6: Commit**

```bash
git add docs/.vitepress/config.ts docs/index.md
git commit -m "feat: add VitePress config with Chinese locale, search, and mermaid support"
```

---

## Chunk 2: Content Structure — Project & Cross-cutting Pages

### Task 3: Create Project Alpha documentation structure

**Files:**
- Create: `docs/projects/project-alpha/index.md`
- Create: `docs/projects/project-alpha/guide/index.md`
- Create: `docs/projects/project-alpha/guide/getting-started.md`
- Create: `docs/projects/project-alpha/reference/index.md`
- Create: `docs/projects/project-alpha/reference/api.md`
- Create: `docs/projects/project-alpha/troubleshooting/index.md`

- [ ] **Step 1: Create directory structure**

Run:
```bash
mkdir -p docs/projects/project-alpha/{assets,guide,reference,troubleshooting}
```

- [ ] **Step 2: Create project overview page**

Create `docs/projects/project-alpha/index.md`:
```markdown
---
title: Project Alpha
---

# Project Alpha

> 在这里写项目概述。

## 快速链接

- [快速开始](./guide/getting-started.md) — 从零开始搭建开发环境
- [API 文档](./reference/api.md) — 接口参考
- [故障排查](./troubleshooting/) — 常见问题与解决方案
```

- [ ] **Step 3: Create guide section pages**

Create `docs/projects/project-alpha/guide/index.md`:
```markdown
---
title: 教程指南
---

# 教程指南

本栏目包含 Project Alpha 的入门教程和操作指南。

- [快速开始](./getting-started.md) — 从零开始搭建开发环境
```

Create `docs/projects/project-alpha/guide/getting-started.md`:
```markdown
---
title: 快速开始
---

# 快速开始

本文介绍如何从零开始搭建 Project Alpha 的开发环境。

## 前置条件

> 在这里列出前置依赖。

## 安装步骤

> 在这里编写安装步骤。

## 验证

> 在这里说明如何验证安装是否成功。
```

- [ ] **Step 4: Create reference section pages**

Create `docs/projects/project-alpha/reference/index.md`:
```markdown
---
title: 技术参考
---

# 技术参考

本栏目包含 Project Alpha 的技术参考文档。

- [API 文档](./api.md) — 接口定义与使用说明
```

Create `docs/projects/project-alpha/reference/api.md`:
```markdown
---
title: API 文档
---

# API 文档

> 在这里编写 API 接口文档。

## 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/example` | GET | 示例接口 |
```

- [ ] **Step 5: Create troubleshooting section page**

Create `docs/projects/project-alpha/troubleshooting/index.md`:
```markdown
---
title: 故障排查
---

# 故障排查

本栏目记录 Project Alpha 常见问题及解决方案。

## 常见问题

> 在这里添加常见问题。
```

- [ ] **Step 6: Verify build succeeds with new pages**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds without errors.

- [ ] **Step 7: Commit**

```bash
git add docs/projects/
git commit -m "feat: add Project Alpha documentation structure with template pages"
```

---

### Task 3b: Create Project Beta placeholder structure

**Files:**
- Create: `docs/projects/project-beta/index.md`
- Create: `docs/projects/project-beta/guide/index.md`
- Create: `docs/projects/project-beta/reference/index.md`

- [ ] **Step 1: Create directory structure**

Run:
```bash
mkdir -p docs/projects/project-beta/{assets,guide,reference}
```

- [ ] **Step 2: Create project overview page**

Create `docs/projects/project-beta/index.md`:
```markdown
---
title: Project Beta
---

# Project Beta

> 在这里写项目概述。

## 快速链接

- [教程指南](./guide/) — 入门教程和操作指南
- [技术参考](./reference/) — 接口与架构文档
```

- [ ] **Step 3: Create guide and reference landing pages**

Create `docs/projects/project-beta/guide/index.md`:
```markdown
---
title: 教程指南
---

# 教程指南

本栏目包含 Project Beta 的入门教程和操作指南。

> 在这里添加教程文档。
```

Create `docs/projects/project-beta/reference/index.md`:
```markdown
---
title: 技术参考
---

# 技术参考

本栏目包含 Project Beta 的技术参考文档。

> 在这里添加参考文档。
```

- [ ] **Step 4: Verify build**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add docs/projects/project-beta/
git commit -m "feat: add Project Beta placeholder documentation structure"
```

---

### Task 4: Create RFC section

**Files:**
- Create: `docs/rfc/index.md`
- Create: `docs/rfc/template.md`
- Create: `docs/rfc/0001-example-rfc.md`

- [ ] **Step 1: Create RFC directory**

Run:
```bash
mkdir -p docs/rfc
```

- [ ] **Step 2: Create RFC listing page**

Create `docs/rfc/index.md`:
```markdown
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
```

- [ ] **Step 3: Create RFC template**

Create `docs/rfc/template.md`:
```markdown
---
title: "RFC-NNNN: 标题"
status: draft  # draft | review | accepted | rejected | superseded
author: your-github-username
date: YYYY-MM-DD
reviewers: []
---

# RFC-NNNN: 标题

## 概述

> 用 1-2 句话描述提案。

## 背景

> 为什么需要这个变更？当前存在什么问题？

## 方案

> 详细描述技术方案。

### 备选方案

> 考虑过但未采用的方案及原因。

## 影响范围

> 这个变更会影响哪些系统/团队？

## 实施计划

> 分阶段的实施步骤和时间线。

## 风险

> 潜在风险和应对措施。
```

- [ ] **Step 4: Create example RFC**

Create `docs/rfc/0001-example-rfc.md`:
```markdown
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
```

- [ ] **Step 5: Verify build**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add docs/rfc/
git commit -m "feat: add RFC section with template and example RFC"
```

---

### Task 5: Create best practices and onboarding sections

**Files:**
- Create: `docs/best-practices/index.md`
- Create: `docs/best-practices/code-review.md`
- Create: `docs/best-practices/coding-standards.md`
- Create: `docs/onboarding/index.md`
- Create: `docs/onboarding/dev-environment.md`

- [ ] **Step 1: Create directories**

Run:
```bash
mkdir -p docs/best-practices docs/onboarding
```

- [ ] **Step 2: Create best practices pages**

Create `docs/best-practices/index.md`:
```markdown
---
title: 最佳实践
---

# 最佳实践

本栏目汇总团队的工程最佳实践和编码规范。

- [代码审查](./code-review.md) — Code Review 流程与规范
- [编码规范](./coding-standards.md) — 代码风格与命名约定
```

Create `docs/best-practices/code-review.md`:
```markdown
---
title: 代码审查
---

# 代码审查指南

> 在这里编写代码审查的流程和规范。

## Review 流程

> 描述 PR Review 的标准流程。

## Review 检查清单

> 列出 Review 时需要关注的要点。
```

Create `docs/best-practices/coding-standards.md`:
```markdown
---
title: 编码规范
---

# 编码规范

> 在这里编写团队的编码规范。

## 命名约定

> 描述变量、函数、类等的命名规则。

## 代码风格

> 描述代码格式化规则和工具配置。
```

- [ ] **Step 3: Create onboarding pages**

Create `docs/onboarding/index.md`:
```markdown
---
title: 新人入门
---

# 新人入门

欢迎加入团队！本栏目帮助你快速了解团队工作流程和工具。

## 入职指南

- [开发环境搭建](./dev-environment.md) — 配置本地开发环境

## 团队资源

> 在这里列出重要的团队资源链接。
```

Create `docs/onboarding/dev-environment.md`:
```markdown
---
title: 开发环境搭建
---

# 开发环境搭建

本文介绍如何搭建本地开发环境。

## 必备工具

> 在这里列出需要安装的工具。

## 配置步骤

> 在这里编写配置步骤。

## 常见问题

> 在这里记录环境搭建过程中的常见问题。
```

- [ ] **Step 4: Verify build**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add docs/best-practices/ docs/onboarding/
git commit -m "feat: add best practices and onboarding sections with template pages"
```

---

## Chunk 3: CI/CD & GitHub Configuration

### Task 6: Add GitHub CI workflow and CODEOWNERS

**Files:**
- Create: `.github/workflows/lint.yml`
- Create: `.github/CODEOWNERS`
- Create: `.markdownlint.yaml`

- [ ] **Step 1: Create GitHub directories**

Run:
```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create lint workflow**

Create `.github/workflows/lint.yml`:
```yaml
name: Docs Lint

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run docs:build
```

- [ ] **Step 3: Create markdownlint configuration**

Create `.markdownlint.yaml`:
```yaml
# Disable line length rule (documentation often has long lines)
MD013: false

# Allow HTML in markdown (VitePress uses custom components)
MD033: false

# Allow duplicate headings in different sections
MD024:
  siblings_only: true

# Allow trailing punctuation in headings (Chinese punctuation)
MD026: false
```

- [ ] **Step 4: Create CODEOWNERS**

Create `.github/CODEOWNERS`:
```
# Global fallback
* @org/wiki-maintainers

# Per-project ownership
/docs/projects/project-alpha/ @org/team-alpha
/docs/projects/project-beta/  @org/team-beta

# RFC requires architecture team review
/docs/rfc/ @org/architecture-team
```

- [ ] **Step 5: Run lint to verify it passes**

Run:
```bash
npm run lint
```
Expected: No lint errors (or fix any that appear).

- [ ] **Step 6: Verify build still works**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 7: Commit**

```bash
git add .github/ .markdownlint.yaml
git commit -m "ci: add markdown lint workflow, CODEOWNERS, and markdownlint config"
```

---

### Task 7: Update README and create public assets directory

**Files:**
- Modify: `README.md`
- Create: `docs/public/images/.gitkeep`

- [ ] **Step 1: Create public assets directory**

Run:
```bash
mkdir -p docs/public/images
touch docs/public/images/.gitkeep
```

- [ ] **Step 2: Rewrite README.md**

Replace `README.md` content with:
```markdown
# Internal Wiki

内部技术知识库 — 基于 [VitePress](https://vitepress.dev/) 构建，部署在 [Cloudflare Pages](https://pages.cloudflare.com/) 上。

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览构建结果
npm run docs:preview
```

## 如何贡献

1. 创建分支：`git checkout -b docs/your-topic`
2. 编写或修改 Markdown 文件（位于 `docs/` 目录下）
3. 本地预览确认无误：`npm run docs:dev`
4. 提交 PR，等待 Review

## 目录结构

- `docs/projects/` — 按项目组织的技术文档
- `docs/rfc/` — 技术决策记录 (RFC)
- `docs/best-practices/` — 工程最佳实践
- `docs/onboarding/` — 新人入门指南

## 静态资源

- 共享图片放在 `docs/public/images/`
- 项目专属图片放在对应项目的 `docs/projects/<name>/assets/` 目录
- 图片大小建议不超过 500KB，优先使用 WebP 格式
```

- [ ] **Step 3: Verify build**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/public/
git commit -m "docs: rewrite README with setup instructions and contributing guide"
```

---

### Task 8: Final verification — full build and dev server

- [ ] **Step 1: Clean build**

Run:
```bash
rm -rf docs/.vitepress/dist docs/.vitepress/cache
npm run docs:build
```
Expected: Clean build succeeds.

- [ ] **Step 2: Run lint**

Run:
```bash
npm run lint
```
Expected: All markdown files pass lint.

- [ ] **Step 3: Start dev server and verify navigation**

Run:
```bash
npm run docs:dev
```
Verify manually:
- Homepage loads with hero section
- Navigation bar shows all sections (项目, RFC, 最佳实践, 新人入门)
- Sidebar navigation works for each section
- Search opens and UI is in Chinese
- "在 GitHub 上编辑此页" link appears at bottom of pages
- Dark mode toggle works

- [ ] **Step 4: Stop dev server, final commit if any fixes needed**

---

## Chunk 4: Cloudflare Configuration Guide (Documentation Only)

### Task 9: Document Cloudflare Pages and Access setup

This task produces a guide for the ops/admin person who will configure Cloudflare. No code changes — purely documentation.

**Files:**
- Create: `docs/onboarding/cloudflare-setup.md`

- [ ] **Step 1: Create Cloudflare setup guide**

Create `docs/onboarding/cloudflare-setup.md`:
```markdown
---
title: Cloudflare 配置指南
---

# Cloudflare Pages + Access 配置指南

本文介绍如何配置 Cloudflare Pages 自动部署和 Cloudflare Access 认证。

## 前置条件

- Cloudflare 账号（Free 计划即可，Access 免费支持 50 用户）
- 自定义域名已托管在 Cloudflare（可选，也可使用 `*.pages.dev` 域名）
- GitHub Organization 管理员权限

## 第一步：配置 Cloudflare Pages

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. 进入 **Workers & Pages** > **Create**
3. 选择 **Connect to Git**，关联 GitHub 仓库 `primatrix/wiki`
4. 配置构建设置：
   - **Production branch**: `main`
   - **Build command**: `npm run docs:build`
   - **Build output directory**: `docs/.vitepress/dist`
   - **Node.js version**: 在环境变量中设置 `NODE_VERSION=20`
5. 点击 **Save and Deploy**

## 第二步：配置自定义域名（可选）

1. 在 Pages 项目设置中，进入 **Custom domains**
2. 添加域名（如 `wiki.yourcompany.com`）
3. Cloudflare 会自动配置 DNS 和 SSL

## 第三步：配置 Cloudflare Access

1. 进入 **Zero Trust** > **Access** > **Applications**
2. 点击 **Add an application** > **Self-hosted**
3. 配置应用：
   - **Application name**: Internal Wiki
   - **Application domain**: 你的 wiki 域名
   - **Session Duration**: 24 hours
4. 配置 Access Policy：
   - **Policy name**: Org Members Only
   - **Action**: Allow
   - **Include**: Login Methods → GitHub
   - **Require**: GitHub Organizations → `你的组织名`
5. 保存应用

## 第四步：配置 GitHub OAuth App

1. Cloudflare Access 会引导你配置 GitHub OAuth
2. 进入 **Zero Trust** > **Settings** > **Authentication**
3. 添加 GitHub 作为 Identity Provider
4. 按提示在 GitHub 创建 OAuth App 并填入 Client ID 和 Secret

## 验证

1. 使用无痕窗口访问 wiki URL
2. 应看到 Cloudflare Access 登录页面
3. 点击 GitHub 登录，授权后应能看到 wiki 内容
4. 使用非组织成员的 GitHub 账号测试，应被拒绝访问

## 预览部署保护

Cloudflare Pages 会为每个 PR 生成预览 URL（`*.pages.dev`），这些 URL 默认**不受** Access 策略保护。保护方案：

- **方案 A**：在 Cloudflare Access 中创建第二个应用，使用通配符策略覆盖 `*.wiki-project.pages.dev`
- **方案 B**：将预览部署限制到一个已受 Access 保护的自定义域名
- **方案 C**：在 Cloudflare Pages 设置中禁用预览部署（最简单，适用于对未发布内容安全性要求高的场景）
```

- [ ] **Step 2: Update onboarding index to include Cloudflare guide link**

Edit `docs/onboarding/index.md` — add to the 入职指南 section:
```markdown
- [Cloudflare 配置指南](./cloudflare-setup.md) — 部署和认证配置
```

- [ ] **Step 3: Update onboarding sidebar config to include the new page**

Edit `docs/.vitepress/config.ts` — add to the `/onboarding/` sidebar:
```ts
{ text: 'Cloudflare 配置', link: '/onboarding/cloudflare-setup' },
```

- [ ] **Step 4: Verify build**

Run:
```bash
npm run docs:build
```
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add docs/onboarding/cloudflare-setup.md docs/onboarding/index.md docs/.vitepress/config.ts
git commit -m "docs: add Cloudflare Pages and Access setup guide"
```

---

## Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Initialize npm project, install dependencies, create .gitignore | None |
| 2 | Create VitePress config.ts with full configuration | Task 1 |
| 3 | Create Project Alpha documentation structure | Task 2 |
| 3b | Create Project Beta placeholder structure | Task 2 |
| 4 | Create RFC section with template and example | Task 2 |
| 5 | Create best practices and onboarding sections | Task 2 |
| 6 | Add GitHub CI workflow, CODEOWNERS, markdownlint config | Task 1 |
| 7 | Update README and create public assets directory | Task 2 |
| 8 | Final verification — full build and dev server | Tasks 3-7 |
| 9 | Document Cloudflare Pages and Access setup | Task 8 |

**Parallelizable tasks:** Tasks 3, 3b, 4, 5 can run in parallel (independent content sections). Task 6 and 7 can also run in parallel with the content tasks.
