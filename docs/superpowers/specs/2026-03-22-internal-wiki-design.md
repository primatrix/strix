# Internal Wiki Knowledge Base Design

## Overview

Build an internal knowledge base for a 50+ person engineering team using VitePress (static site generator) deployed on Cloudflare Pages, with Cloudflare Access providing GitHub OAuth authentication restricted to GitHub Organization members.

## Requirements

| Requirement | Solution |
|-------------|----------|
| Access control (Org members only) | Cloudflare Access + GitHub OAuth |
| Collaborative editing with review | GitHub PR workflow + CODEOWNERS |
| Version history | Git commit history + VitePress last-updated |
| Chinese language UI | VitePress i18n configuration |
| Multi-project support | Project-first directory structure |
| Search | VitePress built-in local search (minisearch) with Chinese tokenization |

## Architecture

```
User Browser
    │
    ▼
Cloudflare Access (Zero Trust Gateway)
    │  - Intercepts all requests
    │  - Redirects unauthenticated users to GitHub OAuth
    │  - Verifies GitHub Org membership
    │  - Issues JWT session token (24h TTL)
    │
    ▼
Cloudflare Pages (CDN + Hosting)
    │  - Serves VitePress static site
    │  - Auto-builds from GitHub repo on push to main
    │  - Auto-generates preview URLs for PRs
    │
    ▼
VitePress Static Site
    - Markdown → HTML rendering
    - Client-side local search (with CJK tokenizer)
    - Dark mode support
```

### Content Management Flow

```
Editor writes/edits Markdown
    │
    ├── Push to main ──────▶ Cloudflare Pages auto-build + deploy to production
    │
    └── Create/Update PR ──▶ Cloudflare Pages auto-build + deploy preview
                               │
                               ▼
                            PR Review (CODEOWNERS-based assignment)
                               │
                               ▼
                            Merge to main → Production deploy
```

## Directory Structure

```
wiki/
├── .github/
│   ├── CODEOWNERS                      # PR review assignment rules
│   └── workflows/
│       └── lint.yml                    # Markdown lint + broken link check
├── docs/
│   ├── .vitepress/
│   │   ├── config.ts                   # VitePress main config
│   │   └── theme/
│   │       └── index.ts                # Custom theme extensions (if needed)
│   ├── public/
│   │   └── images/                     # Shared static assets (logos, icons)
│   ├── index.md                        # Homepage
│   ├── projects/                       # Project-specific docs
│   │   ├── project-alpha/
│   │   │   ├── index.md                # Project overview
│   │   │   ├── assets/                 # Project-specific images & files
│   │   │   ├── guide/                  # Tutorials & guides
│   │   │   │   ├── index.md
│   │   │   │   └── getting-started.md
│   │   │   ├── reference/              # Technical reference
│   │   │   │   ├── index.md
│   │   │   │   └── api.md
│   │   │   └── troubleshooting/        # Troubleshooting
│   │   │       └── index.md
│   │   └── project-beta/
│   │       ├── index.md
│   │       ├── assets/
│   │       ├── guide/
│   │       └── reference/
│   ├── rfc/                            # Cross-project RFCs
│   │   ├── index.md                    # RFC list
│   │   ├── template.md                 # RFC template
│   │   └── 0001-example-rfc.md
│   ├── best-practices/                 # Shared best practices
│   │   ├── index.md
│   │   ├── code-review.md
│   │   └── coding-standards.md
│   └── onboarding/                     # New member onboarding
│       ├── index.md
│       └── dev-environment.md
├── package.json
├── package-lock.json                   # Lockfile for deterministic builds
└── README.md
```

### Content Categories

| Category | Purpose | Example Content |
|----------|---------|----------------|
| `projects/<name>/guide/` | Tutorials & getting started | Onboarding, environment setup, dev workflow |
| `projects/<name>/reference/` | Technical reference | API docs, architecture design, DB schema |
| `projects/<name>/troubleshooting/` | Issue resolution | Common issues, runbooks |
| `rfc/` | Technical decision records | Tech selection proposals, architecture changes |
| `best-practices/` | Engineering standards | Code review guidelines, security practices |
| `onboarding/` | New member guides | Team intro, tooling setup, workflow overview |

### Static Assets Convention

- **Shared assets** (logos, favicons): `docs/public/images/`
- **Project-specific assets**: `docs/projects/<name>/assets/` (co-located with content)
- **Image size limit**: Keep individual images under 500KB; use compressed formats (WebP preferred, PNG/JPG acceptable)
- **Referencing images**: Use relative paths from the Markdown file (e.g., `./assets/architecture.png`)

### Frontmatter Schema

All pages should include frontmatter. Required fields vary by content type:

**All pages (required):**
```yaml
---
title: Page Title
---
```

**RFC pages (required):**
```yaml
---
title: "RFC-0001: Feature Name"
status: draft | review | accepted | rejected | superseded
author: GitHub username
date: YYYY-MM-DD
reviewers:
  - reviewer1
  - reviewer2
---
```

**Project guide/reference pages (optional but recommended):**
```yaml
---
title: Getting Started with Project Alpha
lastUpdated: true
editLink: true
---
```

## Authentication & Access Control

### Cloudflare Access Configuration

1. **Application Type**: Self-hosted Application
2. **Application Domain**: `wiki.yourcompany.com` (or `xxx.pages.dev`)
3. **Access Policy**:
   - **Include Rule**: Login Methods → GitHub
   - **Require Rule**: GitHub Organization → `<your-org-name>`
4. **Session Duration**: 24 hours (configurable)
5. **Same-site Cookie**: Enabled for security

### Preview Deployment Protection

Cloudflare Pages preview deployments use `*.pages.dev` subdomains that are **not** automatically covered by the Access policy. To protect preview content:

- Configure a second Cloudflare Access application with a wildcard policy covering `*.wiki-project.pages.dev`
- Alternatively, restrict preview deployments to a custom preview domain that is covered by the Access policy
- As a simpler option: disable preview deployments in Cloudflare Pages settings if security of unreleased content is critical

### Authentication Flow

1. User navigates to wiki URL
2. Cloudflare Access intercepts the request
3. If no valid session: redirect to GitHub OAuth consent screen
4. User authenticates with GitHub
5. Cloudflare verifies user is a member of the specified GitHub Organization
6. Cloudflare issues a signed JWT cookie
7. Subsequent requests within session duration proceed without re-authentication

### Security Properties

- All requests (including static assets like images, CSS, JS) are protected
- No authentication code in the VitePress application itself
- Audit logs available in Cloudflare dashboard
- Session revocation possible through Cloudflare Access dashboard

## CI/CD Pipeline

### Cloudflare Pages GitHub Integration

Cloudflare Pages natively integrates with GitHub:

- **Production**: Push to `main` triggers automatic build and deployment
- **Preview**: Every PR gets an automatic preview deployment with a unique URL
- **Build Command**: `npm run docs:build`
- **Build Output Directory**: `docs/.vitepress/dist`

### CI Quality Checks (GitHub Actions)

In addition to Cloudflare Pages auto-deploy, a lightweight GitHub Actions workflow runs on PRs:

```yaml
# .github/workflows/lint.yml
name: Docs Lint
on: [pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npx markdownlint-cli2 "docs/**/*.md"
      - run: npm run docs:build  # Also catches broken links
```

### Collaboration Workflow

1. Contributor creates a branch and writes/edits Markdown files
2. Contributor opens a PR against `main`
3. GitHub Actions runs markdown lint and build check
4. Cloudflare Pages automatically builds a preview deployment
5. CODEOWNERS-assigned reviewers review content and preview
6. After approval, PR is merged to `main`
7. Cloudflare Pages automatically deploys to production

### CODEOWNERS Configuration

```
# Global fallback
* @org/wiki-maintainers

# Per-project ownership
/docs/projects/project-alpha/ @org/team-alpha
/docs/projects/project-beta/  @org/team-beta

# RFC requires architecture team review
/docs/rfc/ @org/architecture-team
```

## VitePress Configuration

### config.ts Skeleton

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
            ],
          },
        ],
        // ... other project sidebars
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
                // CJK-aware tokenization using Intl.Segmenter
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

**Sidebar generation strategy**: The sidebar is configured manually in `config.ts`. For a growing knowledge base, consider migrating to the `vitepress-sidebar` plugin (`npm install vitepress-sidebar`) which auto-generates sidebar entries from the directory structure.

### Tech Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Static Site Generator | VitePress | ^1.6.x (pin in package.json) |
| Runtime | Node.js | 20 LTS |
| Package Manager | npm | 10.x |
| Hosting | Cloudflare Pages | - |
| Authentication | Cloudflare Access | - |
| OAuth Provider | GitHub | - |
| Diagrams | vitepress-plugin-mermaid | ^2.x (pin in package.json) |
| Markdown Lint | markdownlint-cli2 | ^0.x (devDependency) |

Use `package-lock.json` for deterministic builds across all contributors and CI.

## Non-Goals

- Full-text search via external service (Algolia) — local search is sufficient for internal use
- Multi-language (i18n) site — Chinese only with English code examples
- User-level access control within the wiki — all Org members have equal access
- Real-time collaborative editing — async PR-based workflow is sufficient
- Comments/discussion on pages — use GitHub PR discussions instead
