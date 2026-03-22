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
| Search | VitePress built-in local search (minisearch) |

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
    - Client-side local search
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
│   └── CODEOWNERS                  # PR review assignment rules
├── .vitepress/
│   ├── config.ts                   # VitePress main config
│   └── theme/
│       └── index.ts                # Custom theme extensions (if needed)
├── docs/
│   ├── index.md                    # Homepage
│   ├── projects/                   # Project-specific docs
│   │   ├── project-alpha/
│   │   │   ├── index.md            # Project overview
│   │   │   ├── guide/              # Tutorials & guides
│   │   │   │   ├── index.md
│   │   │   │   └── getting-started.md
│   │   │   ├── reference/          # Technical reference
│   │   │   │   ├── index.md
│   │   │   │   └── api.md
│   │   │   └── troubleshooting/    # Troubleshooting
│   │   │       └── index.md
│   │   └── project-beta/
│   │       ├── index.md
│   │       ├── guide/
│   │       └── reference/
│   ├── rfc/                        # Cross-project RFCs
│   │   ├── index.md                # RFC list
│   │   ├── template.md             # RFC template
│   │   └── 0001-example-rfc.md
│   ├── best-practices/             # Shared best practices
│   │   ├── index.md
│   │   ├── code-review.md
│   │   └── coding-standards.md
│   └── onboarding/                 # New member onboarding
│       ├── index.md
│       └── dev-environment.md
├── package.json
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

## Authentication & Access Control

### Cloudflare Access Configuration

1. **Application Type**: Self-hosted Application
2. **Application Domain**: `wiki.yourcompany.com` (or `xxx.pages.dev`)
3. **Access Policy**:
   - **Include Rule**: Login Methods → GitHub
   - **Require Rule**: GitHub Organization → `<your-org-name>`
4. **Session Duration**: 24 hours (configurable)
5. **Same-site Cookie**: Enabled for security

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
- **Build Output Directory**: `.vitepress/dist`

No custom GitHub Actions workflow is needed for basic build/deploy — Cloudflare Pages handles this automatically.

### Collaboration Workflow

1. Contributor creates a branch and writes/edits Markdown files
2. Contributor opens a PR against `main`
3. Cloudflare Pages automatically builds a preview deployment
4. CODEOWNERS-assigned reviewers review content and preview
5. After approval, PR is merged to `main`
6. Cloudflare Pages automatically deploys to production

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

### Key Configuration Points

1. **Language**: `zh-CN` with full Chinese UI localization
2. **Sidebar**: Auto-generated from directory structure, collapsible sections
3. **Search**: Built-in `local` search provider (minisearch-based, supports Chinese)
4. **Mermaid Diagrams**: Via `vitepress-plugin-mermaid` for flowcharts and architecture diagrams
5. **Last Updated**: Git-based timestamps shown on each page
6. **Edit Link**: "Edit this page on GitHub" link at the bottom of each page
7. **Dark Mode**: Built-in, no additional configuration needed

### Navigation Bar Structure

```
Homepage | Projects ▾ | RFC | Best Practices | Onboarding
                    ├── Project Alpha
                    ├── Project Beta
                    └── ...
```

### Tech Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Static Site Generator | VitePress | Latest (1.x) |
| Runtime | Node.js | 18+ |
| Package Manager | npm | Latest |
| Hosting | Cloudflare Pages | - |
| Authentication | Cloudflare Access | - |
| OAuth Provider | GitHub | - |
| Diagrams | vitepress-plugin-mermaid | Latest |

## Non-Goals

- Full-text search via external service (Algolia) — local search is sufficient for internal use
- Multi-language (i18n) site — Chinese only with English code examples
- User-level access control within the wiki — all Org members have equal access
- Real-time collaborative editing — async PR-based workflow is sufficient
- Comments/discussion on pages — use GitHub PR discussions instead
