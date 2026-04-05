# Beaver Cloudflare Infrastructure Design

Phase 1 基础架构设计，基于 Cloudflare Workers 平台构建 GitHub 智能通知与跟进系统。

## 1. Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IM Platforms | DingTalk + Feishu | 目标用户群体的主要企业 IM |
| Scale | Large (10+ repos, multi-org) | 需多租户架构 |
| Environments | Production only | 先简单，后续按需加分层 |
| Worker Architecture | Monolith + split points | 单 Worker 代码库，模块化设计预留拆分 |
| IaC Tool | Wrangler CLI | 轻量级，开发体验好 |
| Language | TypeScript | 类型安全，Workers 原生支持 |
| LLM Integration | Pluggable provider | 接口抽象，按需切换 provider |
| CI/CD | GitHub Actions | 与 GitHub-centric 设计一致 |

## 2. Overall Architecture

### Hybrid Model: Fast Path + Queue

单 Worker (`beaver-worker`) 通过 3 个入口处理所有请求：

1. **`fetch` handler** — 接收 GitHub Webhook HTTP 请求
2. **`scheduled` handler** — Cron 定时任务（晨间提醒、日报、周报、过期检测）
3. **`queue` handler** — Queue consumer，处理异步任务

```text
GitHub Webhook ──▶ [Main Worker: fetch handler]
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Fast Path          Heavy Path
     (标签校验/评论)      (LLM分析/报告)
              │                 │
              ▼                 ▼
         GitHub API      ┌───────────┐
         (直接写回)      │ Task Queue│
                         └─────┬─────┘
                               ▼
                        [Main Worker: queue handler]
                          │    │    │
                          ▼    ▼    ▼
                        D1  LLM   Notify
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                            DingTalk   Feishu

Cron Trigger ──▶ [Main Worker: scheduled handler] ──▶ Task Queue
```

### Fast Path vs Heavy Path

**Fast Path** (同步，< 1s):

- Webhook 签名校验 (HMAC-SHA256)
- 事件去重 (KV lookup by delivery ID)
- 标签守门员 (status flow validation)
- Check Run 创建/更新
- 简单 GitHub Comment 回复

**Heavy Path** (异步入队):

- LLM 推理 (事件分析、风险评估、报告生成)
- 批量 GitHub API 查询
- 通知推送 (DingTalk / Feishu)
- 任务拆解审计

### Split Point Strategy

代码按 SRA 框架模块化组织。各模块之间通过 Queue 消息类型解耦：

- `sensing/` → 可独立为 Ingestion Worker（零 LLM 依赖）
- `reasoning/` → 可独立为 Analysis Worker（CPU/内存密集）
- `acting/` → 可独立为 Action Worker（I/O 密集，高扇出）
- `queue/` → 消息类型定义是模块间契约，拆分后保持不变

## 3. Data Layer

### D1 Database: beaver-db

7 张核心表：

**installations** — GitHub App 安装记录，多租户根节点

```sql
CREATE TABLE installations (
  id INTEGER PRIMARY KEY,
  github_app_id INTEGER NOT NULL,
  org_login TEXT NOT NULL,
  permissions TEXT NOT NULL,       -- JSON
  config_json TEXT DEFAULT '{}',   -- per-installation config
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**repositories** — 仓库注册

```sql
CREATE TABLE repositories (
  id INTEGER PRIMARY KEY,
  installation_id INTEGER NOT NULL REFERENCES installations(id),
  full_name TEXT NOT NULL,         -- e.g. "org/repo"
  default_branch TEXT DEFAULT 'main',
  is_active INTEGER DEFAULT 1,
  synced_at TEXT,
  UNIQUE(full_name)
);
```

**events** — Webhook 事件记录（存摘要不存全量 payload）

```sql
CREATE TABLE events (
  id TEXT PRIMARY KEY,             -- UUID
  repo_id INTEGER NOT NULL REFERENCES repositories(id),
  github_delivery_id TEXT NOT NULL UNIQUE,
  event_type TEXT NOT NULL,        -- issues, pull_request, etc.
  action TEXT NOT NULL,            -- opened, labeled, closed, etc.
  payload_summary TEXT NOT NULL,   -- JSON: key fields only
  processed_at TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**notifications** — 通知投递记录

```sql
CREATE TABLE notifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT REFERENCES events(id),
  channel TEXT NOT NULL,           -- dingtalk, feishu, github_comment
  recipient TEXT NOT NULL,         -- webhook URL or user ID
  status TEXT NOT NULL DEFAULT 'pending', -- pending, sent, failed
  retry_count INTEGER DEFAULT 0,
  content_hash TEXT,               -- dedup for reports
  sent_at TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**reports** — LLM 生成的报告

```sql
CREATE TABLE reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id INTEGER NOT NULL REFERENCES repositories(id),
  type TEXT NOT NULL,              -- morning_focus, daily, weekly, retro
  content_md TEXT NOT NULL,
  metadata_json TEXT DEFAULT '{}',
  generated_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**user_mappings** — GitHub 用户到 IM 平台用户的映射

```sql
CREATE TABLE user_mappings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  installation_id INTEGER NOT NULL REFERENCES installations(id),
  github_login TEXT NOT NULL,
  platform TEXT NOT NULL,           -- dingtalk, feishu
  platform_user_id TEXT NOT NULL,   -- DingTalk userId / Feishu open_id (标准化)
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(installation_id, github_login, platform)
);
```

> **Feishu ID 标准化**: `platform_user_id` 对 Feishu 统一存储 `open_id`（应用级唯一标识）。不使用 `union_id`（跨应用）或 `user_id`（租户级），避免多租户/多应用场景下的映射冲突。

**llm_usage** — LLM 调用追踪（成本监控）

```sql
CREATE TABLE llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  task_type TEXT NOT NULL,
  repo_id INTEGER REFERENCES repositories(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### Indexes

```sql
-- FK indexes
CREATE INDEX idx_repositories_installation_id ON repositories(installation_id);
CREATE INDEX idx_events_repo_id ON events(repo_id);
CREATE INDEX idx_notifications_event_id ON notifications(event_id);
CREATE INDEX idx_reports_repo_id ON reports(repo_id);
CREATE INDEX idx_user_mappings_installation_id ON user_mappings(installation_id);
CREATE INDEX idx_llm_usage_repo_id ON llm_usage(repo_id);

-- Query pattern indexes
CREATE INDEX idx_events_github_delivery_id ON events(github_delivery_id);
CREATE INDEX idx_events_processed ON events(repo_id, processed_at);
CREATE INDEX idx_notifications_status ON notifications(status);
CREATE INDEX idx_notifications_channel_status ON notifications(channel, status);
CREATE INDEX idx_reports_type_generated ON reports(repo_id, type, generated_at DESC);
CREATE INDEX idx_user_mappings_lookup ON user_mappings(installation_id, github_login, platform);
```

### KV Namespace: beaver-kv

| Key Pattern | TTL | Purpose |
|-------------|-----|---------|
| `dedup:{delivery_id}` | 24h | Webhook event deduplication |
| `config:{installation_id}` | 1h | Installation config cache (source of truth: D1) |
| `debounce:{repo}:{issue}:{action}` | 5min | Event debounce, prevent notification spam |
| `gh:token:{installation_id}` | 50min | GitHub installation token cache (tokens expire 1h) |
| `llm:cache:{prompt_hash}` | 6h | LLM response cache for identical queries |

Design principle: D1 做结构化持久化，KV 做热路径查询。所有 KV 数据有 TTL，丢失无害。

**Dedup 双重保障**: Fast Path 先查 KV `dedup:{delivery_id}`，命中则丢弃。KV miss 时写入 D1 `events` 表，`github_delivery_id` UNIQUE 约束作为安全网。D1 insert 需用 try-catch 捕获 UNIQUE 冲突，冲突时视为幂等成功（静默跳过），避免 KV 过期后重放导致未处理异常。

## 4. Queue & Async Processing

### Single Queue: beaver-tasks

7 种任务类型通过 `TaskRouter` 分发：

```typescript
type TaskType =
  | "analyze_event"      // LLM 分析事件重要性/风险
  | "send_notification"  // 推送通知到 DingTalk/Feishu
  | "generate_report"    // 生成日报/周报/晨间提醒
  | "audit_split"        // 任务拆解完整性审计
  | "gate_check"         // PR 合并门禁检查
  | "sync_project"       // 批量同步 Project 看板状态
  | "detect_stale"       // 长尾任务/过期检测

interface TaskMessage {
  type: TaskType
  repo_id: number
  payload: {
    event_id?: string
    issue_number?: number
    pr_number?: number
    report_type?: "morning_focus" | "daily" | "weekly" | "retro"
    [key: string]: unknown
  }
  metadata: {
    attempt: number
    enqueued_at: string  // ISO timestamp
    source: "webhook" | "cron" | "retry"
  }
}
```

### Retry & Error Policy

| Policy | Value | Notes |
|--------|-------|-------|
| Max retries | 3 | Queue 内置 retry，失败后自动重新入队 |
| Max batch size | 1 | 逐条处理，避免 LLM 调用累计超过 batch timeout |
| Max batch timeout | 30s | 等待批次填满的最大时间 |
| Retry delay | exponential | CF Queues 内置指数退避 |
| Dead letter queue | beaver-dlq | 3 次重试后进入死信队列，记录到 D1 |
| LLM timeout | 25s | 单次 LLM 调用超时 |

> **Workers Plan**: 需要 Cloudflare Workers Paid plan ($5/mo)。Paid plan 的 Queue consumer 支持最长 15 分钟 wall-clock time，LLM 调用无压力。Free plan 的 CPU 时间限制（10ms）不适用于本项目。
>
> **DLQ 处理策略**: Phase 1 中 `beaver-dlq` 仅做累积，通过 D1 记录失败详情供手动排查。后续可增加 DLQ consumer 实现自动告警或 replay。

### Cron → Queue Mapping (UTC)

| Schedule | Cron | Emitted Tasks |
|----------|------|---------------|
| Morning Focus | `0 1 * * 1-5` (09:00 CST) | Per-repo `generate_report(morning_focus)` + per-user `send_notification` |
| Daily Report | `0 10 * * 1-5` (18:00 CST) | Per-repo `generate_report(daily)` + `send_notification` |
| Weekly Report | `0 10 * * 5` (Fri 18:00 CST) | Per-repo `generate_report(weekly)` + `send_notification` |
| Stale Detection | `0 2 * * *` (10:00 CST) | Per-repo `detect_stale` → 标记 `beaver/stale` 或 `beaver/overdue` |

## 5. LLM Integration Layer

### Pluggable Provider Interface

```typescript
interface LLMProvider {
  name: string
  chat(request: LLMRequest): Promise<LLMResponse>
}

interface LLMRequest {
  system: string
  messages: { role: "user" | "assistant"; content: string }[]
  temperature?: number       // default 0.3
  max_tokens?: number        // default 2048
  response_format?: "text" | "json"
}

interface LLMResponse {
  content: string
  usage: { input_tokens: number; output_tokens: number }
  model: string
  latency_ms: number
}
```

Provider implementations: `AnthropicProvider`, `GeminiProvider`, `OpenAICompatibleProvider`. Factory function `createLLMProvider(env)` reads env config to select provider.

### Reasoning Use Cases

| Use Case | Output | Prompt Template | Input Context |
|----------|--------|-----------------|---------------|
| Event Importance | JSON | `event-analysis` | Event type + payload summary + labels + milestone |
| Risk Assessment | JSON | `risk-assessment` | PR age, review status, CI state, blocking labels |
| Daily/Weekly Report | Markdown | `report-{type}` | Aggregated events, merged PRs, closed issues, milestone progress |
| Split Audit | JSON | `split-audit` | Issue body, sub-task list, LOC estimates |
| Morning Focus | Markdown | `morning-focus` | User's assigned issues/PRs, DDLs, priority labels |

### Cost & Performance Controls

1. **KV Response Cache**: Key = `llm:cache:{sha256(system+messages)}`, TTL 6h. Retry 不会重复调用 LLM.
2. **Prompt Templates**: `src/reasoning/prompts/*.ts`, TypeScript template functions, type-safe input/output.
3. **JSON Output Validation**: Zod schemas 校验结构化输出. Malformed response triggers 1 additional retry with error feedback.
4. **Usage Tracking**: Every call logs to `llm_usage` table (provider, model, tokens, latency, task_type, repo_id).

## 6. Notification Delivery

### Provider Abstraction

```typescript
interface NotifyProvider {
  name: "dingtalk" | "feishu"
  sendGroup(webhook: string, msg: NotifyMessage): Promise<NotifyResult>
  sendDirect(userId: string, msg: NotifyMessage): Promise<NotifyResult>
}

interface NotifyMessage {
  title: string
  body: string           // Markdown content
  urgency: "critical" | "normal" | "low"
  links?: { text: string; url: string }[]
  mentions?: string[]    // @user IDs
}

interface NotifyResult {
  success: boolean
  provider: string
  messageId?: string
  error?: string
}
```

### DingTalk Integration

- **群通知**: Custom Robot Webhook, sign with HmacSHA256 (timestamp + secret), Markdown/ActionCard msgtype
- **私聊**: 企业内部应用 Server API, access token via app credentials, `POST /topapi/message/corpconversation/asyncsend_v2`
- **Secrets**: `DINGTALK_ROBOT_WEBHOOK`, `DINGTALK_ROBOT_SECRET`, `DINGTALK_APP_KEY`, `DINGTALK_APP_SECRET`

### Feishu Integration

- **群通知**: Custom Bot Webhook, sign with HmacSHA256 (timestamp + secret), Interactive Card / Post (富文本)
- **私聊**: 自建应用 Message API, tenant access token, `POST /im/v1/messages?receive_id_type=user_id`
- **Secrets**: `FEISHU_BOT_WEBHOOK`, `FEISHU_BOT_SECRET`, `FEISHU_APP_ID`, `FEISHU_APP_SECRET`

### Channel Routing Matrix

| Scenario | Channel | Urgency | Routing |
|----------|---------|---------|---------|
| P0/Blocker alert | 群 + 私聊 + GitHub | critical | @assignee + @team-lead, always delivered |
| Morning Focus | 私聊 | normal | Per-user DM, skip if no assigned tasks |
| Daily/Weekly Report | 管理群 | normal | Team channel, dedup by content_hash |
| Compliance warning | GitHub Comment | low | On Issue/PR directly, no IM push |
| Stale/Overdue | 私聊 + GitHub | normal | @assignee DM + label, max 1/day |

### Anti-Spam

- **Event debounce**: KV key `debounce:{repo}:{issue}:{action}` (5min TTL) prevents duplicate notifications from rapid changes.
- **Report dedup**: `content_hash` in notifications table prevents resending identical reports.
- **Status-aware muting**: Busy/OOO developers only receive P0 Blockers; others deferred.

## 7. GitHub App Integration

### Authentication Flow

1. App Private Key (PEM) → sign JWT (RS256, 10min expiry)
2. JWT → `POST /app/installations/{id}/access_tokens` → Installation Token (1h expiry)
3. Cache token in KV: `gh:token:{installation_id}` (50min TTL, refresh before expiry)

### Webhook Security

1. Compute `HMAC-SHA256(request_body, WEBHOOK_SECRET)`
2. Compare with `X-Hub-Signature-256` header
3. Check `X-GitHub-Delivery` against KV dedup key
4. Route to handler by `X-GitHub-Event` header

### App Permissions

**Repository**: Issues (R&W), Pull requests (R&W), Checks (R&W), Contents (Read), Metadata (Read)
**Organization**: Projects (R&W), Members (Read)

### Webhook Event Subscriptions

| Event | Actions | Path | SRA Trigger |
|-------|---------|------|-------------|
| `issues` | opened, labeled, unlabeled, assigned, milestoned, closed, reopened | Fast + Heavy | Label guard + importance analysis |
| `pull_request` | opened, synchronize, ready_for_review, closed, review_requested | Fast + Heavy | Gate check + risk analysis |
| `pull_request_review` | submitted | Heavy | Review status tracking + notification |
| `check_suite` | completed | Heavy | CI failure detection + alert |
| `issue_comment` | created, edited | Heavy | DDL / dependency / test evidence parsing |
| `projects_v2_item` | created, edited, deleted | Heavy | Board state tracking |
| `milestone` | closed | Heavy | Trigger retrospective report |

### API Usage Patterns

- **REST API**: Single-entity reads/writes (get issue, post comment, create check run). Used in Fast Path and simple Queue tasks.
- **GraphQL API**: Batch multi-field queries for scheduled tasks (all open issues + labels + milestones, Project V2 board state). Used in Cron-triggered report generation.
- **GitHub Client**: Lightweight `fetch` wrapper (not full octokit). Auto-handles token refresh from KV, rate limit headers, and pagination.

### Secrets

```text
GITHUB_APP_ID            # App ID (numeric)
GITHUB_APP_PRIVATE_KEY   # PEM private key (RS256 signing)
GITHUB_WEBHOOK_SECRET    # Webhook signature verification
```

## 8. Project Structure

```text
beaver/
├── src/
│   ├── index.ts                 # Worker entry: fetch, scheduled, queue handlers
│   ├── router.ts                # Request routing
│   ├── env.ts                   # Env type definitions (bindings, secrets)
│   │
│   ├── sensing/                 # 感知层
│   │   ├── webhook.ts           # Signature verify, dedup, event parsing
│   │   ├── events.ts            # Event type definitions & payload extractors
│   │   └── scheduler.ts         # Cron handler: enumerate repos, emit tasks
│   │
│   ├── reasoning/               # 推理层
│   │   ├── llm/
│   │   │   ├── provider.ts      # LLMProvider interface & factory
│   │   │   ├── anthropic.ts
│   │   │   ├── gemini.ts
│   │   │   └── cache.ts         # KV-based response cache
│   │   ├── prompts/
│   │   │   ├── event-analysis.ts
│   │   │   ├── risk-assessment.ts
│   │   │   ├── report.ts
│   │   │   └── split-audit.ts
│   │   └── schemas/             # Zod schemas for LLM outputs
│   │       ├── analysis.ts
│   │       └── report.ts
│   │
│   ├── acting/                  # 行动层
│   │   ├── notify/
│   │   │   ├── provider.ts      # NotifyProvider interface
│   │   │   ├── dingtalk.ts
│   │   │   ├── feishu.ts
│   │   │   ├── router.ts        # Channel routing & anti-spam
│   │   │   └── formatter.ts     # Platform-specific formatting
│   │   ├── github/
│   │   │   ├── client.ts        # Lightweight GitHub API wrapper
│   │   │   ├── auth.ts          # JWT + installation token management
│   │   │   ├── comments.ts
│   │   │   ├── labels.ts
│   │   │   └── checks.ts
│   │   └── gatekeeper.ts        # Status flow guard + merge gate
│   │
│   ├── queue/                   # Queue infrastructure
│   │   ├── producer.ts          # Typed task enqueue helpers
│   │   ├── consumer.ts          # Batch consumer + TaskRouter
│   │   └── types.ts             # TaskMessage & TaskType definitions
│   │
│   └── db/                      # Data access layer
│       ├── schema.sql           # D1 table definitions
│       ├── migrations/          # Numbered migration files
│       └── repository.ts        # Query helpers
│
├── test/
│   ├── sensing/
│   ├── reasoning/
│   ├── acting/
│   └── fixtures/                # Sample webhook payloads
│
├── wrangler.toml
├── package.json
├── tsconfig.json
├── vitest.config.ts
└── .github/
    └── workflows/
        └── deploy.yml
```

## 9. Wrangler Configuration

```toml
name = "beaver-worker"
main = "src/index.ts"
compatibility_date = "2025-01-01"

# D1 Database
[[d1_databases]]
binding = "DB"
database_name = "beaver-db"
database_id = "<to-be-created>"

# KV Namespace
[[kv_namespaces]]
binding = "KV"
id = "<to-be-created>"

# Queue — producer + consumer in same Worker
[[queues.producers]]
binding = "TASK_QUEUE"
queue = "beaver-tasks"

[[queues.consumers]]
queue = "beaver-tasks"
max_batch_size = 1
max_batch_timeout = 30
max_retries = 3
dead_letter_queue = "beaver-dlq"

[[queues.producers]]
binding = "DLQ"
queue = "beaver-dlq"

# Cron Triggers (UTC)
[triggers]
crons = [
  "0 1 * * 1-5",   # Morning Focus (09:00 CST)
  "0 10 * * 1-5",  # Daily Report  (18:00 CST)
  "0 10 * * 5",    # Weekly Report (Fri 18:00 CST)
  "0 2 * * *",     # Stale Detection (10:00 CST)
]
```

## 10. CI/CD Pipeline

### GitHub Actions: deploy.yml

**Trigger**: push to `main`

```text
Lint (eslint + tsc) → Test (vitest) → Deploy (wrangler deploy + D1 migrations)
```

**PR workflow** (on `pull_request`): Lint + Test only, no deploy.

**Required Secrets** (GitHub repo settings):

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

**Cloudflare Secrets** (via `wrangler secret put`):

- `GITHUB_APP_ID`
- `GITHUB_APP_PRIVATE_KEY`
- `GITHUB_WEBHOOK_SECRET`
- `DINGTALK_ROBOT_WEBHOOK`
- `DINGTALK_ROBOT_SECRET`
- `DINGTALK_APP_KEY`
- `DINGTALK_APP_SECRET`
- `FEISHU_BOT_WEBHOOK`
- `FEISHU_BOT_SECRET`
- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `LLM_API_KEY`
- `LLM_PROVIDER` (anthropic | gemini | openai-compatible)
