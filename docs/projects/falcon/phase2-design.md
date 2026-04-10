# Phase 2 架构设计：Goal-Centric Benchmark 对比系统

> **日期:** 2026-04-10
> **状态:** Draft
> **前置:** Phase 1 — BenchmarkSession 元数据重构 / 多 K8s 任务调度 / Profile 数据存储与展示
> **范围:** OptimizationGoal 模型增强 / 全维度 Comparison 引擎 / 自动触发 + 报告生成

---

## 设计目标

在 Phase 1 的 BenchmarkSession 基础上，构建以 **OptimizationGoal** 为核心的系统性对比能力：

1. **Goal 模型增强** — 将 Goal 从简单的标签提升为一等实体，关联 baseline、target metric、达标标准
2. **全维度 Comparison 引擎** — 无状态计算层，支持 Metrics 数值对比、Config 参数差异、Profile 分析结果对比、硬件环境差异
3. **自动触发 + 报告** — Session COMPLETED 时自动触发对比，生成 CLI 表格 / MCP 结构化输出 / Markdown 报告

---

## Topic 1: Goal 模型增强

### 数据模型

Phase 1 的 `optimization_goals` 表只有 name/description/owner/status，Phase 2 将其扩展为完整的优化目标实体：

```python
@dataclass
class OptimizationGoal:
    goal_id: str                        # "goal-20260410-a1b2c3d4"
    name: str                           # "EP8 训练 MFU 提升"
    description: str | None
    owner: str                          # 负责人
    status: str                         # "open" | "achieved" | "abandoned"

    # ── Phase 2 新增 ──
    workload_type: WorkloadType         # 约束此 Goal 下的 Session 类型
    target_metric: str                  # 主要优化指标 "mfu" | "step_time_ms" | ...
    target_value: float | None          # 目标值（可选），如 MFU >= 0.55
    target_direction: str               # "higher_is_better" | "lower_is_better"
    baseline_session_id: str | None     # 自动设为第一个 COMPLETED session

    created_at: str
    updated_at: str
```

### ID 生成规则

```text
goal_id: "goal-{YYYYMMDD}-{8hex}"
```

### Session 与 Goal 的关联

将 `Collaboration.optimization_goal`（自由文本）改为 sessions 表的一等列 `goal_id`，建立外键关系：

```sql
ALTER TABLE sessions ADD COLUMN goal_id TEXT REFERENCES optimization_goals(goal_id);
CREATE INDEX idx_sessions_goal ON sessions(goal_id);
```

### Baseline 自动选取规则

1. Goal 创建时 `baseline_session_id = NULL`
2. 第一个关联到此 Goal 且状态为 COMPLETED 的 Session 自动成为 baseline
3. 用户可通过 `falcon goal set-baseline <goal-id> <session-id>` 手动更换

### DB Schema 变更

```sql
-- 增强 optimization_goals 表
ALTER TABLE optimization_goals ADD COLUMN workload_type TEXT;
ALTER TABLE optimization_goals ADD COLUMN target_metric TEXT;
ALTER TABLE optimization_goals ADD COLUMN target_value FLOAT;
ALTER TABLE optimization_goals ADD COLUMN target_direction TEXT
    NOT NULL DEFAULT 'higher_is_better'
    CHECK (target_direction IN ('higher_is_better', 'lower_is_better'));
ALTER TABLE optimization_goals ADD COLUMN baseline_session_id TEXT
    REFERENCES sessions(session_id);
ALTER TABLE optimization_goals ADD COLUMN updated_at TIMESTAMPTZ
    NOT NULL DEFAULT now();

-- 更新 status 约束
ALTER TABLE optimization_goals DROP CONSTRAINT IF EXISTS optimization_goals_status_check;
ALTER TABLE optimization_goals ADD CONSTRAINT optimization_goals_status_check
    CHECK (status IN ('open', 'achieved', 'abandoned'));
```

---

## Topic 2: Comparison 计算引擎

### 对比数据结构

```python
@dataclass
class MetricDelta:
    """单个指标的对比结果"""
    metric: str                         # "mfu" | "step_time_ms" | ...
    baseline_value: float | None
    current_value: float | None
    absolute_delta: float | None        # current - baseline
    relative_delta_pct: float | None    # (current - baseline) / baseline * 100
    direction: str                      # "improved" | "regressed" | "unchanged" | "unknown"
    is_target_metric: bool              # 是否为 Goal 的主要优化指标

@dataclass
class ConfigDiff:
    """Config 参数差异"""
    field: str                          # "tp" | "dp" | "remat" | ...
    baseline_value: Any
    current_value: Any
    changed: bool

@dataclass
class TargetDiff:
    """硬件环境差异"""
    field: str                          # "cluster" | "accelerator" | "device_num"
    baseline_value: Any
    current_value: Any
    changed: bool

@dataclass
class ProfileDiff:
    """Profile 分析结果对比"""
    analyzer: str                       # "xprof" | "llo" | "roofline"
    baseline_summary: dict | None
    current_summary: dict | None
    key_changes: list[str]              # 人类可读的关键变化描述

@dataclass
class SessionComparison:
    """一个 Session 与 baseline 的完整对比"""
    session_id: str
    baseline_id: str
    metric_deltas: list[MetricDelta]
    config_diffs: list[ConfigDiff]
    target_diffs: list[TargetDiff]
    profile_diffs: list[ProfileDiff]
    summary: str                        # "MFU +12.3%, step_time -8.1%"

@dataclass
class GoalComparisonReport:
    """Goal 级别的完整对比报告"""
    goal: OptimizationGoal
    baseline: BenchmarkSession
    comparisons: list[SessionComparison]  # 按时间排序
    trend: dict[str, list[float | None]]  # 关键指标的时间趋势
    best_session_id: str | None          # target_metric 最优的 session
    goal_achieved: bool                  # target_value 是否达标
    generated_at: str
```

### ComparisonEngine

```python
class ComparisonEngine:
    """无状态对比引擎"""

    def compare_sessions(self, baseline: BenchmarkSession,
                         current: BenchmarkSession,
                         target_metric: str,
                         target_direction: str) -> SessionComparison:
        """两个 Session 的全维度对比"""
        metric_deltas = self._compute_metric_deltas(
            baseline.result.metrics, current.result.metrics,
            target_metric, target_direction,
        )
        config_diffs = self._compute_config_diffs(
            baseline.config.workload_config, current.config.workload_config,
        )
        target_diffs = self._compute_target_diffs(baseline.target, current.target)

        baseline_analyses = baseline.profile.analyses if baseline.profile else []
        current_analyses = current.profile.analyses if current.profile else []
        profile_diffs = self._compute_profile_diffs(baseline_analyses, current_analyses)

        return SessionComparison(
            session_id=current.session_id,
            baseline_id=baseline.session_id,
            metric_deltas=metric_deltas,
            config_diffs=config_diffs,
            target_diffs=target_diffs,
            profile_diffs=profile_diffs,
            summary=self._build_summary(metric_deltas),
        )

    def compare_goal(self, goal: OptimizationGoal,
                     sessions: list[BenchmarkSession]) -> GoalComparisonReport:
        """Goal 下所有 Session 与 baseline 的对比 + 趋势"""
        if not sessions:
            raise ValueError("No sessions provided for comparison")
        sessions_sorted = sorted(sessions, key=lambda s: s.created_at)
        baseline = next(
            (s for s in sessions_sorted if s.session_id == goal.baseline_session_id),
            sessions_sorted[0],
        )
        others = [s for s in sessions_sorted if s.session_id != baseline.session_id]

        comparisons = [
            self.compare_sessions(baseline, s, goal.target_metric, goal.target_direction)
            for s in others
        ]

        trend = self._build_trend(sessions_sorted)
        best_session_id = self._find_best(sessions_sorted, goal)
        goal_achieved = self._check_achieved(sessions_sorted, goal, best_session_id)

        return GoalComparisonReport(
            goal=goal, baseline=baseline, comparisons=comparisons,
            trend=trend, best_session_id=best_session_id,
            goal_achieved=goal_achieved, generated_at=now_iso(),
        )

    def _compute_metric_deltas(self, baseline_metrics: WorkloadMetrics,
                                current_metrics: WorkloadMetrics,
                                target_metric: str,
                                target_direction: str) -> list[MetricDelta]:
        """利用 dataclass fields 反射，逐一计算 delta"""
        deltas = []
        for f in dataclasses.fields(baseline_metrics):
            b_val = getattr(baseline_metrics, f.name)
            c_val = getattr(current_metrics, f.name)

            abs_delta, rel_pct, direction = None, None, "unknown"
            if b_val is not None and c_val is not None:
                abs_delta = c_val - b_val
                rel_pct = round((abs_delta / b_val) * 100, 2) if b_val != 0 else None
                if abs_delta == 0.0:
                    direction = "unchanged"
                elif target_direction == "higher_is_better":
                    direction = "improved" if abs_delta > 0 else "regressed"
                else:
                    direction = "improved" if abs_delta < 0 else "regressed"

            deltas.append(MetricDelta(
                metric=f.name, baseline_value=b_val, current_value=c_val,
                absolute_delta=abs_delta, relative_delta_pct=rel_pct,
                direction=direction, is_target_metric=(f.name == target_metric),
            ))
        return deltas

    def _compute_config_diffs(self, baseline_config: WorkloadConfig,
                               current_config: WorkloadConfig) -> list[ConfigDiff]:
        """利用 dataclass fields 反射，逐一比较"""
        return [
            ConfigDiff(
                field=f.name,
                baseline_value=getattr(baseline_config, f.name),
                current_value=getattr(current_config, f.name),
                changed=(getattr(baseline_config, f.name) != getattr(current_config, f.name)),
            )
            for f in dataclasses.fields(baseline_config)
        ]

    def _compute_target_diffs(self, baseline_target: SessionTarget,
                               current_target: SessionTarget) -> list[TargetDiff]:
        """比较硬件环境字段"""
        return [
            TargetDiff(
                field=f.name,
                baseline_value=getattr(baseline_target, f.name),
                current_value=getattr(current_target, f.name),
                changed=(getattr(baseline_target, f.name) != getattr(current_target, f.name)),
            )
            for f in dataclasses.fields(baseline_target)
        ]

    def _compute_profile_diffs(self, baseline_analyses: list[AnalysisResult],
                                current_analyses: list[AnalysisResult]) -> list[ProfileDiff]:
        """按 analyzer 类型匹配，对比 summary dict"""
        baseline_by = {a.analyzer: a for a in baseline_analyses}
        current_by = {a.analyzer: a for a in current_analyses}

        diffs = []
        for analyzer in sorted(set(baseline_by) | set(current_by)):
            b = baseline_by.get(analyzer)
            c = current_by.get(analyzer)
            b_sum, c_sum = (b.summary if b else None), (c.summary if c else None)

            key_changes = []
            if b_sum is None:
                key_changes.append(f"{analyzer}: missing in baseline")
            elif c_sum is None:
                key_changes.append(f"{analyzer}: missing in current")
            else:
                for key in set(b_sum) | set(c_sum):
                    if b_sum.get(key) != c_sum.get(key):
                        key_changes.append(f"{analyzer}.{key}: {b_sum.get(key)} → {c_sum.get(key)}")

            diffs.append(ProfileDiff(
                analyzer=analyzer, baseline_summary=b_sum,
                current_summary=c_sum, key_changes=key_changes,
            ))
        return diffs

    def _build_summary(self, deltas: list[MetricDelta]) -> str:
        parts = []
        for d in deltas:
            if d.relative_delta_pct is None:
                continue
            if d.is_target_metric or abs(d.relative_delta_pct) >= 5.0:
                sign = "+" if d.relative_delta_pct > 0 else ""
                parts.append(f"{d.metric} {sign}{d.relative_delta_pct}%")
        return ", ".join(parts) if parts else "no significant changes"

    def _build_trend(self, sessions: list[BenchmarkSession]) -> dict[str, list[float | None]]:
        trend: dict[str, list[float | None]] = {}
        for s in sessions:
            if s.result and s.result.metrics:
                for f in dataclasses.fields(s.result.metrics):
                    val = getattr(s.result.metrics, f.name)
                    if isinstance(val, (int, float)):
                        trend.setdefault(f.name, []).append(val)
                    else:
                        trend.setdefault(f.name, []).append(None)
        return trend

    def _find_best(self, sessions: list[BenchmarkSession],
                   goal: OptimizationGoal) -> str | None:
        best_id, best_val = None, None
        for s in sessions:
            if not (s.result and s.result.metrics):
                continue
            val = getattr(s.result.metrics, goal.target_metric, None)
            if val is None:
                continue
            if best_val is None:
                best_val, best_id = val, s.session_id
            elif goal.target_direction == "higher_is_better" and val > best_val:
                best_val, best_id = val, s.session_id
            elif goal.target_direction == "lower_is_better" and val < best_val:
                best_val, best_id = val, s.session_id
        return best_id

    def _check_achieved(self, sessions: list[BenchmarkSession],
                        goal: OptimizationGoal, best_id: str | None) -> bool:
        if goal.target_value is None or best_id is None:
            return False
        best = next((s for s in sessions if s.session_id == best_id), None)
        if best is None:
            return False
        val = getattr(best.result.metrics, goal.target_metric, None)
        if val is None:
            return False
        if goal.target_direction == "higher_is_better":
            return val >= goal.target_value
        return val <= goal.target_value
```

### 关键设计决策

- **利用 Registry 做字段反射** — `WORKLOAD_METRICS_REGISTRY` 和 `WORKLOAD_CONFIG_REGISTRY` 保证新增 workload 类型时 diff 逻辑自动适配，无需为每种 workload 编写独立的 diff 逻辑
- **`target_direction` 决定语义** — `higher_is_better` 时 positive delta = improved；`lower_is_better` 时 negative delta = improved
- **Profile diff 按 analyzer 匹配** — 同一个 analyzer 的 baseline 和 current 结果做对比，缺失某方的标记为 N/A
- **对比结果临时计算，不持久化** — 避免数据一致性问题，每次查询实时计算

---

## Topic 3: 自动触发机制

### 触发点

挂在 Phase 1 的 Session 状态机 `COLLECTING → COMPLETED` 转换上：

```python
class JobScheduler:
    def __init__(self, ...):
        # ... Phase 1 原有 ...
        self.comparison_engine = ComparisonEngine()
        self.report_renderer = ReportRenderer()

    def collect(self, session: BenchmarkSession) -> None:
        # ... Phase 1 原有收集逻辑 ...
        # Session: COLLECTING → COMPLETED

        # ── Phase 2 新增：自动触发对比 ──
        if session.goal_id:
            self._auto_compare(session)

    def _auto_compare(self, session: BenchmarkSession) -> GoalComparisonReport | None:
        goal = self.goal_store.get(session.goal_id)
        if not goal:
            return None

        # 自动设定 baseline（第一个 COMPLETED session）
        if not goal.baseline_session_id:
            self.goal_store.set_baseline(goal.goal_id, session.session_id)
            return None  # 第一个 session，无需对比

        # 获取 Goal 下所有已完成的 Session
        sessions = self.goal_store.get_sessions(
            goal.goal_id, status="completed"
        )

        # 计算对比
        report = self.comparison_engine.compare_goal(goal, sessions)

        # 生成报告
        md = self.report_renderer.render_markdown(report)
        # 返回给调用方（MCP / CLI）
        return report
```

### 自动触发流程

```text
Session 状态: COLLECTING → COMPLETED
│
├─ goal_id 为空 → 不触发，正常结束
│
├─ goal_id 非空
│   ├─ Goal 的 baseline_session_id 为空
│   │   └─ 设此 Session 为 baseline，不触发对比
│   │
│   └─ Goal 已有 baseline
│       ├─ 获取 Goal 下所有 COMPLETED sessions
│       ├─ ComparisonEngine.compare_goal()
│       ├─ ReportRenderer.render_markdown()
│       └─ 返回 GoalComparisonReport
```

---

## Topic 4: 展示层

### CLI 命令

```bash
# Goal 管理（增强）
falcon goal create --name "EP8 MFU 提升" \
  --workload training \
  --target-metric mfu \
  --target-value 0.55 \
  --target-direction higher_is_better

falcon goal list [--status open] [--owner X]
falcon goal get <goal-id>
falcon goal set-baseline <goal-id> <session-id>

# 对比
falcon goal compare <goal-id>                       # Goal 下全量对比
falcon goal compare <goal-id> --last 5              # 最近 5 个 session
falcon session diff <session-id-1> <session-id-2>   # 任意两个 session pairwise diff

# 报告
falcon goal report <goal-id>                        # Markdown 报告输出到 stdout
falcon goal report <goal-id> --output report.md     # 保存到文件
falcon goal report <goal-id> --format json          # 结构化 JSON 输出
```

### MCP 工具扩展

| 工具 | 用途 | 示例 |
|------|------|------|
| `goal_create` | 创建优化目标 | "创建一个 EP8 MFU 提升的目标" |
| `goal_compare` | 对比 Goal 下所有 Session | "EP8 目标的对比情况怎样" |
| `goal_report` | 生成 Markdown 对比报告 | "生成 EP8 目标的对比报告" |
| `goal_set_baseline` | 手动更换 baseline | "把 bench-xxx 设为 baseline" |
| `session_diff` | 任意两个 Session 的 pairwise diff | "对比 bench-aaa 和 bench-bbb" |

### Markdown 报告格式

```markdown
# 优化目标对比报告: EP8 训练 MFU 提升

**目标:** MFU >= 0.55 | **状态:** 进行中 | **进度:** 0.42 → 0.51 (78.5%)
**生成时间:** 2026-04-10T14:30:00Z

## Baseline: bench-20260408-a1b2c3d4

- Config: TP=4, DP=8, EP=8, FSDP=4, remat=full
- Hardware: tpu-v4-128 @ tpu-v4-prod
- MFU: 0.42 | step_time: 2.8s | memory: 30.2GB

## Session 对比

| Session | MFU | Δ MFU | step_time | Δ step | Config 变更 | 硬件 |
|---------|-----|-------|-----------|--------|-------------|------|
| bench-...-baseline | 0.42 | — | 2.8s | — | — | v4-128 |
| bench-...-exp1 | 0.45 | +7.1% ↑ | 2.5s | -10.7% ↑ | remat=minimal | v4-128 |
| bench-...-exp2 | 0.51 | +21.4% ↑ | 2.2s | -21.4% ↑ | +cp=2 | v4-128 |

## 趋势

MFU: 0.42 → 0.45 → 0.51 (+21.4% 总提升)
step_time: 2.8s → 2.5s → 2.2s (-21.4% 总提升)

## Profile 分析变化

- **bench-...-exp1 vs baseline:** XProf 瓶颈未变化（all-reduce），LLO 无显著差异
- **bench-...-exp2 vs baseline:** XProf 瓶颈从 "all-reduce" 变为 "matmul"，LLO 编译时间减少 15%

## 最佳 Session: bench-...-exp2

目标达成进度: 92.7% (0.51 / 0.55)
```

### 预留 Web API（不在 Phase 2 实现）

```text
GET    /api/v1/goals
GET    /api/v1/goals/{goal_id}
POST   /api/v1/goals
PATCH  /api/v1/goals/{goal_id}
GET    /api/v1/goals/{goal_id}/compare
GET    /api/v1/goals/{goal_id}/report
GET    /api/v1/sessions/{id1}/diff/{id2}
```

---

## Topic 5: 数据访问层

### GoalStore

```python
class GoalStore:
    """Goal CRUD + 查询"""

    def __init__(self, conn) -> None:
        self.conn = conn

    def create(self, goal: OptimizationGoal) -> None: ...
    def update(self, goal: OptimizationGoal) -> None: ...
    def get(self, goal_id: str) -> OptimizationGoal | None: ...
    def get_by_name(self, name: str) -> OptimizationGoal | None: ...
    def list(self, *, status: str | None = None,
             owner: str | None = None) -> list[OptimizationGoal]: ...
    def set_baseline(self, goal_id: str, session_id: str) -> None: ...
    def get_sessions(self, goal_id: str, *,
                     status: str | None = None) -> list[BenchmarkSession]: ...
```

### SessionStore 扩展

```python
class SessionStore:
    # Phase 1 已有，增加 goal_id 过滤
    def list(self, ..., goal_id: str | None = None, ...) -> list[BenchmarkSession]: ...

    # Phase 2：替换原来的空 compare，委托给 ComparisonEngine
    def compare(self, id1: str, id2: str) -> SessionComparison: ...
```

### ReportRenderer

```python
class ReportRenderer:
    """对比报告渲染器"""

    def render_markdown(self, report: GoalComparisonReport) -> str:
        """生成 Markdown 格式的对比报告"""

    def render_table(self, report: GoalComparisonReport) -> str:
        """生成 CLI 表格格式"""

    def render_json(self, report: GoalComparisonReport) -> dict:
        """生成结构化 JSON"""
```

---

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| ComparisonEngine | 纯 Python dataclass 反射 | 利用 Phase 1 的 Registry，零额外依赖 |
| ReportRenderer | Jinja2 模板 | Markdown 报告生成，轻量灵活 |
| 其余组件 | 复用 Phase 1 | PostgreSQL / psycopg 3 / click / FastMCP |

## 不在 Phase 2 范围

- Web Dashboard 可视化对比（图表、交互式 diff）
- 自动推荐下一步优化方向（AI 驱动）
- 跨 Goal 的横向对比
- Goal 自动归档策略
- 对比结果持久化存储
