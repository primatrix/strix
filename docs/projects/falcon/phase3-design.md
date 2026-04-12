# Phase 3 架构设计：深度 Profile 分析与跨 Session 对比

> **日期:** 2026-04-11
> **状态:** Draft
> **前置:** Phase 1 — 平台化改造 / Phase 2 — Goal-Centric 对比系统
> **范围:** XProf 深度分析引擎 / 算子级回归检测 / Roofline 效率追踪 / 内存分布对比 / 瓶颈迁移检测 / 仿真精度追踪 / LLO IR 分析与对比

---

## 设计目标

在 Phase 2 的 `ComparisonEngine` 基础上，构建 **深度 Profile 分析** 能力，将 Falcon 已有的 XProf adapter（op_profile、roofline、memory_breakdown、trace_summary、compare_ops）系统性地接入对比流程：

1. **算子级回归检测** — 自动对比两个 Session 的 top 算子，识别时间/效率回归
2. **Roofline 效率追踪** — 跨 Session 比较每个算子的 compute/memory 效率
3. **内存分布对比** — 模块级内存对比、remat 分析差异
4. **Trace 时间线对比** — compute/communication/idle 比例演进
5. **瓶颈迁移检测** — 自动检测瓶颈类别变化（top 算子变更、bound 比例翻转、模块排名变化）
6. **仿真精度追踪** — simulation vs actual 的偏差趋势
7. **LLO IR 分析与对比** — LLO IR dump 的解析与跨 Session 对比

---

## 与 Phase 2 的关系

Phase 2 的 `ProfileDiff` 对 profile 分析结果做浅层 dict 比较：

```python
# Phase 2 — 仅对比 summary dict 的 key 变化
class ProfileDiff:
    analyzer: str
    baseline_summary: dict | None
    current_summary: dict | None
    key_changes: list[str]     # "xprof.mxu_utilization: 0.42 → 0.51"
```

Phase 3 将其替换为 **结构化的深度对比**，提供算子级 diff、roofline 效率对比、内存分布变化、瓶颈迁移检测等多维度分析。

---

## Topic 1: 结构化分析结果

### XProf 分析摘要

取代 Phase 2 中 `AnalysisResult.summary` 的非结构化 dict，Phase 3 定义强类型分析摘要：

```python
@dataclass
class OpSummary:
    """单个算子的分析摘要"""
    operation: str                     # 简化后的算子名
    self_time_us: float                # 自身耗时 (μs)
    pct: float                         # 占总时间百分比
    flop_rate_gflops: float            # 实测 FLOP 速率
    memory_bw_gbps: float              # 实测内存带宽
    bound_by: str                      # "Compute" | "Memory"
    roofline_efficiency_pct: float     # roofline 效率

@dataclass
class ModuleTimeSummary:
    """模块级时间分布"""
    module: str                        # JAX 模块名 (decoder, mtp_block, moe_mlp, ...)
    time_us: float
    pct: float

@dataclass
class XProfAnalysisSummary:
    """XProf 结构化分析结果 — 存入 analysis_results.summary"""

    # Overview
    step_time_ms: float
    mxu_utilization: float
    compute_ms: float
    idle_ms: float
    comm_ms: float
    compute_pct: float
    comm_pct: float
    idle_pct: float

    # Top ops (top 30 by self time)
    top_ops: list[OpSummary]

    # Roofline
    device_peak_flops_gflops: float
    device_peak_hbm_bw_gbps: float
    ops_compute_bound_count: int
    ops_memory_bound_count: int

    # Memory
    peak_memory_gb: float
    memory_utilization_pct: float
    remat_buffer_gb: float
    non_remat_buffer_gb: float

    # Module breakdown
    module_breakdown: list[ModuleTimeSummary]
```

### LLO 分析摘要

> **TODO:** LLO 分析数据模型待下次迭代详细设计。以下为占位接口。

```python
@dataclass
class LLOAnalysisSummary:
    """LLO IR 分析结果 — 待详细设计"""
    # TODO: 具体字段待定，需要深入调研 LLO IR 的数据格式和分析维度
    raw: dict                          # 临时使用非结构化 dict
```

### 分析结果存储

分析结果存入 Phase 1 的 `analysis_results` 表，`analyzer` 字段区分类型：

| analyzer | summary 类型 | 数据来源 |
|----------|-------------|---------|
| `xprof` | `XProfAnalysisSummary` | xprof_adapter 各函数聚合 |
| `llo` | `LLOAnalysisSummary` | LLO IR dump 解析 |

---

## Topic 2: 深度对比数据结构

### 算子级 Diff

```python
@dataclass
class OpDiff:
    """单个算子在两个 Session 间的差异"""
    operation: str
    time_a_us: float                   # baseline 耗时
    time_b_us: float                   # current 耗时
    delta_time_us: float               # current - baseline
    delta_pct: float                   # 变化百分比
    bound_a: str                       # baseline 的 bound 类型
    bound_b: str                       # current 的 bound 类型
    bound_changed: bool                # bound 类型是否发生变化
    efficiency_a_pct: float            # baseline roofline 效率
    efficiency_b_pct: float            # current roofline 效率
    flop_rate_a_gflops: float
    flop_rate_b_gflops: float
```

### 瓶颈迁移

```python
@dataclass
class BottleneckShift:
    """检测到的瓶颈迁移"""
    category: str                      # "top_op" | "bound_ratio" | "module_rank"
    baseline_value: str
    current_value: str
    description: str                   # 人类可读的说明

    # 检测规则:
    # "top_op":       耗时最高的算子发生变化
    # "bound_ratio":  compute-bound vs memory-bound 的多数比例翻转
    # "module_rank":  模块时间占比排名变化 > 2 位
```

### 内存 Diff

```python
@dataclass
class MemoryDelta:
    """内存分布差异"""
    peak_baseline_gb: float
    peak_current_gb: float
    delta_gb: float
    delta_pct: float
    remat_baseline_gb: float
    remat_current_gb: float
    remat_delta_gb: float
    module_diffs: list[ModuleMemoryDiff]

@dataclass
class ModuleMemoryDiff:
    """模块级内存差异"""
    module: str
    baseline_mb: float
    current_mb: float
    delta_mb: float
    delta_pct: float
```

### Trace Diff

```python
@dataclass
class TraceDiff:
    """Trace 时间线差异 — compute/communication/idle 比例对比"""
    compute_pct_baseline: float
    compute_pct_current: float
    compute_delta_pp: float            # percentage point delta
    comm_pct_baseline: float
    comm_pct_current: float
    comm_delta_pp: float
    idle_pct_baseline: float
    idle_pct_current: float
    idle_delta_pp: float
    step_time_baseline_ms: float
    step_time_current_ms: float
```

### LLO Diff

> **TODO:** 具体字段待 LLO 分析数据模型设计后展开。

```python
@dataclass
class LLODiff:
    """LLO IR 差异 — 待详细设计"""
    # TODO: 具体字段待定
    raw: dict                          # 临时使用非结构化 dict
```

### 仿真精度

```python
@dataclass
class PredictionAccuracy:
    """仿真预测 vs 实测的精度比较"""
    baseline_predicted_ms: float | None
    baseline_actual_ms: float | None
    baseline_error_pct: float | None   # (predicted - actual) / actual * 100
    current_predicted_ms: float | None
    current_actual_ms: float | None
    current_error_pct: float | None
    accuracy_improved: bool | None     # 仿真精度是否在改善
```

### 完整深度对比结果

```python
@dataclass
class DeepProfileComparison:
    """Phase 3 深度 Profile 对比结果 — 扩展 Phase 2 的 ProfileDiff"""

    # 算子级 diff (按时间变化量排序)
    op_regressions: list[OpDiff]       # 变慢的算子 (delta > 0)
    op_improvements: list[OpDiff]      # 变快的算子 (delta < 0)

    # Roofline 对比
    compute_bound_ratio_baseline: float   # baseline 中 compute-bound 算子占比
    compute_bound_ratio_current: float    # current 中 compute-bound 算子占比

    # 内存对比
    memory_delta: MemoryDelta

    # 瓶颈迁移
    bottleneck_shifts: list[BottleneckShift]

    # Trace 时间线对比
    trace_diff: TraceDiff

    # LLO 对比
    llo_diff: LLODiff | None

    # 仿真精度
    prediction_accuracy: PredictionAccuracy | None
```

---

## Topic 3: DeepProfileAnalyzer 分析引擎

### 架构概览

```text
                    DeepProfileAnalyzer
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   XProfAnalyzer    LLOAnalyzer    PredictionTracker
          │               │               │
          ▼               ▼               ▼
   xprof_adapter     LLO Parser      Analyzer
   (已有实现)        (Phase 3 新增)   (已有实现)
```

### 功能说明

`DeepProfileAnalyzer` 提供以下核心方法：

- **`analyze_xprof(session)`** — 聚合 `xprof_adapter` 的 overview / op_profile / roofline / memory / memory_breakdown / trace_summary 调用，生成 `XProfAnalysisSummary`
- **`analyze_llo(session)`** — LLO IR 分析（TODO: 待详细设计）
- **`compare_deep(baseline, current)`** — 两个 Session 的深度 Profile 对比，返回 `DeepProfileComparison`
- **`_compare_ops_by_run`** — 算子级 diff，委托给 `xprof_adapter.compare_ops`，补充 roofline 效率信息
- **`_compare_memory_by_run`** — 内存分布对比，模块级 + remat 分析（仅报告 >= 1MB 的变化）
- **`_detect_bottleneck_shifts`** — 瓶颈迁移检测：top-1 算子变化、compute/memory bound 比例翻转、模块排名变化 > 2 位
- **`_compare_trace`** — Trace 时间线对比（compute/communication/idle 比例）
- **`_compare_llo`** — LLO 对比（TODO: 待详细设计）
- **`_compare_prediction_accuracy`** — 仿真精度对比（step_time_ms 仅存在于 TrainingMetrics 和 RLMetrics）
- **`_get_or_compute_xprof`** — 优先从 `analysis_results` 表加载已有结果，不存在则实时计算；从 DB 加载时递归反序列化嵌套 dataclass

---

## Topic 4: 与 Phase 2 ComparisonEngine 的集成

> **前置修复:** Phase 2 的 `ComparisonEngine._compute_config_diffs` 引用了 `WorkloadConfig` 类型，
> 但 Phase 1 的 `SessionConfig.config` 是 schema-less `dict`，无法对其调用 `dataclasses.fields()`。
> 实现 Phase 3 前需先修复此问题：将 `_compute_config_diffs` 改为对两个 dict 做 key-level diff。

### 扩展 SessionComparison

```python
@dataclass
class SessionComparison:
    """Phase 2 原有 + Phase 3 扩展"""
    # Phase 2 原有
    session_id: str
    baseline_id: str
    metric_deltas: list[MetricDelta]
    config_diffs: list[ConfigDiff]
    target_diffs: list[TargetDiff]
    profile_diffs: list[ProfileDiff]          # Phase 2 浅层对比 (保留向后兼容)
    summary: str

    # Phase 3 新增
    deep_profile: DeepProfileComparison | None   # 深度 Profile 对比
```

### ComparisonEngine 扩展

`ComparisonEngine` 在 Phase 2 的基础上新增 `deep_analyzer: DeepProfileAnalyzer` 依赖。`compare_sessions()` 在完成 Phase 2 的指标/配置/硬件/浅层 profile diff 后，调用 `deep_analyzer.compare_deep(baseline, current)` 生成深度对比。深度分析失败不影响基础对比（捕获 `RuntimeError` / `ValueError`）。

### GoalComparisonReport 扩展

```python
@dataclass
class GoalComparisonReport:
    """Phase 2 原有 + Phase 3 扩展"""
    # Phase 2 原有
    goal: OptimizationGoal
    baseline: BenchmarkSession
    comparisons: list[SessionComparison]
    trend: dict[str, list[float | None]]
    best_session_id: str | None
    goal_achieved: bool
    generated_at: str

    # Phase 3 新增
    prediction_accuracy_trend: list[dict] | None    # 仿真精度趋势
    bottleneck_evolution: list[BottleneckShift]      # Goal 维度的瓶颈演进 (聚合各 comparison 的 shifts)
```

`bottleneck_evolution` 的填充逻辑 — 在 `ComparisonEngine.compare_goal()` 中按时间顺序聚合各 comparison 的 `deep_profile.bottleneck_shifts`。

---

## Topic 5: 自动触发集成

在 Phase 2 的自动触发机制上扩展：

```text
Session: COLLECTING → COMPLETED
│
├─ Phase 2: 基础对比
│   └─ ComparisonEngine.compare_goal() → metric/config/target/profile diffs
│
└─ Phase 3: 深度分析 (当 Session 有 xplane artifact 时)
    │
    ├─ Step 1: DeepProfileAnalyzer.analyze_xprof(session)
    │   → 存入 analysis_results 表 (analyzer="xprof")
    │
    ├─ Step 2: DeepProfileAnalyzer.analyze_llo(session) [当有 LLO artifact]
    │   → 存入 analysis_results 表 (analyzer="llo")
    │
    └─ Step 3: 如果 Goal 已有 baseline
        ├─ DeepProfileAnalyzer.compare_deep(baseline, current)
        └─ 附加到 GoalComparisonReport.comparisons[].deep_profile
```

`JobScheduler` 在 `__init__` 中将 `DeepProfileAnalyzer` 注入 `ComparisonEngine`。`_auto_compare()` 在 Session 完成时自动执行：当 Session 有 xplane artifact 时调用 `analyze_xprof()` 并存入 `analysis_results` 表（`analyzer="xprof"`）；当有 LLO artifact 时类似调用 `analyze_llo()`（`analyzer="llo"`）。分析失败仅记录 warning，不影响后续 Goal 级对比流程。

---

## Topic 6: 展示层

### CLI 命令

```bash
# ── 单 Session 深度分析 ──
falcon profile deep-analyze <session-id>           # 完整 XProf 深度分析
falcon profile deep-analyze <session-id> --json     # JSON 结构化输出

# ── 跨 Session 深度对比 ──
falcon profile deep-diff <id-1> <id-2>             # 完整深度对比
falcon profile op-diff <id-1> <id-2> [--top 30]    # 算子级回归报告
falcon profile roofline-diff <id-1> <id-2>         # Roofline 效率对比
falcon profile memory-diff <id-1> <id-2>           # 内存分布对比
falcon profile trace-diff <id-1> <id-2>            # Trace 时间线对比
falcon profile bottleneck-shift <id-1> <id-2>      # 瓶颈迁移检测

# ── LLO 分析 (TODO: 待详细设计) ──
falcon profile llo-analyze <session-id>             # LLO IR 分析
falcon profile llo-diff <id-1> <id-2>              # LLO 对比

# ── Goal 级分析 ──
falcon goal prediction-accuracy <goal-id>           # 仿真精度趋势
falcon goal deep-report <goal-id>                   # 带深度分析的 Goal 报告

# ── 仿真精度 ──
falcon session prediction-accuracy <session-id>     # 单 Session 预测 vs 实测
falcon goal prediction-accuracy-trend [--last 20]    # 预测精度趋势
```

### Markdown 报告格式扩展

Phase 2 的 Goal 报告在 Phase 3 中新增 "Deep Profile Analysis" 段落：

```markdown
## Deep Profile Analysis

### 算子级变化 (bench-exp2 vs baseline)

**回归算子 (变慢):**

| 算子 | Baseline | Current | Δ | Bound |
|------|----------|---------|---|-------|
| all-reduce.0 | 8.5ms | 9.2ms | +8.2% ↓ | Memory → Memory |

**改善算子 (变快):**

| 算子 | Baseline | Current | Δ | Bound | Efficiency |
|------|----------|---------|---|-------|------------|
| matmul/dot_general | 12.3ms | 10.1ms | -17.9% ↑ | Compute → Compute | 42% → 51% |
| moe_mlp/einsum | 6.1ms | 4.8ms | -21.3% ↑ | Memory → Compute ⚠ | 38% → 55% |

### 瓶颈迁移

- ⚠ **Top op shifted:** `all-reduce` → `matmul` (compute 成为主要耗时)
- ⚠ **Bound ratio flipped:** memory-dominant (65%) → compute-dominant (58%)
- ⚠ **Module 'moe_mlp'** rose 3 positions (rank #5 → #2)

### 内存对比

| 模块 | Baseline | Current | Δ |
|------|----------|---------|---|
| decoder | 12595 MB | 12083 MB | -4.1% |
| mtp_block | 8294 MB | 8294 MB | 0% |
| remat buffers | 3277 MB | 2149 MB | -34.4% |
| **Peak total** | **30.2 GB** | **28.6 GB** | **-5.3%** |

### Trace 时间线

| | Baseline | Current | Δ |
|------|----------|---------|---|
| Compute | 72.1% | 78.3% | +6.2pp |
| Communication | 21.3% | 17.5% | -3.8pp |
| Idle | 6.6% | 4.2% | -2.4pp |

### Roofline 效率

- Compute-bound ops: 12/30 → 18/30
- Avg compute efficiency: 45.2% → 51.3% (+6.1pp)
- Avg memory BW utilization: 62.1% → 58.3% (-3.8pp)

### 仿真精度

| | Predicted | Actual | Error |
|------|-----------|--------|-------|
| Baseline | 2.81s | 2.80s | +0.3% |
| Current | 2.10s | 2.20s | -4.5% |

### LLO Analysis

> TODO: LLO 对比详情待数据模型设计后补充。
```

---

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| DeepProfileAnalyzer | 纯 Python，组合已有 xprof_adapter | 无新依赖，复用已验证的 XProf HTTP API |
| 瓶颈检测算法 | 基于规则的阈值检测 | 简单可解释，无需 ML |
| LLO Parser | 待设计 | 依赖 LLO IR 格式调研 |
| 报告渲染 | 扩展 Phase 2 的 Jinja2 模板 | 复用已有基础 |
| 其余组件 | 复用 Phase 1/2 | PostgreSQL / psycopg 3 / click |

### 线程安全

`DeepProfileAnalyzer` 在 `JobScheduler.__init__` 中创建并共享。当多个 Session 并发完成时，`analyze_xprof` 和 `compare_deep` 方法可能被并发调用。这些方法设计为**无状态**（不修改实例属性），因此并发安全。实现时需确保：

- 不在方法内修改 `self` 的任何属性
- `xprof_adapter` 的 runs cache 使用模块级变量，已有时间戳保护，并发读取安全（最坏情况多刷新一次）

## 不在 Phase 3 范围

- 跨 Goal 横向对比
- 跨硬件归一化对比 (TPU v4 vs v5e)
- Web Dashboard 可视化 (图表、交互式 diff)
- 自动推荐下一步优化方向 (AI 驱动)
- 回归检测告警管道 (CI 集成)
- 对比结果持久化存储 (始终实时计算)
- LLO 分析数据模型的完整定义 (标记为 TODO)
