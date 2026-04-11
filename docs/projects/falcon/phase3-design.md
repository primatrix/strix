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

### 单 Session 分析

```python
class DeepProfileAnalyzer:
    """Phase 3 深度分析引擎"""

    def __init__(self, analyzer: Analyzer | None = None):
        self.analyzer = analyzer

    def analyze_xprof(self, session: BenchmarkSession) -> XProfAnalysisSummary:
        """生成 XProf 结构化分析 — 聚合已有 adapter 函数"""
        # 获取 xprof run name (从 session 的 profile artifact)
        xprof_run = self._resolve_xprof_run(session)

        # 聚合多个 xprof API 调用
        ov = xprof_adapter.overview(xprof_run)
        ops = xprof_adapter.op_profile(xprof_run, top_n=30)
        rf = xprof_adapter.roofline(xprof_run, top_n=30)
        mem = xprof_adapter.memory(xprof_run)
        mem_bd = xprof_adapter.memory_breakdown(xprof_run)
        trace = xprof_adapter.trace_summary(xprof_run)

        # 构建结构化摘要
        return XProfAnalysisSummary(
            step_time_ms=ov["step_time_ms"],
            mxu_utilization=ov["mxu_utilization"],
            compute_ms=trace["step"]["compute_ms"],
            idle_ms=trace["step"]["idle_ms"],
            comm_ms=trace["step"]["communication_ms"],
            compute_pct=trace["step"]["compute_pct"],
            comm_pct=trace["step"]["comm_pct"],
            idle_pct=trace["step"]["idle_pct"],
            top_ops=[OpSummary(
                operation=op["operation"],
                self_time_us=op["total_self_time_us"],
                pct=op["pct"],
                flop_rate_gflops=op["flop_rate_gflops"],
                memory_bw_gbps=op["memory_bw_gbps"],
                bound_by=op["bound_by"],
                roofline_efficiency_pct=self._get_roofline_eff(op, rf),
            ) for op in ops],
            device_peak_flops_gflops=rf["device"]["peak_flop_rate_gflops"],
            device_peak_hbm_bw_gbps=rf["device"]["peak_hbm_bw_gbps"],
            ops_compute_bound_count=sum(1 for o in ops if o.get("bound_by") == "Compute"),
            ops_memory_bound_count=sum(1 for o in ops if o.get("bound_by") == "Memory"),
            peak_memory_gb=mem.get("summary", {}).get("max_peak_used_gb", 0),
            memory_utilization_pct=mem.get("summary", {}).get("max_utilization_pct", 0),
            remat_buffer_gb=mem_bd.get("summary", {}).get("remat_output_gb", 0),
            non_remat_buffer_gb=mem_bd.get("summary", {}).get("non_remat_output_gb", 0),
            module_breakdown=[ModuleTimeSummary(
                module=m["module"], time_us=m["time_us"], pct=m["pct"],
            ) for m in trace.get("module_breakdown", [])[:15]],
        )

    def analyze_llo(self, session: BenchmarkSession) -> LLOAnalysisSummary:
        """LLO IR 分析 — TODO: 待详细设计"""
        # 1. 从 GCS 下载 LLO IR dump
        # 2. 解析 LLO IR 结构
        # 3. 提取关键指标
        raise NotImplementedError("LLO analysis data model pending design")

    def _resolve_xprof_run(self, session: BenchmarkSession) -> str:
        """从 Session 的 profile artifact 解析 xprof run name

        支持两种 URI 格式:
        - 旧格式: gs://bucket/.../ci-prof-runNNN/tensorboard/
          → 匹配 ci-prof-run / opt- / baseline- 前缀
        - 新格式: gs://falcon-{cluster}/profiles/{session_id}/xplane/
          → 匹配 session_id (bench-YYYYMMDD-hex) 前缀
        """
        if not session.profile or not session.profile.artifacts:
            raise ValueError(f"Session {session.session_id} has no profile artifacts")

        for art in session.profile.artifacts:
            if art.type == "xplane":
                # 先尝试已有的前缀匹配
                run = xprof_adapter.find_xprof_run(art.uri)
                if run:
                    return run
                # 再尝试按 session_id 匹配
                run = xprof_adapter.find_xprof_run_by_name(session.session_id)
                if run:
                    return run

        raise ValueError(f"No xprof run found for session {session.session_id}")

    def _get_roofline_eff(self, op: dict, rf: dict) -> float:
        """从 roofline 数据中查找对应算子的效率"""
        for rf_op in rf.get("ops", []):
            if rf_op["operation"] == op["operation"]:
                return rf_op.get("roofline_efficiency_pct", 0)
        return 0
```

### 跨 Session 深度对比

```python
class DeepProfileAnalyzer:
    # ... (续上)

    def compare_deep(self, baseline: BenchmarkSession,
                     current: BenchmarkSession) -> DeepProfileComparison:
        """两个 Session 的深度 Profile 对比"""

        # 解析 xprof run name — 每个 session 只解析一次
        run_a = self._resolve_xprof_run(baseline)
        run_b = self._resolve_xprof_run(current)

        # 1. 算子级 diff
        op_diffs = self._compare_ops_by_run(run_a, run_b)
        op_regressions = [d for d in op_diffs if d.delta_time_us > 0]
        op_improvements = [d for d in op_diffs if d.delta_time_us < 0]

        # 2. 获取或计算分析摘要 (复用已有结果)
        rf_a = self._get_or_compute_xprof(baseline)
        rf_b = self._get_or_compute_xprof(current)
        cb_ratio_a = rf_a.ops_compute_bound_count / max(1, len(rf_a.top_ops))
        cb_ratio_b = rf_b.ops_compute_bound_count / max(1, len(rf_b.top_ops))

        # 3. 内存对比 (传入已解析的 run name)
        memory_delta = self._compare_memory_by_run(run_a, run_b)

        # 4. 瓶颈迁移检测
        bottleneck_shifts = self._detect_bottleneck_shifts(rf_a, rf_b)

        # 5. Trace 时间线对比
        trace_diff = self._compare_trace(rf_a, rf_b)

        # 6. LLO 对比
        llo_diff = self._compare_llo(baseline, current)

        # 7. 仿真精度
        prediction_accuracy = self._compare_prediction_accuracy(baseline, current)

        return DeepProfileComparison(
            op_regressions=sorted(op_regressions,
                                  key=lambda d: d.delta_time_us, reverse=True)[:20],
            op_improvements=sorted(op_improvements,
                                   key=lambda d: d.delta_time_us)[:20],
            compute_bound_ratio_baseline=cb_ratio_a,
            compute_bound_ratio_current=cb_ratio_b,
            memory_delta=memory_delta,
            bottleneck_shifts=bottleneck_shifts,
            trace_diff=trace_diff,
            llo_diff=llo_diff,
            prediction_accuracy=prediction_accuracy,
        )

    def _compare_ops_by_run(self, run_a: str, run_b: str) -> list[OpDiff]:
        """算子级 diff — 委托给 xprof_adapter.compare_ops"""
        raw_diffs = xprof_adapter.compare_ops(run_a, run_b, top_n=50)

        # 获取 roofline 数据以补充效率信息
        rf_a = xprof_adapter.roofline(run_a, top_n=50)
        rf_b = xprof_adapter.roofline(run_b, top_n=50)
        rf_a_map = {op["operation"]: op for op in rf_a.get("ops", [])}
        rf_b_map = {op["operation"]: op for op in rf_b.get("ops", [])}

        return [
            OpDiff(
                operation=d["operation"],
                time_a_us=d["time_a_us"],
                time_b_us=d["time_b_us"],
                delta_time_us=d["delta_time_us"],
                delta_pct=round(
                    d["delta_time_us"] / d["time_a_us"] * 100, 2
                ) if d["time_a_us"] > 0 else 0,
                bound_a=d["bound_a"],
                bound_b=d["bound_b"],
                bound_changed=(d["bound_a"] != d["bound_b"]
                               and d["bound_a"] != "" and d["bound_b"] != ""),
                efficiency_a_pct=rf_a_map.get(d["operation"], {}).get(
                    "roofline_efficiency_pct", 0),
                efficiency_b_pct=rf_b_map.get(d["operation"], {}).get(
                    "roofline_efficiency_pct", 0),
                flop_rate_a_gflops=d.get("flop_rate_a", 0),
                flop_rate_b_gflops=d.get("flop_rate_b", 0),
            )
            for d in raw_diffs
        ]

    def _compare_memory_by_run(self, run_a: str, run_b: str) -> MemoryDelta:
        """内存分布对比 — 模块级 + remat 分析"""

        mem_a = xprof_adapter.memory(run_a)
        mem_b = xprof_adapter.memory(run_b)
        bd_a = xprof_adapter.memory_breakdown(run_a)
        bd_b = xprof_adapter.memory_breakdown(run_b)

        peak_a = mem_a.get("summary", {}).get("max_peak_used_gb", 0)
        peak_b = mem_b.get("summary", {}).get("max_peak_used_gb", 0)
        remat_a = bd_a.get("summary", {}).get("remat_output_gb", 0)
        remat_b = bd_b.get("summary", {}).get("remat_output_gb", 0)

        # 模块级对比
        mod_a = {m["module"]: m["total_mb"] for m in bd_a.get("by_module", [])}
        mod_b = {m["module"]: m["total_mb"] for m in bd_b.get("by_module", [])}
        all_modules = sorted(set(mod_a) | set(mod_b))

        module_diffs = []
        for mod in all_modules:
            a_mb = mod_a.get(mod, 0)
            b_mb = mod_b.get(mod, 0)
            delta = b_mb - a_mb
            delta_pct = round(delta / a_mb * 100, 1) if a_mb > 0 else (100.0 if b_mb > 0 else 0)
            if abs(delta) >= 1.0:  # 只报告 >= 1MB 的变化
                module_diffs.append(ModuleMemoryDiff(
                    module=mod, baseline_mb=a_mb, current_mb=b_mb,
                    delta_mb=round(delta, 1), delta_pct=delta_pct,
                ))

        module_diffs.sort(key=lambda d: abs(d.delta_mb), reverse=True)

        return MemoryDelta(
            peak_baseline_gb=peak_a,
            peak_current_gb=peak_b,
            delta_gb=round(peak_b - peak_a, 2),
            delta_pct=round((peak_b - peak_a) / peak_a * 100, 1) if peak_a > 0 else 0,
            remat_baseline_gb=remat_a,
            remat_current_gb=remat_b,
            remat_delta_gb=round(remat_b - remat_a, 2),
            module_diffs=module_diffs[:20],
        )

    def _detect_bottleneck_shifts(self, rf_a: XProfAnalysisSummary,
                                   rf_b: XProfAnalysisSummary) -> list[BottleneckShift]:
        """瓶颈迁移检测"""
        shifts = []

        # 1. Top-1 算子变化
        if rf_a.top_ops and rf_b.top_ops:
            top_a = rf_a.top_ops[0].operation
            top_b = rf_b.top_ops[0].operation
            if top_a != top_b:
                shifts.append(BottleneckShift(
                    category="top_op",
                    baseline_value=top_a,
                    current_value=top_b,
                    description=f"Top op shifted: {top_a} → {top_b}",
                ))

        # 2. Compute vs Memory bound 比例翻转
        total_a = len(rf_a.top_ops) or 1
        total_b = len(rf_b.top_ops) or 1
        cb_a = rf_a.ops_compute_bound_count / total_a
        cb_b = rf_b.ops_compute_bound_count / total_b
        if (cb_a > 0.5) != (cb_b > 0.5):
            shifts.append(BottleneckShift(
                category="bound_ratio",
                baseline_value=f"compute-bound {cb_a:.0%}",
                current_value=f"compute-bound {cb_b:.0%}",
                description=(
                    f"Bound ratio flipped: "
                    f"{'compute-dominant' if cb_a > 0.5 else 'memory-dominant'} → "
                    f"{'compute-dominant' if cb_b > 0.5 else 'memory-dominant'}"
                ),
            ))

        # 3. 模块排名变化 > 2 位
        mod_rank_a = {m.module: i for i, m in enumerate(rf_a.module_breakdown)}
        mod_rank_b = {m.module: i for i, m in enumerate(rf_b.module_breakdown)}
        for mod in set(mod_rank_a) & set(mod_rank_b):
            rank_delta = mod_rank_b[mod] - mod_rank_a[mod]
            if abs(rank_delta) > 2:
                direction = "rose" if rank_delta < 0 else "dropped"
                shifts.append(BottleneckShift(
                    category="module_rank",
                    baseline_value=f"rank #{mod_rank_a[mod] + 1}",
                    current_value=f"rank #{mod_rank_b[mod] + 1}",
                    description=f"Module '{mod}' {direction} {abs(rank_delta)} positions",
                ))

        return shifts

    def _compare_trace(self, rf_a: XProfAnalysisSummary,
                       rf_b: XProfAnalysisSummary) -> TraceDiff:
        """Trace 时间线对比"""
        return TraceDiff(
            compute_pct_baseline=rf_a.compute_pct,
            compute_pct_current=rf_b.compute_pct,
            compute_delta_pp=round(rf_b.compute_pct - rf_a.compute_pct, 1),
            comm_pct_baseline=rf_a.comm_pct,
            comm_pct_current=rf_b.comm_pct,
            comm_delta_pp=round(rf_b.comm_pct - rf_a.comm_pct, 1),
            idle_pct_baseline=rf_a.idle_pct,
            idle_pct_current=rf_b.idle_pct,
            idle_delta_pp=round(rf_b.idle_pct - rf_a.idle_pct, 1),
            step_time_baseline_ms=rf_a.step_time_ms,
            step_time_current_ms=rf_b.step_time_ms,
        )

    def _compare_llo(self, baseline: BenchmarkSession,
                     current: BenchmarkSession) -> LLODiff | None:
        """LLO 对比 — TODO: 待详细设计"""
        # 检查两个 Session 是否都有 LLO artifact
        has_llo_a = any(a.type == "llo" for a in (baseline.profile.artifacts if baseline.profile else []))
        has_llo_b = any(a.type == "llo" for a in (current.profile.artifacts if current.profile else []))
        if not has_llo_a or not has_llo_b:
            return None
        # TODO: 实现 LLO IR diff
        return None

    def _compare_prediction_accuracy(self, baseline: BenchmarkSession,
                                      current: BenchmarkSession) -> PredictionAccuracy | None:
        """仿真精度对比

        注意: step_time_ms 仅存在于 TrainingMetrics 和 RLMetrics 中，
        对于 InferenceMetrics/ServingMetrics/KernelMetrics 使用 getattr 安全访问。
        """
        if not baseline.prediction or not current.prediction:
            return None

        b_pred = baseline.prediction.step_time_ms
        b_actual = (getattr(baseline.result.metrics, "step_time_ms", None)
                    if baseline.result and baseline.result.metrics else None)
        c_pred = current.prediction.step_time_ms
        c_actual = (getattr(current.result.metrics, "step_time_ms", None)
                    if current.result and current.result.metrics else None)

        b_err = (round((b_pred - b_actual) / b_actual * 100, 2)
                 if b_pred is not None and b_actual and b_actual > 0 else None)
        c_err = (round((c_pred - c_actual) / c_actual * 100, 2)
                 if c_pred is not None and c_actual and c_actual > 0 else None)

        return PredictionAccuracy(
            baseline_predicted_ms=b_pred,
            baseline_actual_ms=b_actual,
            baseline_error_pct=b_err,
            current_predicted_ms=c_pred,
            current_actual_ms=c_actual,
            current_error_pct=c_err,
            accuracy_improved=(abs(c_err) < abs(b_err)
                               if b_err is not None and c_err is not None else None),
        )

    def _get_or_compute_xprof(self, session: BenchmarkSession) -> XProfAnalysisSummary:
        """获取已有分析结果或重新计算

        注意: 从 DB 加载时需要递归反序列化嵌套 dataclass。
        analysis.summary 是 JSONB dict，内部的 top_ops 和 module_breakdown
        存储为 list[dict]，需要转换回 OpSummary / ModuleTimeSummary 实例。
        """
        # 优先从 analysis_results 表中查找已有结果
        if session.profile and session.profile.analyses:
            for analysis in session.profile.analyses:
                if analysis.analyzer == "xprof":
                    return self._deserialize_xprof_summary(analysis.summary)

        # 不存在则实时计算
        return self.analyze_xprof(session)

    @staticmethod
    def _deserialize_xprof_summary(d: dict) -> XProfAnalysisSummary:
        """从 JSONB dict 反序列化为 XProfAnalysisSummary，处理嵌套 dataclass"""
        d = dict(d)  # shallow copy
        d["top_ops"] = [OpSummary(**op) for op in d.get("top_ops", [])]
        d["module_breakdown"] = [
            ModuleTimeSummary(**m) for m in d.get("module_breakdown", [])
        ]
        return XProfAnalysisSummary(**d)
```

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

```python
class ComparisonEngine:
    """Phase 2 对比引擎 + Phase 3 深度分析"""

    def __init__(self, deep_analyzer: DeepProfileAnalyzer | None = None):
        self.deep_analyzer = deep_analyzer

    def compare_sessions(self, baseline: BenchmarkSession,
                         current: BenchmarkSession,
                         target_metric: str,
                         target_direction: str) -> SessionComparison:
        """Phase 2 原有逻辑 + Phase 3 深度对比"""
        # Phase 2: 指标 diff、配置 diff、硬件 diff、浅层 profile diff
        metric_deltas = self._compute_metric_deltas(...)
        config_diffs = self._compute_config_diffs(...)
        target_diffs = self._compute_target_diffs(...)
        profile_diffs = self._compute_profile_diffs(...)

        # Phase 3: 深度 profile 对比
        deep_profile = None
        if self.deep_analyzer:
            try:
                deep_profile = self.deep_analyzer.compare_deep(baseline, current)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Deep profile analysis failed: {e}")
                # 深度分析失败不影响基础对比

        return SessionComparison(
            session_id=current.session_id,
            baseline_id=baseline.session_id,
            metric_deltas=metric_deltas,
            config_diffs=config_diffs,
            target_diffs=target_diffs,
            profile_diffs=profile_diffs,
            summary=self._build_summary(metric_deltas),
            deep_profile=deep_profile,
        )
```

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

`bottleneck_evolution` 的填充逻辑 — 在 `ComparisonEngine.compare_goal()` 中聚合：

```python
class ComparisonEngine:
    def compare_goal(self, goal, sessions) -> GoalComparisonReport:
        # ... Phase 2 原有逻辑 ...
        comparisons = [
            self.compare_sessions(baseline, s, goal.target_metric, goal.target_direction)
            for s in others
        ]

        # Phase 3: 聚合瓶颈演进 — 按时间顺序收集所有 bottleneck_shifts
        bottleneck_evolution = []
        for comp in comparisons:
            if comp.deep_profile and comp.deep_profile.bottleneck_shifts:
                bottleneck_evolution.extend(comp.deep_profile.bottleneck_shifts)

        return GoalComparisonReport(
            ...,
            bottleneck_evolution=bottleneck_evolution,
        )
```

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

```python
class JobScheduler:
    def __init__(self, ...):
        # Phase 2
        self.comparison_engine = ComparisonEngine(
            deep_analyzer=DeepProfileAnalyzer(analyzer=self.analyzer)
        )

    def _auto_compare(self, session: BenchmarkSession) -> GoalComparisonReport | None:
        # Phase 2 原有逻辑 ...

        # Phase 3: 自动分析 profile (无论是否有 Goal)
        if self._has_xplane_artifact(session):
            try:
                xprof_summary = self.comparison_engine.deep_analyzer.analyze_xprof(session)
                self.analysis_store.add(AnalysisResult(
                    analysis_id=generate_analysis_id(session.session_id, "xprof"),
                    session_id=session.session_id,
                    artifact_id=self._get_xplane_artifact_id(session),
                    analyzer="xprof",
                    summary=dataclasses.asdict(xprof_summary),
                    created_at=now_iso(),
                ))
            except Exception as e:
                logger.warning(f"XProf analysis failed for {session.session_id}: {e}")

        if self._has_llo_artifact(session):
            try:
                llo_summary = self.comparison_engine.deep_analyzer.analyze_llo(session)
                self.analysis_store.add(AnalysisResult(
                    analysis_id=generate_analysis_id(session.session_id, "llo"),
                    session_id=session.session_id,
                    artifact_id=self._get_llo_artifact_id(session),
                    analyzer="llo",
                    summary=dataclasses.asdict(llo_summary),
                    created_at=now_iso(),
                ))
            except Exception as e:
                logger.warning(f"LLO analysis failed for {session.session_id}: {e}")

        # 继续 Phase 2 的 Goal 级对比 ...
        return report

    def _has_xplane_artifact(self, session: BenchmarkSession) -> bool:
        return bool(session.profile and any(
            a.type == "xplane" for a in session.profile.artifacts
        ))

    def _has_llo_artifact(self, session: BenchmarkSession) -> bool:
        return bool(session.profile and any(
            a.type == "llo" for a in session.profile.artifacts
        ))
```

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

### MCP 工具

| 工具 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `deep_analyze` | 单 Session 深度 XProf 分析 | session_id | XProfAnalysisSummary |
| `deep_diff` | 两个 Session 的深度对比 | session_id_a, session_id_b | DeepProfileComparison |
| `op_regression_report` | 算子级回归报告 | session_id_a, session_id_b, top_n | list[OpDiff] |
| `roofline_compare` | Roofline 效率对比 | session_id_a, session_id_b | roofline diff |
| `memory_diff` | 内存分布对比 | session_id_a, session_id_b | MemoryDelta |
| `bottleneck_shift` | 瓶颈迁移检测 | session_id_a, session_id_b | list[BottleneckShift] |
| `prediction_accuracy_trend` | 仿真精度趋势 | goal_id / last_n | trend data |
| `llo_analyze` | LLO IR 分析 (TODO) | session_id | LLOAnalysisSummary |
| `llo_diff` | LLO IR 对比 (TODO) | session_id_a, session_id_b | LLODiff |

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
| 其余组件 | 复用 Phase 1/2 | PostgreSQL / psycopg 3 / click / FastMCP |

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
