from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .op_events import OpEvent, OpKind, OpStream, RepeatedBlockEvent, PatternBlockEvent


@dataclass
class AnalysisReport:
    bottleneck: str
    stall_ratio: float
    arithmetic_intensity: float
    instruction_mix: Dict[str, float]
    total_flops: int = 0
    total_bytes: int = 0
    total_time_ns: int = 0
    makespan_ns: int = 0


class PerformanceAnalyzer:
    """
    Interprets an OpEvent tree into high-level insights:
      * bottleneck classification (compute / memory / latency),
      * stall ratio,
      * arithmetic intensity,
      * instruction mix (by time share).

    All aggregation happens here; the simulator is responsible only for
    producing per-execution OpEvents.
    """

    def __init__(self, root_event: OpEvent):
        self.root = root_event

    def analyze(self) -> AnalysisReport:
        if self.root is None:
            return AnalysisReport(
                bottleneck="Empty",
                stall_ratio=0.0,
                arithmetic_intensity=0.0,
                instruction_mix={},
                total_flops=0,
                total_bytes=0,
                total_time_ns=0,
                makespan_ns=0,
            )

        # Aggregate metrics from the OpEvent tree.
        total_flops = 0
        total_bytes = 0
        total_time_ns = 0
        vpu_active = 0
        dma_active = 0
        stall_total = 0
        per_op_time_ns: Dict[str, int] = {}

        def visit(ev: OpEvent) -> None:
            nonlocal total_flops, total_bytes, total_time_ns
            nonlocal vpu_active, dma_active, stall_total

            # Leaf / block / loop / stall events contribute directly.
            # Container events without their own work (ROOT/IF) are accounted
            # for via their children.
            # LOOP events now carry scaled metrics (flops/bytes/time × trip_count)
            # and should be counted as a unit, not recursed into.
            if ev.kind in (OpKind.LEAF, OpKind.BLOCK):
                total_flops += ev.flops
                total_bytes += ev.bytes
                total_time_ns += ev.duration_ns
                per_op_time_ns[ev.name] = per_op_time_ns.get(ev.name, 0) + ev.duration_ns

                if ev.stream == OpStream.VPU:
                    vpu_active += ev.duration_ns
                elif ev.stream == OpStream.DMA:
                    dma_active += ev.duration_ns

                # Recurse into children for these types
                for child in ev.children:
                    visit(child)

            elif ev.kind == OpKind.LOOP:
                # LOOP events are now self-contained with scaled metrics.
                # We count the loop as a whole, not its children.
                total_flops += ev.flops
                total_bytes += ev.bytes
                total_time_ns += ev.duration_ns
                per_op_time_ns[ev.name] = per_op_time_ns.get(ev.name, 0) + ev.duration_ns

                if ev.stream == OpStream.VPU:
                    vpu_active += ev.duration_ns
                elif ev.stream == OpStream.DMA:
                    dma_active += ev.duration_ns
                # Do NOT recurse into loop children (already scaled)

            elif ev.kind == OpKind.STALL:
                stall_total += ev.duration_ns
                # Stalls don't have children

            elif isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                # These block events are just containers for compressed children
                # Don't count them directly - their children are already counted
                # No recursion needed - children timing already aggregated
                pass

            else:
                # Container events like ROOT, IF: recurse into children
                for child in ev.children:
                    visit(child)

        visit(self.root)

        makespan = max(self.root.end_time_ns - self.root.start_time_ns, 0)
        if makespan == 0:
            return AnalysisReport(
                bottleneck="Empty",
                stall_ratio=0.0,
                arithmetic_intensity=0.0,
                instruction_mix={},
                    total_flops=total_flops,
                    total_bytes=total_bytes,
                    total_time_ns=total_time_ns,
                    makespan_ns=makespan,
            )

        vpu_util = vpu_active / makespan
        dma_util = dma_active / makespan
        stall_ratio = stall_total / makespan

        if vpu_util > 0.85:
            bottleneck = "Compute Bound (VPU)"
        elif dma_util > 0.85:
            bottleneck = "Memory Bound (DMA)"
        elif stall_ratio > 0.4:
            bottleneck = "Latency Bound (Stalled)"
        else:
            bottleneck = "Balanced / Mixed"

        # Instruction mix by time (leaf events only).
        mix: Dict[str, float] = {}
        if total_time_ns > 0:
            for op, t in per_op_time_ns.items():
                mix[op] = t / total_time_ns

        if total_bytes > 0:
            arithmetic_intensity = float(total_flops) / float(total_bytes)
        else:
            arithmetic_intensity = 0.0

        return AnalysisReport(
            bottleneck=bottleneck,
            stall_ratio=stall_ratio,
            arithmetic_intensity=arithmetic_intensity,
            instruction_mix=mix,
            total_flops=total_flops,
            total_bytes=total_bytes,
            total_time_ns=total_time_ns,
            makespan_ns=makespan,
        )
