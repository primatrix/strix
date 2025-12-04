from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .analyzer import AnalysisReport
from .op_events import OpEvent, OpKind, OpStream, RepeatedBlockEvent, PatternBlockEvent


class BaseExporter(ABC):
    @abstractmethod
    def export(
        self,
        root_event: OpEvent,
        report: AnalysisReport,
        output_path: Optional[str],
    ) -> None:
        raise NotImplementedError


class ChromeTraceExporter(BaseExporter):
    """
    Emit a Chrome trace JSON compatible with Perfetto, based on the OpEvent tree.

    We use three logical tracks:
      * TID=1: VPU (compute + scalar/overhead ops)
      * TID=2: DMA (enqueue_dma)
      * TID=3: Stall bars (visualizing bubbles on the VPU pipeline)
    """

    # Minimum duration threshold in nanoseconds
    # Events with duration below this are filtered out to reduce trace clutter
    MIN_DURATION_NS = 1  # 1 nanosecond

    def export(
        self,
        root_event: OpEvent,
        report: AnalysisReport,
        output_path: Optional[str],
    ) -> None:
        if not output_path:
            return

        trace_events: List[dict] = []
        pid = 1

        # Add thread name metadata events
        trace_events.extend([
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 1,
                "args": {"name": "Compute"}
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 2,
                "args": {"name": "DMA"}
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 3,
                "args": {"name": "Stall"}
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 4,
                "args": {"name": "Control Flow"}
            }
        ])

        def add_trace(ev: OpEvent, time_offset: int = 0) -> None:
            # Materialize executable, block-style, stall and control blocks
            # (e.g. scf.for / scf.if) in the trace.

            # Special handling for LOOP
            if ev.kind == OpKind.LOOP:
                expanded = ev.attributes.get("expanded", False)
                trip_count = ev.attributes.get("trip_count", 1)

                # Add the loop container to TID=4 (Control Flow track)
                loop_name = f"{ev.name} (×{trip_count})" if trip_count > 1 else ev.name
                trace_events.append(
                    {
                        "name": loop_name,
                        "cat": "CONTROL",
                        "ph": "X",
                        "ts": (ev.start_time_ns + time_offset) / 1_000.0,
                        "dur": ev.duration_ns / 1_000.0,
                        "pid": pid,
                        "tid": 4,  # Control Flow track
                        "args": {**ev.attributes, "kind": "loop"},
                        "cname": "blue",  # Blue for loop containers
                    }
                )

                # Then add loop children
                if expanded:
                    # Loop was already expanded during prepare()
                    # Just recurse into children normally
                    for child in ev.children:
                        add_trace(child, time_offset)
                else:
                    # Loop was not expanded (trip_count > 20)
                    # For large loops, we show sampled iterations within the loop's actual time range

                    # Calculate single iteration duration from children
                    if ev.children:
                        # Children times are relative to loop start
                        children_start = min(c.start_time_ns for c in ev.children) if ev.children else 0
                        children_end = max(c.end_time_ns for c in ev.children) if ev.children else 0
                        single_iter_dur = children_end - children_start
                    else:
                        single_iter_dur = 0

                    if single_iter_dur <= 0 or trip_count <= 1:
                        # Can't sample, just show children as-is
                        for child in ev.children:
                            add_trace(child, time_offset)
                    else:
                        # Sample strategy: show representative iterations
                        # Spread them across the loop's actual time range
                        loop_duration_per_iter = ev.duration_ns / trip_count

                        # Show fewer samples for very large loops
                        sample_indices = list(range(min(3, trip_count)))  # First 3
                        if trip_count > 10:
                            sample_indices.append(trip_count // 2)  # Middle
                        if trip_count > 5:
                            sample_indices.extend(range(max(trip_count - 3, 3), trip_count))  # Last 3

                        # Deduplicate and sort
                        sample_indices = sorted(set(i for i in sample_indices if 0 <= i < trip_count))

                        # Emit sampled iterations at their correct time positions
                        for iter_num in sample_indices:
                            # Each iteration starts at: loop_start + iter_num * duration_per_iter
                            iter_start_time = ev.start_time_ns + int(iter_num * loop_duration_per_iter)
                            # Children are relative to iteration start
                            iter_offset = iter_start_time - children_start
                            for child in ev.children:
                                add_trace_with_offset(child, iter_offset, iter_num)

                return  # Done processing loop

            if ev.kind in (
                OpKind.LEAF,
                OpKind.BLOCK,
                OpKind.STALL,
            ):
                # Only add to trace if duration is above threshold
                if ev.duration_ns >= self.MIN_DURATION_NS:
                    if ev.stream == OpStream.VPU:
                        tid = 1
                        cat = "Compute"
                    elif ev.stream == OpStream.DMA:
                        tid = 2
                        cat = "DMA"
                    else:
                        tid = 3
                        cat = "Stall"

                    if ev.kind == OpKind.STALL:
                        cname = "terrible"
                    elif ev.stream == OpStream.DMA:
                        cname = "yellow"
                    elif isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                        cname = "good"  # Green for compressed blocks
                    else:
                        # Compute vs overhead can be refined later; for now use grey.
                        cname = "grey"

                    # Chrome trace format expects timestamps in microseconds.
                    trace_events.append(
                        {
                            "name": ev.name,
                            "cat": cat,
                            "ph": "X",
                            "ts": (ev.start_time_ns + time_offset) / 1_000.0,
                            "dur": ev.duration_ns / 1_000.0,
                            "pid": pid,
                            "tid": tid,
                            "args": ev.attributes,
                            "cname": cname,
                        }
                    )

            # Handle scf.if: add to TID=4
            if ev.kind == OpKind.IF:
                trace_events.append(
                    {
                        "name": ev.name,
                        "cat": "CONTROL",
                        "ph": "X",
                        "ts": (ev.start_time_ns + time_offset) / 1_000.0,
                        "dur": ev.duration_ns / 1_000.0,
                        "pid": pid,
                        "tid": 4,
                        "args": {**ev.attributes, "kind": "if"},
                        "cname": "olive",  # Different color for conditionals
                    }
                )

                # Recurse into children
                for child in ev.children:
                    add_trace(child, time_offset)
                return

            # Don't recurse into compressed block children - they're already represented
            if not isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                for child in ev.children:
                    add_trace(child, time_offset)

        def add_trace_with_offset(ev: OpEvent, iter_offset: int, iter_num: int) -> None:
            """Add event with iteration-specific time offset and annotation."""
            # For sampled loop iterations, we add control flow containers and leaf ops

            # Control flow containers (LOOP/IF) get added to TID=4
            if ev.kind == OpKind.LOOP:
                trip_count = ev.attributes.get("trip_count", 1)
                loop_name = f"{ev.name} (×{trip_count})" if trip_count > 1 else ev.name
                trace_events.append({
                    "name": f"{loop_name} [iter {iter_num}]",
                    "cat": "CONTROL",
                    "ph": "X",
                    "ts": (ev.start_time_ns + iter_offset) / 1_000.0,
                    "dur": ev.duration_ns / 1_000.0,
                    "pid": pid,
                    "tid": 4,
                    "args": {**ev.attributes, "kind": "loop", "iteration": iter_num},
                    "cname": "blue",
                })
                # Recurse into children
                for child in ev.children:
                    add_trace_with_offset(child, iter_offset, iter_num)
                return

            if ev.kind == OpKind.IF:
                trace_events.append({
                    "name": f"{ev.name} [iter {iter_num}]",
                    "cat": "CONTROL",
                    "ph": "X",
                    "ts": (ev.start_time_ns + iter_offset) / 1_000.0,
                    "dur": ev.duration_ns / 1_000.0,
                    "pid": pid,
                    "tid": 4,
                    "args": {**ev.attributes, "kind": "if", "iteration": iter_num},
                    "cname": "olive",
                })
                # Recurse into children
                for child in ev.children:
                    add_trace_with_offset(child, iter_offset, iter_num)
                return

            if ev.kind in (
                OpKind.LEAF,
                OpKind.BLOCK,
                OpKind.STALL,
            ):
                # Only add to trace if duration is above threshold
                if ev.duration_ns >= self.MIN_DURATION_NS:
                    if ev.stream == OpStream.VPU:
                        tid = 1
                        cat = "Compute"
                    elif ev.stream == OpStream.DMA:
                        tid = 2
                        cat = "DMA"
                    else:
                        tid = 3
                        cat = "Stall"

                    if ev.kind == OpKind.STALL:
                        cname = "terrible"
                    elif ev.stream == OpStream.DMA:
                        cname = "yellow"
                    elif isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                        cname = "good"
                    else:
                        cname = "grey"

                    # Add iteration number to name for clarity
                    display_name = f"{ev.name} [iter {iter_num}]" if iter_num < 5 or iter_num >= 19995 else ev.name

                    trace_events.append(
                        {
                            "name": display_name,
                            "cat": cat,
                            "ph": "X",
                            "ts": (ev.start_time_ns + iter_offset) / 1_000.0,
                            "dur": ev.duration_ns / 1_000.0,
                            "pid": pid,
                            "tid": tid,
                            "args": {**ev.attributes, "iteration": iter_num},
                            "cname": cname,
                        }
                    )

            # Recurse for nested structures (but not for compressed blocks)
            if not isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                for child in ev.children:
                    add_trace_with_offset(child, iter_offset, iter_num)

        add_trace(root_event)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace_events, f)


class ConsoleExporter(BaseExporter):
    """
    Simple human-readable console summary. This focuses on the high-level
    questions during the inner development loop.
    """

    def export(
        self,
        root_event: OpEvent,
        report: AnalysisReport,
        output_path: Optional[str] = None,
    ) -> None:
        # ANSI colors for a tiny bit of emphasis.
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        print("\n=== Mosaic Static Profiler ===")
        print(f"Total Est. Time:   {report.total_time_ns:,} ns")

        color = RED if "Bound" in report.bottleneck else GREEN
        print(f"Bottleneck:        {color}{report.bottleneck}{RESET}")
        print(f"Stall Ratio:       {report.stall_ratio:.1%}")

        if report.stall_ratio > 0.3:
            print(
                f"{RED}[WARN] High stall ratio detected. "
                f"Check DMA/Compute overlap.{RESET}"
            )

        # Static roofline-style summary.
        if report.total_bytes > 0:
            print(
                f"Arithmetic Intensity: "
                f"{report.arithmetic_intensity:.3f} FLOPs/Byte"
            )
        else:
            print("Arithmetic Intensity: N/A (no modeled bytes)")

        # Lightweight instruction mix by time share (top few only).
        if report.instruction_mix:
            print("\n[Instruction Mix by Estimated Cycles]")
            sorted_items = sorted(
                report.instruction_mix.items(), key=lambda kv: kv[1], reverse=True
            )
            for op, frac in sorted_items[:10]:
                print(f"- {op:20} {frac * 100.0:6.2f}%")


class JsonSummaryExporter(BaseExporter):
    """
    Export a compact JSON summary suitable for regression testing or
    piping into additional tooling.
    """

    def export(
        self,
        root_event: OpEvent,
        report: AnalysisReport,
        output_path: Optional[str],
    ) -> None:
        if not output_path:
            return

        data = {
            "total_time_ns": report.total_time_ns,
            "makespan_ns": report.makespan_ns,
            "bottleneck": report.bottleneck,
            "stall_ratio": report.stall_ratio,
            "arithmetic_intensity": report.arithmetic_intensity,
            "total_flops": report.total_flops,
            "total_bytes": report.total_bytes,
            "instruction_mix": report.instruction_mix,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
