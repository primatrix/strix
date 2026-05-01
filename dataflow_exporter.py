"""Render a DataFlowGraph as Graphviz DOT format.

Produces a DOT digraph with:
- Hardware-based subgraph clusters (VPU, DMA) for non-loop nodes
- Loop subgraph clusters for nodes inside loops
- Time-bucket rank constraints for concurrent-node alignment
- Cross-stream edge highlighting (red dashed)
- Node styling by stream/kind
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from .dataflow import DataFlowGraph, DFNode
from .op_events import OpKind, OpStream


def _node_ref(node_id: int) -> str:
    """DOT-safe node identifier from integer id."""
    return f"ev_{node_id}"


def _escape_dot(text: str) -> str:
    """Escape characters that are special inside DOT label strings."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _node_label(node: DFNode) -> str:
    """Build the multi-line label for a node.

    Format::

        opcode
        {start_ns}->{end_ns} ns
        {flops}F {bytes}B          (only if > 0)
    """
    parts = [node.name, f"{node.start_ns}->{node.end_ns} ns"]
    metrics: List[str] = []
    if node.flops > 0:
        metrics.append(f"{node.flops}F")
    if node.bytes > 0:
        metrics.append(f"{node.bytes}B")
    if metrics:
        parts.append(" ".join(metrics))
    return "\\n".join(parts)


def _node_style(node: DFNode) -> str:
    """Return DOT attribute string for a node based on stream/kind."""
    if node.kind == OpKind.STALL:
        return 'shape=box, fillcolor=salmon'
    if node.stream == OpStream.CONTROL:
        return 'shape=diamond, fillcolor=lightgrey'
    if node.stream == OpStream.DMA:
        return 'shape=box, fillcolor=lightyellow'
    # VPU compute (LEAF, BLOCK, etc.)
    return 'shape=box, fillcolor=lightblue'


class DataFlowDotExporter:
    """Renders a ``DataFlowGraph`` as Graphviz DOT text."""

    def render(self, graph: DataFlowGraph) -> str:
        """Return the DOT string for *graph*."""
        lines: List[str] = []
        lines.append("digraph dataflow {")
        lines.append("  rankdir=LR;")
        lines.append('  node [style=filled, fontsize=10];')
        lines.append('  edge [fontsize=8];')
        lines.append("")

        node_map: Dict[int, DFNode] = {n.id: n for n in graph.nodes}

        # Determine which nodes belong to loops.
        loop_node_ids: Set[int] = set()
        for members in graph.loops.values():
            loop_node_ids.update(members)

        # --- Loop clusters ---------------------------------------------------
        for loop_id in sorted(graph.loops):
            member_ids = graph.loops[loop_id]
            lines.append(f"  subgraph cluster_loop_{loop_id} {{")
            lines.append(f'    label="loop_{loop_id}";')
            for nid in member_ids:
                node = node_map.get(nid)
                if node is None:
                    continue
                ref = _node_ref(nid)
                label = _node_label(node)
                style = _node_style(node)
                lines.append(
                    f'    {ref} [label="{_escape_dot(label)}", {style}];'
                )
            lines.append("  }")
            lines.append("")

        # --- Hardware clusters (for nodes NOT inside loops) -------------------
        hw_buckets: Dict[str, List[DFNode]] = defaultdict(list)
        for node in graph.nodes:
            if node.id in loop_node_ids:
                continue
            if node.stream == OpStream.DMA:
                hw_buckets["dma"].append(node)
            elif node.stream == OpStream.CONTROL:
                hw_buckets["vpu"].append(node)
            else:
                hw_buckets["vpu"].append(node)

        for hw_name in sorted(hw_buckets):
            nodes_in_hw = hw_buckets[hw_name]
            lines.append(f"  subgraph cluster_{hw_name} {{")
            lines.append(f'    label="{hw_name.upper()}";')
            for node in nodes_in_hw:
                ref = _node_ref(node.id)
                label = _node_label(node)
                style = _node_style(node)
                lines.append(
                    f'    {ref} [label="{_escape_dot(label)}", {style}];'
                )
            lines.append("  }")
            lines.append("")

        # --- Time bucket rank constraints ------------------------------------
        if graph.nodes:
            all_starts = [n.start_ns for n in graph.nodes]
            all_ends = [n.end_ns for n in graph.nodes]
            min_t = min(all_starts)
            max_t = max(all_ends)
            makespan = max_t - min_t
            num_buckets = min(100, len(graph.nodes))

            if makespan > 0 and num_buckets > 0:
                bucket_width = makespan / num_buckets
                buckets: Dict[int, List[str]] = defaultdict(list)
                for node in graph.nodes:
                    midpoint = (node.start_ns + node.end_ns) / 2.0
                    idx = int((midpoint - min_t) / bucket_width)
                    idx = min(idx, num_buckets - 1)
                    buckets[idx].append(_node_ref(node.id))

                for idx in sorted(buckets):
                    refs = buckets[idx]
                    if len(refs) >= 2:
                        joined = "; ".join(refs)
                        lines.append(f"  {{ rank=same; {joined}; }}")

                lines.append("")

        # --- Edges ------------------------------------------------------------
        for edge in graph.edges:
            src_ref = _node_ref(edge.src)
            dst_ref = _node_ref(edge.dst)
            src_node = node_map.get(edge.src)
            dst_node = node_map.get(edge.dst)

            label = _escape_dot(edge.variable)

            if (
                src_node is not None
                and dst_node is not None
                and src_node.stream != dst_node.stream
            ):
                # Cross-stream: red dashed
                lines.append(
                    f'  {src_ref} -> {dst_ref} '
                    f'[label="{label}", color=red, style=dashed, penwidth=2.0];'
                )
            else:
                # Same stream: default black solid
                lines.append(
                    f'  {src_ref} -> {dst_ref} [label="{label}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def export(self, graph: DataFlowGraph, output_path: str) -> None:
        """Write the DOT representation of *graph* to *output_path*."""
        dot = self.render(graph)
        with open(output_path, "w") as f:
            f.write(dot)
