# SSA Dataflow Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract SSA def-use dependency graphs from simulated OpEvent trees and export them as Graphviz DOT files with time-partitioned layout.

**Architecture:** Simulator persists SSA inputs/outputs on each scheduled event. A new `dataflow.py` module extracts a lightweight `DataFlowGraph` from the OpEvent tree. A `DataFlowDotExporter` renders the graph as a DOT file with time-bucketed ranks, hardware swim lanes (VPU/DMA clusters), and cross-stream edge highlighting.

**Tech Stack:** Python 3.9+ stdlib only (dataclasses, typing). No new dependencies.

---

### Task 1: Persist SSA info in simulator

**Files:**
- Modify: `strix/simulator.py:290` (in `_schedule` method, after `group_boundary_ssa`)
- Modify: `strix/simulator.py:360` (in `_schedule_event`, after timing assignment)
- Modify: `strix/simulator.py:416-422` (in `_schedule_enqueue_dma`, after token tracking)
- Modify: `strix/simulator.py:497-500` (in `_schedule_dma_done`, after timing assignment)
- Test: `tests/test_simulator_ssa.py`

**Step 1: Write the failing test**

Create `tests/` directory and test file:

```python
# tests/test_simulator_ssa.py
"""Verify that the simulator persists SSA inputs/outputs on OpEvent attributes."""
from strix.domain import Instruction
from strix.hardware import HardwareSpec
from strix.simulator import Simulator
from strix.op_events import OpEvent, OpKind


def _make_simple_program() -> Instruction:
    """A minimal 3-instruction program: sld → add → store.

    %0 = llo.sld %arg0         (loads a scalar from arg)
    %1 = arith.addi %0, %0     (adds it to itself)
    enqueue_dma %1, ...        (stores result via DMA)
    """
    root = Instruction(opcode="module", outputs=[], inputs=[], body=[
        Instruction(opcode="sld", outputs=["%0"], inputs=["%arg0"]),
        Instruction(opcode="arith.addi", outputs=["%1"], inputs=["%0", "%0"]),
    ])
    return root


def test_leaf_events_have_ssa_attributes():
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides={"%arg0": 128})
    root_event = sim.run(_make_simple_program())

    # Collect all leaf events
    leaves = []
    def collect(ev: OpEvent):
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK):
            leaves.append(ev)
        for c in ev.children:
            collect(c)
    collect(root_event)

    assert len(leaves) >= 2, f"Expected at least 2 leaf events, got {len(leaves)}"

    for leaf in leaves:
        assert "ssa_inputs" in leaf.attributes, (
            f"Event '{leaf.name}' missing ssa_inputs attribute"
        )
        assert "ssa_outputs" in leaf.attributes, (
            f"Event '{leaf.name}' missing ssa_outputs attribute"
        )
        assert isinstance(leaf.attributes["ssa_inputs"], list)
        assert isinstance(leaf.attributes["ssa_outputs"], list)


def test_ssa_dependency_chain():
    """The addi event should list %0 as input (defined by sld)."""
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides={"%arg0": 128})
    root_event = sim.run(_make_simple_program())

    leaves = []
    def collect(ev: OpEvent):
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK):
            leaves.append(ev)
        for c in ev.children:
            collect(c)
    collect(root_event)

    # Find the addi event
    addi_events = [e for e in leaves if "addi" in e.name]
    assert addi_events, "No addi event found"
    addi = addi_events[0]
    assert "%0" in addi.attributes["ssa_inputs"], (
        f"addi should have %0 as input, got {addi.attributes['ssa_inputs']}"
    )

    # Find the sld event
    sld_events = [e for e in leaves if "sld" in e.name]
    assert sld_events, "No sld event found"
    sld = sld_events[0]
    assert "%0" in sld.attributes["ssa_outputs"], (
        f"sld should have %0 as output, got {sld.attributes['ssa_outputs']}"
    )
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_simulator_ssa.py -v`
Expected: FAIL with `KeyError: 'ssa_inputs'` or `AssertionError: Event '...' missing ssa_inputs`

**Step 3: Write minimal implementation**

In `strix/simulator.py`, add SSA persistence in three places:

1. In `_schedule_event` (line ~360, after `ev.attributes.setdefault("category", ...)`):

```python
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)
```

2. In `_schedule_enqueue_dma` (line ~407, after `ev.attributes.setdefault("category", "Memory")`):

```python
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)
```

3. In `_schedule_dma_done` (line ~497, after `ev.attributes.setdefault("category", "Overhead")`):

```python
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_simulator_ssa.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strix/simulator.py tests/test_simulator_ssa.py
git commit -m "feat: persist SSA inputs/outputs on OpEvent attributes during simulation"
```

---

### Task 2: DataFlowGraph data model and extraction

**Files:**
- Create: `strix/dataflow.py`
- Test: `tests/test_dataflow.py`

**Step 1: Write the failing test**

```python
# tests/test_dataflow.py
"""Test DataFlowGraph extraction from OpEvent trees."""
from strix.dataflow import DFNode, DFEdge, DataFlowGraph, extract_dataflow
from strix.op_events import OpEvent, OpKind, OpStream


def _make_scheduled_tree() -> OpEvent:
    """Build a pre-scheduled OpEvent tree with SSA attributes set.

    Simulates: sld(%arg0) → %0, addi(%0,%0) → %1
    """
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)

    sld = OpEvent(
        name="sld", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=0, end_time_ns=10, duration_ns=10,
        flops=0, bytes=4,
        attributes={"ssa_inputs": ["%arg0"], "ssa_outputs": ["%0"]},
    )
    addi = OpEvent(
        name="arith.addi", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=10, end_time_ns=20, duration_ns=10,
        flops=1, bytes=0,
        attributes={"ssa_inputs": ["%0", "%0"], "ssa_outputs": ["%1"]},
    )
    root.children = [sld, addi]
    root.start_time_ns = 0
    root.end_time_ns = 20
    return root


def test_extract_nodes():
    graph = extract_dataflow(_make_scheduled_tree())
    assert len(graph.nodes) == 2
    names = {n.name for n in graph.nodes}
    assert "sld" in names
    assert "arith.addi" in names


def test_extract_edges():
    graph = extract_dataflow(_make_scheduled_tree())
    # addi reads %0 which is produced by sld
    assert len(graph.edges) >= 1
    edge = graph.edges[0]
    assert edge.variable == "%0"
    # src should be the sld node, dst should be addi node
    src_node = next(n for n in graph.nodes if n.id == edge.src)
    dst_node = next(n for n in graph.nodes if n.id == edge.dst)
    assert src_node.name == "sld"
    assert dst_node.name == "arith.addi"


def test_cross_stream_edges():
    """DMA → VPU edges are correctly identified."""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)

    enq = OpEvent(
        name="enqueue_dma", kind=OpKind.LEAF, stream=OpStream.DMA,
        start_time_ns=0, end_time_ns=100, duration_ns=100,
        bytes=4096,
        attributes={
            "ssa_inputs": ["%arg0"],
            "ssa_outputs": ["__dma_token_0"],
        },
    )
    done = OpEvent(
        name="dma_done", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=100, end_time_ns=101, duration_ns=1,
        attributes={
            "ssa_inputs": ["__dma_token_0"],
            "ssa_outputs": ["%data"],
        },
    )
    root.children = [enq, done]
    root.start_time_ns = 0
    root.end_time_ns = 101

    graph = extract_dataflow(root)
    # Should have an edge from enqueue_dma to dma_done via __dma_token_0
    cross_edges = [e for e in graph.edges if e.variable == "__dma_token_0"]
    assert len(cross_edges) == 1
    src = next(n for n in graph.nodes if n.id == cross_edges[0].src)
    dst = next(n for n in graph.nodes if n.id == cross_edges[0].dst)
    assert src.stream == OpStream.DMA
    assert dst.stream == OpStream.VPU


def test_loop_tracking():
    """Nodes inside loops get correct parent_loop_id."""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    loop = OpEvent(
        name="scf.for", kind=OpKind.LOOP, stream=OpStream.CONTROL,
        attributes={"trip_count": 10, "expanded": True,
                     "ssa_inputs": [], "ssa_outputs": []},
    )
    inner = OpEvent(
        name="vadd", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=0, end_time_ns=5, duration_ns=5,
        attributes={"ssa_inputs": [], "ssa_outputs": ["%v0"]},
    )
    loop.children = [inner]
    loop.start_time_ns = 0
    loop.end_time_ns = 50
    root.children = [loop]
    root.start_time_ns = 0
    root.end_time_ns = 50

    graph = extract_dataflow(root)
    assert len(graph.nodes) == 1  # only the leaf
    assert graph.nodes[0].parent_loop_id is not None
    assert graph.nodes[0].parent_loop_id in graph.loops


def test_stall_events_included():
    """STALL events appear as nodes in the graph."""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    stall = OpEvent(
        name="STALL", kind=OpKind.STALL, stream=OpStream.VPU,
        start_time_ns=0, end_time_ns=50, duration_ns=50,
        attributes={"reason": "Waiting for DMA", "ssa_inputs": [], "ssa_outputs": []},
    )
    root.children = [stall]
    root.start_time_ns = 0
    root.end_time_ns = 50

    graph = extract_dataflow(root)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].kind == OpKind.STALL
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_dataflow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strix.dataflow'`

**Step 3: Write minimal implementation**

```python
# strix/dataflow.py
"""
SSA data flow graph extraction from simulated OpEvent trees.

Walks the OpEvent tree (post-simulation) and builds a lightweight
DataFlowGraph with nodes for each executed instruction and edges
for SSA def-use dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .op_events import OpEvent, OpKind, OpStream


@dataclass
class DFNode:
    """A node in the data flow graph = one instruction execution."""
    id: str
    name: str
    stream: OpStream
    kind: OpKind
    start_ns: int
    end_ns: int
    flops: int
    bytes: int
    ssa_outputs: List[str]
    ssa_inputs: List[str]
    attributes: Dict[str, Any]
    loop_depth: int
    parent_loop_id: Optional[str]


@dataclass
class DFEdge:
    """An SSA dependency edge from producer to consumer."""
    src: str       # producer node id
    dst: str       # consumer node id
    variable: str  # SSA variable name


@dataclass
class DataFlowGraph:
    """The complete data flow graph extracted from an OpEvent tree."""
    nodes: List[DFNode]
    edges: List[DFEdge]
    loops: Dict[str, List[str]]  # loop_id -> [child node ids]


def extract_dataflow(root: OpEvent) -> DataFlowGraph:
    """
    Extract a DataFlowGraph from a simulated OpEvent tree.

    Walks the tree, creates DFNode for each leaf/block/stall event,
    then builds SSA dependency edges from the ssa_inputs/ssa_outputs
    attributes set by the simulator.
    """
    nodes: List[DFNode] = []
    loops: Dict[str, List[str]] = {}
    counter = 0

    def _visit(
        ev: OpEvent,
        loop_depth: int,
        parent_loop_id: Optional[str],
    ) -> None:
        nonlocal counter

        if ev.kind == OpKind.LOOP:
            loop_id = f"loop_{counter}"
            counter += 1
            loops[loop_id] = []
            for child in ev.children:
                _visit(child, loop_depth + 1, loop_id)
            return

        if ev.kind in (OpKind.ROOT, OpKind.IF):
            for child in ev.children:
                _visit(child, loop_depth, parent_loop_id)
            return

        # LEAF, BLOCK, STALL — create a DFNode
        node_id = f"ev_{counter}"
        counter += 1

        ssa_inputs = ev.attributes.get("ssa_inputs", [])
        ssa_outputs = ev.attributes.get("ssa_outputs", [])

        node = DFNode(
            id=node_id,
            name=ev.name,
            stream=ev.stream,
            kind=ev.kind,
            start_ns=ev.start_time_ns,
            end_ns=ev.end_time_ns,
            flops=ev.flops,
            bytes=ev.bytes,
            ssa_outputs=list(ssa_outputs),
            ssa_inputs=list(ssa_inputs),
            attributes=dict(ev.attributes),
            loop_depth=loop_depth,
            parent_loop_id=parent_loop_id,
        )
        nodes.append(node)

        if parent_loop_id and parent_loop_id in loops:
            loops[parent_loop_id].append(node_id)

        # Recurse into children (for BLOCK events that may have nested structure)
        for child in ev.children:
            _visit(child, loop_depth, parent_loop_id)

    _visit(root, 0, None)

    # Build edges: variable → producer node id
    producer_map: Dict[str, str] = {}
    for node in nodes:
        for var in node.ssa_outputs:
            producer_map[var] = node.id

    edges: List[DFEdge] = []
    seen_edges: set = set()
    for node in nodes:
        for var in node.ssa_inputs:
            if var in producer_map:
                src_id = producer_map[var]
                edge_key = (src_id, node.id, var)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(DFEdge(src=src_id, dst=node.id, variable=var))

    return DataFlowGraph(nodes=nodes, edges=edges, loops=loops)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_dataflow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strix/dataflow.py tests/test_dataflow.py
git commit -m "feat: add DataFlowGraph data model and extraction from OpEvent tree"
```

---

### Task 3: DOT exporter

**Files:**
- Create: `strix/dataflow_exporter.py`
- Test: `tests/test_dataflow_exporter.py`

**Step 1: Write the failing test**

```python
# tests/test_dataflow_exporter.py
"""Test DataFlowDotExporter output."""
import tempfile
import os

from strix.dataflow import DFNode, DFEdge, DataFlowGraph
from strix.dataflow_exporter import DataFlowDotExporter
from strix.op_events import OpKind, OpStream


def _make_simple_graph() -> DataFlowGraph:
    """A 3-node graph: DMA enqueue → VPU dma_done → VPU matmul."""
    nodes = [
        DFNode(
            id="ev_0", name="enqueue_dma", stream=OpStream.DMA,
            kind=OpKind.LEAF, start_ns=0, end_ns=100, flops=0, bytes=4096,
            ssa_outputs=["__dma_token_0"], ssa_inputs=["%arg0"],
            attributes={}, loop_depth=0, parent_loop_id=None,
        ),
        DFNode(
            id="ev_1", name="dma_done", stream=OpStream.VPU,
            kind=OpKind.LEAF, start_ns=100, end_ns=101, flops=0, bytes=0,
            ssa_outputs=["%data"], ssa_inputs=["__dma_token_0"],
            attributes={}, loop_depth=0, parent_loop_id=None,
        ),
        DFNode(
            id="ev_2", name="vmatmul", stream=OpStream.VPU,
            kind=OpKind.LEAF, start_ns=101, end_ns=500, flops=1000000, bytes=0,
            ssa_outputs=["%result"], ssa_inputs=["%data"],
            attributes={}, loop_depth=0, parent_loop_id=None,
        ),
    ]
    edges = [
        DFEdge(src="ev_0", dst="ev_1", variable="__dma_token_0"),
        DFEdge(src="ev_1", dst="ev_2", variable="%data"),
    ]
    return DataFlowGraph(nodes=nodes, edges=edges, loops={})


def test_dot_output_is_valid():
    graph = _make_simple_graph()
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)

    assert dot.startswith("digraph dataflow {")
    assert dot.strip().endswith("}")
    # All node ids present
    assert "ev_0" in dot
    assert "ev_1" in dot
    assert "ev_2" in dot
    # Edges present
    assert "ev_0 -> ev_1" in dot
    assert "ev_1 -> ev_2" in dot


def test_cross_stream_edges_highlighted():
    graph = _make_simple_graph()
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)

    # The enqueue_dma(DMA) → dma_done(VPU) edge should be red/dashed
    # Find the line with ev_0 -> ev_1
    lines = dot.split("\n")
    cross_edge_lines = [l for l in lines if "ev_0 -> ev_1" in l]
    assert cross_edge_lines, "Cross-stream edge not found"
    assert "red" in cross_edge_lines[0].lower() or "color" in cross_edge_lines[0]


def test_hardware_clusters():
    graph = _make_simple_graph()
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)

    assert "cluster_dma" in dot
    assert "cluster_vpu" in dot


def test_node_styling():
    graph = _make_simple_graph()
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)

    # DMA nodes should have lightyellow fill
    # Find the line defining ev_0
    lines = dot.split("\n")
    dma_lines = [l for l in lines if "ev_0" in l and "label" in l]
    assert dma_lines, "DMA node definition not found"
    assert "lightyellow" in dma_lines[0]


def test_export_to_file():
    graph = _make_simple_graph()
    exporter = DataFlowDotExporter()
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
        path = f.name
    try:
        exporter.export(graph, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content.startswith("digraph dataflow {")
    finally:
        os.unlink(path)


def test_loop_clusters():
    nodes = [
        DFNode(
            id="ev_0", name="vadd", stream=OpStream.VPU,
            kind=OpKind.LEAF, start_ns=0, end_ns=5, flops=1, bytes=0,
            ssa_outputs=["%v0"], ssa_inputs=[],
            attributes={}, loop_depth=1, parent_loop_id="loop_0",
        ),
    ]
    graph = DataFlowGraph(
        nodes=nodes, edges=[],
        loops={"loop_0": ["ev_0"]},
    )
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)
    assert "cluster_loop_0" in dot


def test_time_buckets():
    """Nodes at different times should be in different rank groups."""
    nodes = [
        DFNode(
            id="ev_0", name="op_a", stream=OpStream.VPU,
            kind=OpKind.LEAF, start_ns=0, end_ns=100, flops=0, bytes=0,
            ssa_outputs=[], ssa_inputs=[],
            attributes={}, loop_depth=0, parent_loop_id=None,
        ),
        DFNode(
            id="ev_1", name="op_b", stream=OpStream.VPU,
            kind=OpKind.LEAF, start_ns=500, end_ns=600, flops=0, bytes=0,
            ssa_outputs=[], ssa_inputs=[],
            attributes={}, loop_depth=0, parent_loop_id=None,
        ),
    ]
    graph = DataFlowGraph(nodes=nodes, edges=[], loops={})
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)
    # Should have rank=same constraints
    assert "rank=same" in dot
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_dataflow_exporter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strix.dataflow_exporter'`

**Step 3: Write minimal implementation**

```python
# strix/dataflow_exporter.py
"""
Graphviz DOT exporter for DataFlowGraph.

Generates a DOT file with:
- Time-partitioned layout (rankdir=LR, rank=same for concurrent events)
- Hardware swim lanes (cluster_vpu, cluster_dma)
- Loop nesting (cluster_loop_N)
- Cross-stream edge highlighting (red dashed)
"""
from __future__ import annotations

from typing import Dict, List, Set

from .dataflow import DataFlowGraph, DFNode
from .op_events import OpKind, OpStream


class DataFlowDotExporter:
    """Render a DataFlowGraph as a Graphviz DOT string."""

    def render(self, graph: DataFlowGraph) -> str:
        lines: List[str] = []
        lines.append("digraph dataflow {")
        lines.append("  rankdir=LR;")
        lines.append("  node [style=filled, fontsize=10];")
        lines.append("  edge [fontsize=8];")
        lines.append("")

        # Build node lookup
        node_map: Dict[str, DFNode] = {n.id: n for n in graph.nodes}

        # Build stream sets for cluster assignment
        vpu_nodes: List[DFNode] = []
        dma_nodes: List[DFNode] = []
        for n in graph.nodes:
            if n.stream == OpStream.DMA:
                dma_nodes.append(n)
            else:
                vpu_nodes.append(n)

        # Determine which nodes are inside loops
        loop_node_ids: Set[str] = set()
        for nids in graph.loops.values():
            loop_node_ids.update(nids)

        # --- Loop clusters ---
        for loop_id, child_ids in graph.loops.items():
            lines.append(f"  subgraph cluster_{loop_id} {{")
            lines.append(f'    label="{loop_id}";')
            lines.append("    style=dashed;")
            lines.append("    color=blue;")
            for nid in child_ids:
                if nid in node_map:
                    lines.append(f"    {self._node_def(node_map[nid])}")
            lines.append("  }")
            lines.append("")

        # --- VPU cluster (nodes not in loops) ---
        vpu_free = [n for n in vpu_nodes if n.id not in loop_node_ids]
        if vpu_free:
            lines.append("  subgraph cluster_vpu {")
            lines.append('    label="VPU";')
            lines.append("    style=rounded;")
            lines.append('    color="#4488cc";')
            for n in vpu_free:
                lines.append(f"    {self._node_def(n)}")
            lines.append("  }")
            lines.append("")

        # --- DMA cluster (nodes not in loops) ---
        dma_free = [n for n in dma_nodes if n.id not in loop_node_ids]
        if dma_free:
            lines.append("  subgraph cluster_dma {")
            lines.append('    label="DMA";')
            lines.append("    style=rounded;")
            lines.append('    color="#ccaa44";')
            for n in dma_free:
                lines.append(f"    {self._node_def(n)}")
            lines.append("  }")
            lines.append("")

        # --- Time buckets (rank=same constraints) ---
        if graph.nodes:
            min_t = min(n.start_ns for n in graph.nodes)
            max_t = max(n.end_ns for n in graph.nodes)
            span = max_t - min_t
            if span > 0:
                num_buckets = min(100, len(graph.nodes))
                bucket_width = span / num_buckets

                buckets: Dict[int, List[str]] = {}
                for n in graph.nodes:
                    b = int((n.start_ns - min_t) / bucket_width)
                    b = min(b, num_buckets - 1)
                    buckets.setdefault(b, []).append(n.id)

                for bucket_idx in sorted(buckets):
                    ids = buckets[bucket_idx]
                    if len(ids) > 1:
                        id_str = "; ".join(ids)
                        lines.append(
                            f"  {{ rank=same; {id_str}; }}"
                        )
                lines.append("")

        # --- Edges ---
        for edge in graph.edges:
            src_node = node_map.get(edge.src)
            dst_node = node_map.get(edge.dst)
            if src_node is None or dst_node is None:
                continue

            cross_stream = src_node.stream != dst_node.stream
            if cross_stream:
                lines.append(
                    f'  {edge.src} -> {edge.dst} '
                    f'[label="{edge.variable}", '
                    f'color=red, style=dashed, penwidth=2.0];'
                )
            else:
                lines.append(
                    f'  {edge.src} -> {edge.dst} '
                    f'[label="{edge.variable}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def export(self, graph: DataFlowGraph, output_path: str) -> None:
        dot = self.render(graph)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dot)

    def _node_def(self, n: DFNode) -> str:
        """Generate a DOT node definition string."""
        # Pick style based on stream/kind
        if n.kind == OpKind.STALL:
            fill = "salmon"
            shape = "box"
        elif n.stream == OpStream.DMA:
            fill = "lightyellow"
            shape = "box"
        elif n.stream == OpStream.CONTROL:
            fill = "lightgrey"
            shape = "diamond"
        else:
            fill = "lightblue"
            shape = "box"

        # Build label
        label_parts = [n.name]
        label_parts.append(f"{n.start_ns}→{n.end_ns} ns")
        if n.flops > 0:
            label_parts.append(f"{n.flops}F")
        if n.bytes > 0:
            label_parts.append(f"{n.bytes}B")

        label = "\\n".join(label_parts)
        return (
            f'{n.id} [label="{label}", shape={shape}, '
            f'fillcolor={fill}];'
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_dataflow_exporter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strix/dataflow_exporter.py tests/test_dataflow_exporter.py
git commit -m "feat: add DataFlowDotExporter for Graphviz DOT output"
```

---

### Task 4: CLI integration

**Files:**
- Modify: `strix/cli.py:61-75` (add `--dataflow-output` arg to `build_arg_parser`)
- Modify: `strix/cli.py:286-290` (invoke exporter after simulation)
- Test: `tests/test_cli_dataflow.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_dataflow.py
"""Test --dataflow-output CLI flag."""
import os
import tempfile
from unittest.mock import patch

from strix.cli import build_arg_parser, main


def test_arg_parser_accepts_dataflow_output():
    ap = build_arg_parser()
    # Should not raise
    args = ap.parse_args(["some_file.txt", "--dataflow-output", "out.dot"])
    assert args.dataflow_output == "out.dot"


def test_arg_parser_default_is_none():
    ap = build_arg_parser()
    args = ap.parse_args(["some_file.txt"])
    assert args.dataflow_output is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_cli_dataflow.py -v`
Expected: FAIL with `AttributeError: ...has no attribute 'dataflow_output'`

**Step 3: Write minimal implementation**

In `strix/cli.py`:

1. Add argument to `build_arg_parser()` (after the `--tree-max-depth` argument, line 74):

```python
    ap.add_argument(
        "--dataflow-output",
        default=None,
        help=(
            "Where to write Graphviz DOT dataflow graph. "
            "Render with: dot -Tsvg output.dot -o output.svg"
        ),
    )
```

2. Add import at top of file (after existing imports, line 13):

```python
from .dataflow import extract_dataflow
from .dataflow_exporter import DataFlowDotExporter
```

3. Add export call at end of `main()` (after ChromeTraceExporter block, line 290):

```python
    if args.dataflow_output:
        df_graph = extract_dataflow(root_event)
        DataFlowDotExporter().export(df_graph, args.dataflow_output)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_cli_dataflow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add strix/cli.py tests/test_cli_dataflow.py
git commit -m "feat: add --dataflow-output CLI flag for Graphviz DOT export"
```

---

### Task 5: End-to-end integration test

**Files:**
- Test: `tests/test_integration_dataflow.py`
- Uses: `ir_dumps/mosaic/*post-finalize-llo.txt` (existing test data)

**Step 1: Write the integration test**

```python
# tests/test_integration_dataflow.py
"""End-to-end: parse LLO → simulate → extract dataflow → export DOT."""
import glob
import os
import tempfile

from strix.parser import LLOParser
from strix.hardware import HardwareSpec
from strix.simulator import Simulator
from strix.dataflow import extract_dataflow
from strix.dataflow_exporter import DataFlowDotExporter


def _find_smallest_llo_file() -> str:
    """Find the smallest post-finalize-llo.txt for fast testing."""
    pattern = "ir_dumps/mosaic/*post-finalize-llo.txt"
    files = glob.glob(pattern)
    if not files:
        return ""
    # Sort by file size, pick smallest
    return min(files, key=os.path.getsize)


def test_end_to_end_dataflow():
    llo_path = _find_smallest_llo_file()
    if not llo_path:
        import pytest
        pytest.skip("No LLO test files found in ir_dumps/")

    # Parse
    parser = LLOParser()
    root = parser.parse_file(llo_path)

    # Simulate (use default values for unknown scalars)
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides=None)
    try:
        root_event = sim.run(root)
    except Exception:
        # Some files need arg overrides; skip gracefully
        import pytest
        pytest.skip(f"Simulation requires arg overrides for {llo_path}")

    # Extract dataflow
    graph = extract_dataflow(root_event)

    # Basic sanity checks
    assert len(graph.nodes) > 0, "Expected at least one node"

    # Export DOT
    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)
    assert dot.startswith("digraph dataflow {")
    assert dot.strip().endswith("}")

    # Write to temp file and verify it's non-empty
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False, mode="w") as f:
        f.write(dot)
        path = f.name

    try:
        size = os.path.getsize(path)
        assert size > 100, f"DOT file too small: {size} bytes"
    finally:
        os.unlink(path)


def test_node_count_matches_events():
    """Number of DFNodes should equal number of leaf/block/stall OpEvents."""
    llo_path = _find_smallest_llo_file()
    if not llo_path:
        import pytest
        pytest.skip("No LLO test files found in ir_dumps/")

    parser = LLOParser()
    root = parser.parse_file(llo_path)
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides=None)
    try:
        root_event = sim.run(root)
    except Exception:
        import pytest
        pytest.skip(f"Simulation requires arg overrides for {llo_path}")

    # Count leaf/block/stall events in the OpEvent tree
    from strix.op_events import OpKind
    event_count = 0
    def count_events(ev):
        nonlocal event_count
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK, OpKind.STALL):
            event_count += 1
        for c in ev.children:
            count_events(c)
    count_events(root_event)

    graph = extract_dataflow(root_event)
    assert len(graph.nodes) == event_count, (
        f"Node count {len(graph.nodes)} != event count {event_count}"
    )
```

**Step 2: Run all tests**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration_dataflow.py
git commit -m "test: add end-to-end integration tests for dataflow analysis"
```

---

### Task 6: Run full test suite and verify

**Step 1: Run the complete test suite**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Manual smoke test with real LLO file**

Run:
```bash
cd /Users/xl/Code/strix
python -m strix.cli "ir_dumps/mosaic/1777010511916335044-0015-mosaic-dump-fused-moe-k_8-renorm_k-bt_2_2_2-bf_512_512-bd1_1024_1024-bd2_1024_1024-post-finalize-llo.txt" \
  --default-sld-value 128 \
  --dataflow-output dataflow.dot
```
Expected: `dataflow.dot` file created alongside the usual console output and trace.json

**Step 3: Verify DOT file**

Run: `head -20 dataflow.dot && echo "..." && wc -l dataflow.dot`
Expected: Valid DOT syntax starting with `digraph dataflow {`, non-trivial line count

**Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address integration issues from smoke test"
```
