"""DataFlowGraph data model and extraction from OpEvent trees.

Builds a directed graph of SSA data dependencies by walking the
simulated OpEvent tree produced by the simulator.  Each leaf / block /
stall event becomes a DFNode; SSA variable names carried in
``ev.attributes["ssa_inputs"]`` / ``ev.attributes["ssa_outputs"]``
determine the DFEdge connections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .op_events import OpEvent, OpKind, OpStream


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DFNode:
    """One instruction execution in the dataflow graph."""

    id: int
    name: str
    stream: OpStream
    kind: OpKind
    start_ns: int = 0
    end_ns: int = 0
    flops: int = 0
    bytes: int = 0
    ssa_outputs: List[str] = field(default_factory=list)
    ssa_inputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    loop_depth: int = 0
    parent_loop_id: Optional[int] = None


@dataclass
class DFEdge:
    """SSA dependency edge from producer to consumer."""

    src: int
    dst: int
    variable: str


@dataclass
class DataFlowGraph:
    """Container for the extracted dataflow graph."""

    nodes: List[DFNode] = field(default_factory=list)
    edges: List[DFEdge] = field(default_factory=list)
    loops: Dict[int, List[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_NODE_KINDS = {OpKind.LEAF, OpKind.BLOCK, OpKind.STALL}


def extract_dataflow(root: OpEvent) -> DataFlowGraph:
    """Walk an OpEvent tree and extract a DataFlowGraph.

    Parameters
    ----------
    root:
        The root OpEvent produced by the simulator.

    Returns
    -------
    DataFlowGraph
        Nodes, edges, and loop metadata extracted from *root*.
    """
    nodes: List[DFNode] = []
    loops: Dict[int, List[int]] = {}
    next_id = 0
    next_loop_id = 0

    def _walk(
        ev: OpEvent,
        loop_depth: int,
        parent_loop_id: Optional[int],
    ) -> None:
        nonlocal next_id, next_loop_id

        node_id: Optional[int] = None
        loop_id: Optional[int] = None

        # Create DFNode for LEAF/BLOCK/STALL/LOOP/IF events.
        if ev.kind in _NODE_KINDS or ev.kind in (OpKind.LOOP, OpKind.IF):
            node_id = next_id
            next_id += 1

            if ev.kind == OpKind.LOOP:
                loop_id = next_loop_id
                next_loop_id += 1
                loops.setdefault(loop_id, [])

            node = DFNode(
                id=node_id,
                name=ev.name,
                stream=ev.stream,
                kind=ev.kind,
                start_ns=ev.start_time_ns,
                end_ns=ev.end_time_ns,
                flops=ev.flops,
                bytes=ev.bytes,
                ssa_outputs=list(ev.attributes.get("ssa_outputs", [])),
                ssa_inputs=list(ev.attributes.get("ssa_inputs", [])),
                attributes=dict(ev.attributes),
                loop_depth=loop_depth,
                parent_loop_id=parent_loop_id,
            )
            nodes.append(node)

            if parent_loop_id is not None:
                loops.setdefault(parent_loop_id, []).append(node_id)
            if loop_id is not None:
                loops[loop_id].append(node_id)

        # LEAF/BLOCK/STALL have no children — stop here.
        if ev.kind in _NODE_KINDS:
            return

        # Recurse into children.
        child_loop_depth = loop_depth + 1 if ev.kind == OpKind.LOOP else loop_depth
        child_loop_id = loop_id if ev.kind == OpKind.LOOP else parent_loop_id

        for child in ev.children:
            _walk(child, child_loop_depth, child_loop_id)

    _walk(root, loop_depth=0, parent_loop_id=None)

    edges = _build_edges(nodes)

    return DataFlowGraph(nodes=nodes, edges=edges, loops=loops)


def _build_edges(nodes: List[DFNode]) -> List[DFEdge]:
    """Build and deduplicate SSA dependency edges.

    For every consumer node that lists a variable in its ``ssa_inputs``,
    find the producer node that has that variable in ``ssa_outputs`` and
    create an edge.  Edges are deduplicated on (src, dst, variable).

    The producer map is built incrementally: each node's inputs are
    resolved against producers seen *so far*, then its outputs update
    the map.  This ensures correct edges in expanded loops where the
    same SSA name is redefined across iterations.
    """
    producer: Dict[str, int] = {}
    seen: Set[Tuple[int, int, str]] = set()
    edges: List[DFEdge] = []

    for node in nodes:
        # Resolve inputs against producers defined before this node.
        for var in node.ssa_inputs:
            src_id = producer.get(var)
            if src_id is None:
                continue  # external input (e.g. %arg0)
            key = (src_id, node.id, var)
            if key not in seen:
                seen.add(key)
                edges.append(DFEdge(src=src_id, dst=node.id, variable=var))

        # Update producer map with this node's outputs.
        for var in node.ssa_outputs:
            producer[var] = node.id

    return edges
