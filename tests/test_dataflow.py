"""Test DataFlowGraph extraction from OpEvent trees."""
from strix.dataflow import DFNode, DFEdge, DataFlowGraph, extract_dataflow
from strix.op_events import OpEvent, OpKind, OpStream


def _make_scheduled_tree():
    """Pre-scheduled OpEvent tree: sld(%arg0)->%0, addi(%0,%0)->%1"""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    sld = OpEvent(
        name="sld", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=0, end_time_ns=10, duration_ns=10, flops=0, bytes=4,
        attributes={"ssa_inputs": ["%arg0"], "ssa_outputs": ["%0"]},
    )
    addi = OpEvent(
        name="arith.addi", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=10, end_time_ns=20, duration_ns=10, flops=1, bytes=0,
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
    assert len(graph.edges) >= 1
    edge = graph.edges[0]
    assert edge.variable == "%0"
    src_node = next(n for n in graph.nodes if n.id == edge.src)
    dst_node = next(n for n in graph.nodes if n.id == edge.dst)
    assert src_node.name == "sld"
    assert dst_node.name == "arith.addi"


def test_cross_stream_edges():
    """DMA -> VPU edges via DMA token."""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    enq = OpEvent(
        name="enqueue_dma", kind=OpKind.LEAF, stream=OpStream.DMA,
        start_time_ns=0, end_time_ns=100, duration_ns=100, bytes=4096,
        attributes={"ssa_inputs": ["%arg0"], "ssa_outputs": ["__dma_token_0"]},
    )
    done = OpEvent(
        name="dma_done", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=100, end_time_ns=101, duration_ns=1,
        attributes={"ssa_inputs": ["__dma_token_0"], "ssa_outputs": ["%data"]},
    )
    root.children = [enq, done]
    root.start_time_ns = 0
    root.end_time_ns = 101

    graph = extract_dataflow(root)
    cross = [e for e in graph.edges if e.variable == "__dma_token_0"]
    assert len(cross) == 1
    src = next(n for n in graph.nodes if n.id == cross[0].src)
    dst = next(n for n in graph.nodes if n.id == cross[0].dst)
    assert src.stream == OpStream.DMA
    assert dst.stream == OpStream.VPU


def test_loop_tracking():
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    loop = OpEvent(
        name="scf.for", kind=OpKind.LOOP, stream=OpStream.CONTROL,
        attributes={"trip_count": 10, "expanded": True},
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
    # LOOP node (scf.for) + inner vadd = 2 nodes
    assert len(graph.nodes) == 2
    loop_nodes = [n for n in graph.nodes if n.kind == OpKind.LOOP]
    inner_nodes = [n for n in graph.nodes if n.kind == OpKind.LEAF]
    assert len(loop_nodes) == 1
    assert len(inner_nodes) == 1
    assert inner_nodes[0].parent_loop_id is not None
    assert inner_nodes[0].parent_loop_id in graph.loops


def test_stall_events_included():
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


def test_deduplicate_edges():
    """Same variable used twice as input should produce only one edge."""
    root = OpEvent(name="root", kind=OpKind.ROOT, stream=OpStream.CONTROL)
    sld = OpEvent(
        name="sld", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=0, end_time_ns=10, duration_ns=10,
        attributes={"ssa_inputs": [], "ssa_outputs": ["%0"]},
    )
    addi = OpEvent(
        name="arith.addi", kind=OpKind.LEAF, stream=OpStream.VPU,
        start_time_ns=10, end_time_ns=20, duration_ns=10,
        attributes={"ssa_inputs": ["%0", "%0"], "ssa_outputs": ["%1"]},
    )
    root.children = [sld, addi]
    root.start_time_ns = 0
    root.end_time_ns = 20

    graph = extract_dataflow(root)
    edges_for_var = [e for e in graph.edges if e.variable == "%0"]
    assert len(edges_for_var) == 1  # deduplicated
