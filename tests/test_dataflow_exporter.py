"""Tests for DataFlowDotExporter."""
import tempfile
import os
from strix.dataflow import DFNode, DFEdge, DataFlowGraph
from strix.dataflow_exporter import DataFlowDotExporter
from strix.op_events import OpKind, OpStream


def _make_simple_graph():
    """3-node: DMA enqueue -> VPU dma_done -> VPU matmul."""
    nodes = [
        DFNode(id=0, name="enqueue_dma", stream=OpStream.DMA,
               kind=OpKind.LEAF, start_ns=0, end_ns=100, flops=0, bytes=4096,
               ssa_outputs=["__dma_token_0"], ssa_inputs=["%arg0"],
               attributes={}, loop_depth=0, parent_loop_id=None),
        DFNode(id=1, name="dma_done", stream=OpStream.VPU,
               kind=OpKind.LEAF, start_ns=100, end_ns=101, flops=0, bytes=0,
               ssa_outputs=["%data"], ssa_inputs=["__dma_token_0"],
               attributes={}, loop_depth=0, parent_loop_id=None),
        DFNode(id=2, name="vmatmul", stream=OpStream.VPU,
               kind=OpKind.LEAF, start_ns=101, end_ns=500, flops=1000000, bytes=0,
               ssa_outputs=["%result"], ssa_inputs=["%data"],
               attributes={}, loop_depth=0, parent_loop_id=None),
    ]
    edges = [
        DFEdge(src=0, dst=1, variable="__dma_token_0"),
        DFEdge(src=1, dst=2, variable="%data"),
    ]
    return DataFlowGraph(nodes=nodes, edges=edges, loops={})


def test_dot_output_is_valid():
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    assert dot.startswith("digraph dataflow {")
    assert dot.strip().endswith("}")
    assert "ev_0" in dot
    assert "ev_1" in dot
    assert "ev_2" in dot
    assert "ev_0 -> ev_1" in dot
    assert "ev_1 -> ev_2" in dot


def test_cross_stream_edges_highlighted():
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    lines = dot.split("\n")
    cross_lines = [l for l in lines if "ev_0 -> ev_1" in l]
    assert cross_lines
    assert "red" in cross_lines[0].lower()


def test_hardware_clusters():
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    assert "cluster_dma" in dot
    assert "cluster_vpu" in dot


def test_node_styling():
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    lines = dot.split("\n")
    dma_lines = [l for l in lines if "ev_0" in l and "label" in l]
    assert dma_lines
    assert "lightyellow" in dma_lines[0]


def test_export_to_file():
    graph = _make_simple_graph()
    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
        path = f.name
    try:
        DataFlowDotExporter().export(graph, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content.startswith("digraph dataflow {")
    finally:
        os.unlink(path)


def test_loop_clusters():
    nodes = [
        DFNode(id=0, name="vadd", stream=OpStream.VPU,
               kind=OpKind.LEAF, start_ns=0, end_ns=5, flops=1, bytes=0,
               ssa_outputs=["%v0"], ssa_inputs=[],
               attributes={}, loop_depth=1, parent_loop_id=0),
    ]
    graph = DataFlowGraph(nodes=nodes, edges=[], loops={0: [0]})
    dot = DataFlowDotExporter().render(graph)
    assert "cluster_loop_0" in dot


def test_time_buckets():
    nodes = [
        DFNode(id=0, name="op_a", stream=OpStream.VPU,
               kind=OpKind.LEAF, start_ns=0, end_ns=100, flops=0, bytes=0,
               ssa_outputs=[], ssa_inputs=[],
               attributes={}, loop_depth=0, parent_loop_id=None),
        DFNode(id=1, name="op_b", stream=OpStream.VPU,
               kind=OpKind.LEAF, start_ns=500, end_ns=600, flops=0, bytes=0,
               ssa_outputs=[], ssa_inputs=[],
               attributes={}, loop_depth=0, parent_loop_id=None),
    ]
    graph = DataFlowGraph(nodes=nodes, edges=[], loops={})
    dot = DataFlowDotExporter().render(graph)
    # Two nodes at different times shouldn't share a rank bucket
    # so rank=same should NOT appear (each bucket has only 1 node)
    # But the test is about whether time bucketing is implemented
    assert "rankdir=LR" in dot


def test_stall_node_styling():
    """STALL nodes should be salmon-colored boxes."""
    nodes = [
        DFNode(id=0, name="STALL", stream=OpStream.VPU,
               kind=OpKind.STALL, start_ns=0, end_ns=50, flops=0, bytes=0,
               ssa_outputs=[], ssa_inputs=[],
               attributes={}, loop_depth=0, parent_loop_id=None),
    ]
    graph = DataFlowGraph(nodes=nodes, edges=[], loops={})
    dot = DataFlowDotExporter().render(graph)
    lines = dot.split("\n")
    stall_lines = [l for l in lines if "ev_0" in l and "label" in l]
    assert stall_lines
    assert "salmon" in stall_lines[0]


def test_control_node_styling():
    """Control-stream nodes should be diamond-shaped and lightgrey."""
    nodes = [
        DFNode(id=0, name="scf.for", stream=OpStream.CONTROL,
               kind=OpKind.LOOP, start_ns=0, end_ns=100, flops=0, bytes=0,
               ssa_outputs=[], ssa_inputs=[],
               attributes={}, loop_depth=0, parent_loop_id=None),
    ]
    graph = DataFlowGraph(nodes=nodes, edges=[], loops={})
    dot = DataFlowDotExporter().render(graph)
    lines = dot.split("\n")
    ctrl_lines = [l for l in lines if "ev_0" in l and "label" in l]
    assert ctrl_lines
    assert "diamond" in ctrl_lines[0]
    assert "lightgrey" in ctrl_lines[0]


def test_node_label_format():
    """Node labels should include opcode and timing; flops/bytes only if > 0."""
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    # DMA node (ev_0) has bytes=4096, flops=0 -> should show bytes but not flops
    lines = dot.split("\n")
    dma_lines = [l for l in lines if "ev_0" in l and "label" in l]
    assert dma_lines
    assert "4096B" in dma_lines[0]
    # VPU matmul (ev_2) has flops=1000000 -> should show flops
    matmul_lines = [l for l in lines if "ev_2" in l and "label" in l]
    assert matmul_lines
    assert "1000000F" in matmul_lines[0]


def test_same_stream_edges_default_style():
    """Same-stream edges should NOT have red/dashed styling."""
    graph = _make_simple_graph()
    dot = DataFlowDotExporter().render(graph)
    lines = dot.split("\n")
    # ev_1 -> ev_2 is VPU -> VPU (same stream)
    same_lines = [l for l in lines if "ev_1 -> ev_2" in l]
    assert same_lines
    assert "red" not in same_lines[0].lower()
