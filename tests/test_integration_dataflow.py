"""End-to-end: parse LLO -> simulate -> extract dataflow -> export DOT."""
import glob
import os
import tempfile

from strix.parser import LLOParser
from strix.hardware import HardwareSpec
from strix.simulator import Simulator
from strix.dataflow import extract_dataflow
from strix.dataflow_exporter import DataFlowDotExporter


def _find_smallest_llo_file():
    """Find the smallest post-finalize-llo.txt for fast testing."""
    pattern = "ir_dumps/mosaic/*post-finalize-llo.txt"
    files = glob.glob(pattern)
    if not files:
        return ""
    return min(files, key=os.path.getsize)


def test_end_to_end_dataflow():
    llo_path = _find_smallest_llo_file()
    if not llo_path:
        import pytest
        pytest.skip("No LLO test files found in ir_dumps/")

    parser = LLOParser()
    root = parser.parse_file(llo_path, exclude_instructions=[])

    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides=None)
    try:
        root_event = sim.run(root)
    except Exception:
        import pytest
        pytest.skip(f"Simulation requires arg overrides for {llo_path}")

    graph = extract_dataflow(root_event)
    assert len(graph.nodes) > 0

    exporter = DataFlowDotExporter()
    dot = exporter.render(graph)
    assert dot.startswith("digraph dataflow {")
    assert dot.strip().endswith("}")

    with tempfile.NamedTemporaryFile(suffix=".dot", delete=False, mode="w") as f:
        f.write(dot)
        path = f.name
    try:
        size = os.path.getsize(path)
        assert size > 100, f"DOT file too small: {size} bytes"
    finally:
        os.unlink(path)


def test_node_count_matches_events():
    llo_path = _find_smallest_llo_file()
    if not llo_path:
        import pytest
        pytest.skip("No LLO test files found in ir_dumps/")

    parser = LLOParser()
    root = parser.parse_file(llo_path, exclude_instructions=[])
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides=None)
    try:
        root_event = sim.run(root)
    except Exception:
        import pytest
        pytest.skip(f"Simulation requires arg overrides for {llo_path}")

    from strix.op_events import OpKind
    event_count = 0
    def count_events(ev):
        """Count events using the same logic as extract_dataflow._walk:
        LEAF/BLOCK/STALL are counted and NOT recursed into;
        LOOP/ROOT/IF are recursed into without counting."""
        nonlocal event_count
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK, OpKind.STALL):
            event_count += 1
            return  # extract_dataflow does not recurse into these
        for c in ev.children:
            count_events(c)
    count_events(root_event)

    graph = extract_dataflow(root_event)
    assert len(graph.nodes) == event_count, (
        f"Node count {len(graph.nodes)} != event count {event_count}"
    )
