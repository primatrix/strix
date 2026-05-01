"""Verify that the simulator persists SSA inputs/outputs on OpEvent attributes."""
from strix.domain import Instruction
from strix.hardware import HardwareSpec
from strix.simulator import Simulator
from strix.op_events import OpEvent, OpKind


def _make_simple_program():
    """A minimal 2-instruction program: sld -> addi."""
    root = Instruction(opcode="module", outputs=[], inputs=[], body=[
        Instruction(opcode="sld", outputs=["%0"], inputs=["%arg0"]),
        Instruction(opcode="arith.addi", outputs=["%1"], inputs=["%0", "%0"]),
    ])
    return root


def test_leaf_events_have_ssa_attributes():
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides={"%arg0": 128})
    root_event = sim.run(_make_simple_program())

    leaves = []
    def collect(ev):
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK):
            leaves.append(ev)
        for c in ev.children:
            collect(c)
    collect(root_event)

    assert len(leaves) >= 2
    for leaf in leaves:
        assert "ssa_inputs" in leaf.attributes, f"'{leaf.name}' missing ssa_inputs"
        assert "ssa_outputs" in leaf.attributes, f"'{leaf.name}' missing ssa_outputs"
        assert isinstance(leaf.attributes["ssa_inputs"], list)
        assert isinstance(leaf.attributes["ssa_outputs"], list)


def test_ssa_dependency_chain():
    spec = HardwareSpec()
    sim = Simulator(spec, arg_overrides={"%arg0": 128})
    root_event = sim.run(_make_simple_program())

    leaves = []
    def collect(ev):
        if ev.kind in (OpKind.LEAF, OpKind.BLOCK):
            leaves.append(ev)
        for c in ev.children:
            collect(c)
    collect(root_event)

    addi_events = [e for e in leaves if "addi" in e.name]
    assert addi_events
    addi = addi_events[0]
    assert "%0" in addi.attributes["ssa_inputs"]

    sld_events = [e for e in leaves if "sld" in e.name]
    assert sld_events
    sld = sld_events[0]
    assert "%0" in sld.attributes["ssa_outputs"]
