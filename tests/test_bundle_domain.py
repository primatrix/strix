"""Tests for Bundle data model: SourceLoc, BundleInstruction, Bundle, BundleProgram."""
from strix.bundle_domain import BundleInstruction, Bundle, BundleProgram, SourceLoc


class TestSourceLoc:
    def test_single_line_loc(self):
        loc = SourceLoc(
            file="/workspace/src/kernel.py",
            start_line=587,
            start_col=14,
            end_line=587,
            end_col=42,
        )
        assert loc.file == "/workspace/src/kernel.py"
        assert loc.start_line == 587
        assert loc.end_line == 587

    def test_cross_line_loc(self):
        loc = SourceLoc(
            file="/workspace/src/kernel.py",
            start_line=665,
            start_col=12,
            end_line=669,
            end_col=13,
        )
        assert loc.start_line == 665
        assert loc.end_line == 669

    def test_frozen_hashable(self):
        """SourceLoc must be hashable for use as dict key in source_index."""
        loc1 = SourceLoc("f.py", 1, 2, 1, 10)
        loc2 = SourceLoc("f.py", 1, 2, 1, 10)
        assert loc1 == loc2
        assert hash(loc1) == hash(loc2)
        d = {loc1: "value"}
        assert d[loc2] == "value"

    def test_different_locs_not_equal(self):
        loc1 = SourceLoc("f.py", 1, 2, 1, 10)
        loc2 = SourceLoc("f.py", 1, 2, 1, 11)
        assert loc1 != loc2


class TestBundleInstruction:
    def test_with_loc(self):
        loc = SourceLoc("k.py", 684, 10, 684, 34)
        instr = BundleInstruction(
            opcode="sshll.u32",
            raw_text='%s147 = sshll.u32 %s30_s17, 7 /* loc("k.py":684:10 to :34) */',
            outputs=["%s147"],
            loc=loc,
        )
        assert instr.opcode == "sshll.u32"
        assert instr.outputs == ["%s147"]
        assert instr.loc is not None
        assert instr.loc.start_line == 684

    def test_without_loc(self):
        instr = BundleInstruction(
            opcode="sfence",
            raw_text="%1 = sfence",
            outputs=["%1"],
            loc=None,
        )
        assert instr.loc is None

    def test_no_outputs(self):
        instr = BundleInstruction(
            opcode="vdelay",
            raw_text="%0 = vdelay 1",
            outputs=["%0"],
            loc=None,
        )
        assert instr.outputs == ["%0"]


class TestBundle:
    def test_simple_bundle(self):
        instr = BundleInstruction("sfence", "%1 = sfence", ["%1"], None)
        bundle = Bundle(
            address=1,
            control_flags=[],
            nesting_depth=0,
            instructions=[instr],
            comments=[],
            is_empty=False,
        )
        assert bundle.address == 1
        assert len(bundle.instructions) == 1
        assert not bundle.is_empty

    def test_empty_bundle(self):
        bundle = Bundle(
            address=5,
            control_flags=[],
            nesting_depth=0,
            instructions=[],
            comments=[],
            is_empty=True,
        )
        assert bundle.is_empty
        assert len(bundle.instructions) == 0

    def test_with_control_flags_and_nesting(self):
        bundle = Bundle(
            address=0x2D4,
            control_flags=["LB"],
            nesting_depth=1,
            instructions=[],
            comments=["Start region 745"],
            is_empty=False,
        )
        assert bundle.control_flags == ["LB"]
        assert bundle.nesting_depth == 1
        assert bundle.address == 0x2D4


class TestBundleProgram:
    def test_empty_program(self):
        prog = BundleProgram(bundles=[], source_index={})
        assert len(prog.bundles) == 0
        assert len(prog.source_index) == 0

    def test_with_source_index(self):
        loc = SourceLoc("k.py", 587, 14, 587, 42)
        prog = BundleProgram(
            bundles=[],
            source_index={loc: [(0x11, 0), (0x12, 0)]},
        )
        assert prog.source_index[loc] == [(0x11, 0), (0x12, 0)]
