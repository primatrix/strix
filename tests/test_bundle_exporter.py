"""Tests for BundleConsoleExporter and BundleJsonExporter."""
import json
import os
import tempfile
from io import StringIO

from strix.bundle_domain import Bundle, BundleInstruction, BundleProgram, SourceLoc
from strix.bundle_exporter import BundleConsoleExporter


# --------------- Fixtures ---------------

def _make_program() -> BundleProgram:
    """A small BundleProgram with two source locations for testing."""
    loc_587 = SourceLoc("kernel.py", 587, 14, 587, 42)
    loc_684 = SourceLoc("kernel.py", 684, 10, 684, 34)

    bundles = [
        Bundle(
            address=0x00,
            control_flags=[],
            nesting_depth=0,
            instructions=[
                BundleInstruction("vdelay", "%0 = vdelay 1", ["%0"], None),
            ],
            comments=["Start region 0"],
            is_empty=False,
        ),
        Bundle(
            address=0x11,
            control_flags=[],
            nesting_depth=0,
            instructions=[
                BundleInstruction("sld", '%s30 = sld ...', ["%s30"], loc_587),
                BundleInstruction("sld", '%s31 = sld ...', ["%s31"], loc_587),
            ],
            comments=[],
            is_empty=False,
        ),
        Bundle(
            address=0x12,
            control_flags=["LB"],
            nesting_depth=1,
            instructions=[
                BundleInstruction("sshll.u32", '%s147 = sshll.u32 ...', ["%s147"], loc_684),
            ],
            comments=[],
            is_empty=False,
        ),
        Bundle(
            address=0x13,
            control_flags=[],
            nesting_depth=0,
            instructions=[
                BundleInstruction("sadd.s32", '%s148 = sadd.s32 ...', ["%s148"], loc_684),
                BundleInstruction("sshll.u32", '%s149 = sshll.u32 ...', ["%s149"], loc_587),
            ],
            comments=[],
            is_empty=False,
        ),
        Bundle(
            address=0x20,
            control_flags=[],
            nesting_depth=0,
            instructions=[],
            comments=[],
            is_empty=True,
        ),
    ]

    # Build source index manually
    source_index = {
        loc_587: [(0x11, 0), (0x11, 1), (0x13, 1)],
        loc_684: [(0x12, 0), (0x13, 0)],
    }

    return BundleProgram(bundles=bundles, source_index=source_index)


class TestBundleConsoleExporter:
    def test_output_sorted_by_line_number(self):
        """AC1: output sorted by source line number (587 before 684)."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        pos_587 = text.index("L587")
        pos_684 = text.index("L684")
        assert pos_587 < pos_684

    def test_summary_statistics(self):
        """AC5: Summary shows total bundles, annotated instrs, source locations."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        assert "5 bundles" in text
        assert "5 annotated" in text
        assert "2 source locations" in text

    def test_instr_count_and_bundle_count(self):
        """Each loc entry shows instruction count and bundle count."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        # loc_587 has 3 instrs across 2 bundles (0x11, 0x13)
        assert "3 instrs" in text
        assert "2 bundles" in text

    def test_opcode_frequency(self):
        """Each loc entry shows opcode frequency distribution."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        # loc_587: sld x2, sshll.u32 x1
        assert "sld" in text

    def test_line_filter(self):
        """AC3: --line filters to only matching source locations."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out, line_filter=684)
        text = out.getvalue()
        assert "L684" in text
        assert "L587" not in text

    def test_line_filter_no_match(self):
        """--line with no matching locations produces empty mapping."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out, line_filter=999)
        text = out.getvalue()
        assert "0 source locations" in text

    def test_source_root_shows_code(self):
        """AC4: --source-root reads local source and displays code lines."""
        prog = _make_program()

        # Create a temporary source file
        tmpdir = tempfile.mkdtemp()
        src_path = os.path.join(tmpdir, "kernel.py")
        lines = ["\n"] * 700  # pad to line 700
        lines[586] = "gate = pl.load(gate_ref, ...)\n"  # line 587 (0-indexed: 586)
        lines[683] = "result = pl.dot(a, b)\n"  # line 684
        with open(src_path, "w") as f:
            f.writelines(lines)

        try:
            out = StringIO()
            BundleConsoleExporter().export(prog, file=out, source_root=tmpdir)
            text = out.getvalue()
            assert "gate = pl.load(gate_ref, ...)" in text
            assert "result = pl.dot(a, b)" in text
        finally:
            os.unlink(src_path)
            os.rmdir(tmpdir)

    def test_source_root_suffix_match(self):
        """--source-root uses suffix matching for TPU pod paths."""
        loc_pod = SourceLoc(
            "/workspace/src/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py",
            10, 1, 10, 20,
        )
        bundles = [
            Bundle(0x01, [], 0, [
                BundleInstruction("sadd.s32", "...", ["%s1"], loc_pod),
            ], [], False),
        ]
        prog = BundleProgram(bundles=bundles, source_index={loc_pod: [(0x01, 0)]})

        tmpdir = tempfile.mkdtemp()
        # Create nested directory matching the suffix
        nested = os.path.join(tmpdir, "srt", "kernels", "fused_moe", "v1")
        os.makedirs(nested)
        src_path = os.path.join(nested, "kernel.py")
        lines = ["\n"] * 15
        lines[9] = "x = jnp.add(a, b)\n"
        with open(src_path, "w") as f:
            f.writelines(lines)

        try:
            out = StringIO()
            BundleConsoleExporter().export(prog, file=out, source_root=tmpdir)
            text = out.getvalue()
            assert "x = jnp.add(a, b)" in text
        finally:
            os.unlink(src_path)
            for d in [nested,
                      os.path.join(tmpdir, "srt", "kernels", "fused_moe"),
                      os.path.join(tmpdir, "srt", "kernels"),
                      os.path.join(tmpdir, "srt"),
                      tmpdir]:
                os.rmdir(d)

    def test_empty_program(self):
        """Empty program produces summary with zeros."""
        prog = BundleProgram(bundles=[], source_index={})
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        assert "0 bundles" in text
        assert "0 source locations" in text

    def test_header_present(self):
        """Output contains header line."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        assert "Bundle-Source Mapping" in text

    def test_file_grouping(self):
        """Source locations are grouped by file."""
        prog = _make_program()
        out = StringIO()
        BundleConsoleExporter().export(prog, file=out)
        text = out.getvalue()
        assert "kernel.py" in text
