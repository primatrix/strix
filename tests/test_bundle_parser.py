"""Tests for BundleParser."""
import os
import tempfile

from strix.bundle_domain import BundleInstruction, SourceLoc
from strix.bundle_parser import BundleParser


class TestParseLoc:
    def setup_method(self):
        self.parser = BundleParser()

    def test_single_line_loc(self):
        loc = self.parser._parse_loc('"kernel.py":587:14 to :42')
        assert loc == SourceLoc("kernel.py", 587, 14, 587, 42)

    def test_cross_line_loc(self):
        loc = self.parser._parse_loc('"kernel.py":665:12 to 669:13')
        assert loc == SourceLoc("kernel.py", 665, 12, 669, 13)

    def test_full_path(self):
        loc = self.parser._parse_loc(
            '"/workspace/src/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py":684:10 to :34'
        )
        assert loc is not None
        assert loc.file == "/workspace/src/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"
        assert loc.start_line == 684
        assert loc.start_col == 10
        assert loc.end_line == 684
        assert loc.end_col == 34

    def test_invalid_loc_returns_none(self):
        assert self.parser._parse_loc("not a loc") is None

    def test_empty_string_returns_none(self):
        assert self.parser._parse_loc("") is None


class TestParseInstruction:
    def setup_method(self):
        self.parser = BundleParser()

    def test_instruction_with_loc(self):
        text = '%s147 = sshll.u32 %s30_s17, 7 /* loc("kernel.py":684:10 to :34) */'
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "sshll.u32"
        assert instr.outputs == ["%s147"]
        assert instr.loc == SourceLoc("kernel.py", 684, 10, 684, 34)

    def test_instruction_without_loc(self):
        text = "%1 = sfence"
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "sfence"
        assert instr.outputs == ["%1"]
        assert instr.loc is None

    def test_instruction_materialized_constant(self):
        text = "%s4_s0 = smov 0 /* materialized constant */"
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "smov"
        assert instr.loc is None

    def test_instruction_phi_with_multiple_locs(self):
        """Phi instructions can have multiple :: separated loc annotations; take the first."""
        text = (
            '%s17103_s30 = sphi %s17475_s30, %s1203_s30 '
            '/* phi copy :: loc("kernel.py":952:18 to 958:9) '
            ':: loc("kernel.py":952:18 to 958:9) */'
        )
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "sphi"
        assert instr.outputs == ["%s17103_s30"]
        assert instr.loc == SourceLoc("kernel.py", 952, 18, 958, 9)

    def test_instruction_no_output(self):
        """Some instructions like vdelay have outputs parsed from assignment."""
        text = "%0 = vdelay 1"
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "vdelay"
        assert instr.outputs == ["%0"]

    def test_instruction_dma(self):
        text = (
            '%150 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s21164_s4, '
            '/*size_in_granules=*/32, /*vmem=*/%s148_s22, '
            '/*dst_syncflagno=*/[#allocation20 + $0xd] '
            '/* loc("kernel.py":684:10 to :34) */'
        )
        instr = self.parser._parse_instruction(text)
        assert instr.opcode == "dma.hbm_to_vmem"
        assert instr.outputs == ["%150"]
        assert instr.loc is not None

    def test_raw_text_preserved(self):
        text = "%1 = sfence"
        instr = self.parser._parse_instruction(text)
        assert instr.raw_text == text


class TestParseBundleLine:
    def setup_method(self):
        self.parser = BundleParser()

    def test_simple_single_instruction(self):
        line = "   0x1   :  { %1 = sfence }"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.address == 0x1
        assert bundle.control_flags == []
        assert bundle.nesting_depth == 0
        assert len(bundle.instructions) == 1
        assert bundle.instructions[0].opcode == "sfence"
        assert not bundle.is_empty

    def test_address_zero(self):
        line = "     0   :  { %0 = vdelay 1 } /* Start region 0 */"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.address == 0

    def test_empty_bundle(self):
        line = "   0x5   :  { }"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.is_empty
        assert len(bundle.instructions) == 0

    def test_multiple_instructions(self):
        line = (
            '  0x15   :  { %p34_p0 = scmp.lt.s32.totalorder %s33_s22, 0 '
            '/* loc("k.py":587:14 to :42) */  ;;  '
            '%s35_s23 = ssub.s32 0, %s33_s22 '
            '/* loc("k.py":587:14 to :42) */ }'
        )
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.address == 0x15
        assert len(bundle.instructions) == 2
        assert bundle.instructions[0].opcode == "scmp.lt.s32.totalorder"
        assert bundle.instructions[1].opcode == "ssub.s32"

    def test_control_flags_and_nesting(self):
        line = " 0x2d4 LB: > { %s17484_s16 = sshll.u32 %s17103_s30, 7 }"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.address == 0x2D4
        assert bundle.control_flags == ["LB"]
        assert bundle.nesting_depth == 1

    def test_pf_flag_double_nesting(self):
        line = " 0x8e6 PF: >> { %s21403_s7 = smov %s18731_s10 }"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.control_flags == ["PF"]
        assert bundle.nesting_depth == 2

    def test_triple_nesting(self):
        line = ' 0xc26 PF: >>> { %s13779_s14 = sshll.u32 %s17167_s23, 3 }'
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert bundle.nesting_depth == 3

    def test_comments_after_bundle(self):
        line = "     0   :  { %0 = vdelay 1 } /* Start region 0 */"
        bundle = self.parser._parse_bundle_line(line)
        assert bundle is not None
        assert "Start region 0" in bundle.comments

    def test_non_bundle_line_returns_none(self):
        assert self.parser._parse_bundle_line("LH: loop header") is None
        assert self.parser._parse_bundle_line("") is None
        assert self.parser._parse_bundle_line("= control target key start") is None


# --------------- Fixtures for parse_file tests ---------------

SIMPLE_BUNDLES = """\
= control target key start
LH: loop header
LB: loop body
LE: loop exit
PB: predicated region body
PF: predicated region fallthrough
CT: control target
= control target key end

     0   :  { %0 = vdelay 1 } /* Start region 0 */
   0x1   :  { %1 = sfence }
   0x2   :  { %s4_s0 = smov 0 /* materialized constant */ }
   0x3   :  { %2 = sst [smem:[#allocation0]] %s4_s0 } /* End region 0 */
"""

BUNDLES_WITH_LOCS = """\
= control target key start
LH: loop header
LB: loop body
LE: loop exit
PB: predicated region body
PF: predicated region fallthrough
CT: control target
= control target key end

     0   :  { %0 = vdelay 1 }
   0x1   :  { %s30 = sld [smem:[#a]] /* loc("kernel.py":587:14 to :42) */ }
   0x2   :  { %s31 = sld [smem:[#b]] /* loc("kernel.py":587:14 to :42) */ }
   0x3   :  { %s32 = sadd.s32 %s30, %s31 /* loc("kernel.py":665:12 to 669:13) */ }
"""

MULTILINE_BUNDLE = """\
= control target key start
LH: loop header
LB: loop body
LE: loop exit
PB: predicated region body
PF: predicated region fallthrough
CT: control target
= control target key end

     0   :  { %0 = vdelay 1 }
   0x1   :  { %100 = shalt.err (!%p1) /* BoundsCheck 0 [deref of %s4] for %150 = dma.hbm_to_vmem /* loc("k.py":684:10 to :34) */
hlo: fused-moe
 */ }
   0x2   :  { %1 = sfence }
"""


class TestParseFile:
    def setup_method(self):
        self.parser = BundleParser()

    def _write_temp(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix="_final_bundles.txt")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_simple_file(self):
        path = self._write_temp(SIMPLE_BUNDLES)
        try:
            prog = self.parser.parse_file(path)
            assert len(prog.bundles) == 4
            assert prog.bundles[0].address == 0
            assert prog.bundles[1].address == 1
            assert prog.bundles[2].address == 2
            assert prog.bundles[3].address == 3
        finally:
            os.unlink(path)

    def test_header_skipped(self):
        path = self._write_temp(SIMPLE_BUNDLES)
        try:
            prog = self.parser.parse_file(path)
            # Header lines should not produce bundles
            for b in prog.bundles:
                assert b.address >= 0
        finally:
            os.unlink(path)

    def test_source_index_built(self):
        path = self._write_temp(BUNDLES_WITH_LOCS)
        try:
            prog = self.parser.parse_file(path)
            loc587 = SourceLoc("kernel.py", 587, 14, 587, 42)
            loc665 = SourceLoc("kernel.py", 665, 12, 669, 13)
            assert loc587 in prog.source_index
            assert loc665 in prog.source_index
            # loc587 appears at bundle 0x1 slot 0 and bundle 0x2 slot 0
            slots587 = prog.source_index[loc587]
            assert (0x1, 0) in slots587
            assert (0x2, 0) in slots587
            # loc665 appears at bundle 0x3 slot 0
            assert (0x3, 0) in prog.source_index[loc665]
        finally:
            os.unlink(path)

    def test_multiline_bundle(self):
        path = self._write_temp(MULTILINE_BUNDLE)
        try:
            prog = self.parser.parse_file(path)
            # Should have 3 bundles: vdelay, shalt.err (multi-line), sfence
            assert len(prog.bundles) == 3
            assert prog.bundles[0].address == 0
            assert prog.bundles[1].address == 1
            assert prog.bundles[1].instructions[0].opcode == "shalt.err"
            assert prog.bundles[2].address == 2
        finally:
            os.unlink(path)

    def test_multiline_bundle_loc_extracted(self):
        path = self._write_temp(MULTILINE_BUNDLE)
        try:
            prog = self.parser.parse_file(path)
            shalt_bundle = prog.bundles[1]
            assert shalt_bundle.instructions[0].loc == SourceLoc("k.py", 684, 10, 684, 34)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = self._write_temp(
            "= control target key start\n= control target key end\n"
        )
        try:
            prog = self.parser.parse_file(path)
            assert len(prog.bundles) == 0
            assert len(prog.source_index) == 0
        finally:
            os.unlink(path)
