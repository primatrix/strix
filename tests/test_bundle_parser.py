"""Tests for BundleParser."""
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
