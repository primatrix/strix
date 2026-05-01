"""Tests for BundleParser."""
from strix.bundle_domain import SourceLoc
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
