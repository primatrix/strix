"""Tests for the analyze-bundles CLI subcommand."""
import json
import os
import tempfile

from strix.cli import build_arg_parser, main, preprocess_argv


# Fixture: a minimal *-final_bundles.txt
BUNDLES_FIXTURE = """\
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
   0x3   :  { %s32 = sadd.s32 %s30, %s31 /* loc("kernel.py":684:10 to :34) */ }
"""


def _write_fixture() -> str:
    fd, path = tempfile.mkstemp(suffix="_final_bundles.txt")
    with os.fdopen(fd, "w") as f:
        f.write(BUNDLES_FIXTURE)
    return path


class TestAnalyzeBundlesSubcommand:
    def test_basic_output(self, capsys):
        """AC1: basic analyze-bundles prints mapping table."""
        path = _write_fixture()
        try:
            main(["analyze-bundles", path])
            out = capsys.readouterr().out
            assert "Bundle-Source Mapping" in out
            assert "L587" in out
            assert "L684" in out
            assert "Summary:" in out
        finally:
            os.unlink(path)

    def test_sorted_by_line(self, capsys):
        """AC1: output sorted by source line number."""
        path = _write_fixture()
        try:
            main(["analyze-bundles", path])
            out = capsys.readouterr().out
            assert out.index("L587") < out.index("L684")
        finally:
            os.unlink(path)

    def test_json_flag(self):
        """AC2: --json writes structured JSON."""
        bundle_path = _write_fixture()
        fd, json_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            main(["analyze-bundles", bundle_path, "--json", json_path])
            with open(json_path) as f:
                data = json.load(f)
            assert "total_bundles" in data
            assert "annotated_instructions" in data
            assert "mappings" in data
            assert len(data["mappings"]) == 2  # two distinct source locations
        finally:
            os.unlink(bundle_path)
            os.unlink(json_path)

    def test_line_filter(self, capsys):
        """AC3: --line filters to specified line."""
        path = _write_fixture()
        try:
            main(["analyze-bundles", path, "--line", "684"])
            out = capsys.readouterr().out
            assert "L684" in out
            assert "L587" not in out
        finally:
            os.unlink(path)

    def test_source_root(self, capsys):
        """AC4: --source-root shows Pallas source lines."""
        bundle_path = _write_fixture()
        tmpdir = tempfile.mkdtemp()
        src_path = os.path.join(tmpdir, "kernel.py")
        lines = ["\n"] * 700
        lines[586] = "gate = pl.load(gate_ref, ...)\n"
        lines[683] = "result = pl.dot(a, b)\n"
        with open(src_path, "w") as f:
            f.writelines(lines)
        try:
            main(["analyze-bundles", bundle_path, "--source-root", tmpdir])
            out = capsys.readouterr().out
            assert "gate = pl.load(gate_ref, ...)" in out
        finally:
            os.unlink(bundle_path)
            os.unlink(src_path)
            os.rmdir(tmpdir)

    def test_summary_stats(self, capsys):
        """AC5: summary includes total bundles, annotated instrs, source locations."""
        path = _write_fixture()
        try:
            main(["analyze-bundles", path])
            out = capsys.readouterr().out
            assert "4 bundles" in out
            assert "2 source locations" in out
        finally:
            os.unlink(path)


class TestPreprocessArgvAnalyzeBundles:
    def test_analyze_bundles_not_prepended(self):
        """analyze-bundles is a known subcommand, not prepended with 'analyze'."""
        result = preprocess_argv(["analyze-bundles", "foo.txt"])
        assert result[0] == "analyze-bundles"

    def test_parser_recognizes_subcommand(self):
        """build_arg_parser recognizes analyze-bundles."""
        ap = build_arg_parser()
        args = ap.parse_args(["analyze-bundles", "foo.txt"])
        assert args.subcommand == "analyze-bundles"

    def test_parser_json_flag(self):
        ap = build_arg_parser()
        args = ap.parse_args(["analyze-bundles", "foo.txt", "--json", "out.json"])
        assert args.json == "out.json"

    def test_parser_line_flag(self):
        ap = build_arg_parser()
        args = ap.parse_args(["analyze-bundles", "foo.txt", "--line", "684"])
        assert args.line == 684

    def test_parser_source_root_flag(self):
        ap = build_arg_parser()
        args = ap.parse_args(["analyze-bundles", "foo.txt", "--source-root", "/src"])
        assert args.source_root == "/src"
