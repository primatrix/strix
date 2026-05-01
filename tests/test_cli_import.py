"""Tests for CLI subcommand refactor (Issue #209).

Acceptance criteria:
1. `python -m strix.cli import kernels.chunk_kda_fwd --shape 1,2048,4,128,128 --chunk-size 64` parses args
2. `python -m strix.cli path/to/llo.txt` old mode continues to work
3. import subcommand calls subprocess to run scripts/run_benchmark.sh
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from strix.cli import build_arg_parser, main, preprocess_argv


class TestImportSubcommandParsing:
    """import 子命令参数解析。"""

    def test_import_parses_kernel_module(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.chunk_kda_fwd", "--shape", "1,2048,4,128,128"]
        )
        assert args.kernel == "kernels.chunk_kda_fwd"

    def test_import_parses_shape(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.chunk_kda_fwd", "--shape", "1,2048,4,128,128"]
        )
        assert args.shape == "1,2048,4,128,128"

    def test_import_parses_chunk_size(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--chunk-size",
                "64",
            ]
        )
        assert args.chunk_size == 64

    def test_import_parses_tpu_type_default(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.chunk_kda_fwd", "--shape", "1,2048,4,128,128"]
        )
        assert args.tpu_type == "v7x"

    def test_import_parses_tpu_topology_default(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.chunk_kda_fwd", "--shape", "1,2048,4,128,128"]
        )
        assert args.tpu_topology == "2x2x1"

    def test_import_parses_custom_tpu_type(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--tpu-type",
                "v6e",
            ]
        )
        assert args.tpu_type == "v6e"

    def test_import_parses_custom_tpu_topology(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--tpu-topology",
                "4x4x1",
            ]
        )
        assert args.tpu_topology == "4x4x1"

    def test_import_requires_shape(self):
        ap = build_arg_parser()
        with pytest.raises(SystemExit):
            ap.parse_args(["import", "kernels.chunk_kda_fwd"])

    def test_import_subcommand_name(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.chunk_kda_fwd", "--shape", "1,2048,4,128,128"]
        )
        assert args.subcommand == "import"


class TestBackwardCompatibility:
    """旧模式（直接传 LLO 路径）继续工作。"""

    def test_old_mode_parses_path(self):
        ap = build_arg_parser()
        args = ap.parse_args(preprocess_argv(["path/to/llo.txt"]))
        assert args.path == "path/to/llo.txt"

    def test_old_mode_trace_output_default(self):
        ap = build_arg_parser()
        args = ap.parse_args(preprocess_argv(["path/to/llo.txt"]))
        assert args.trace_output == "trace.json"

    def test_old_mode_with_arg_override(self):
        ap = build_arg_parser()
        args = ap.parse_args(preprocess_argv(["path/to/llo.txt", "--arg", "%arg0=128"]))
        assert args.arg_override == ["%arg0=128"]

    def test_old_mode_with_dataflow_output(self):
        ap = build_arg_parser()
        args = ap.parse_args(
            preprocess_argv(["path/to/llo.txt", "--dataflow-output", "df.dot"])
        )
        assert args.dataflow_output == "df.dot"

    def test_old_mode_subcommand_is_analyze(self):
        ap = build_arg_parser()
        args = ap.parse_args(preprocess_argv(["path/to/llo.txt"]))
        assert args.subcommand == "analyze"

    def test_python_m_strix_help_still_works(self):
        """Existing test_project_infra expects --help to work."""
        result = subprocess.run(
            [sys.executable, "-m", "strix", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()


class TestImportSubcommandExecution:
    """import 子命令调用 subprocess 执行 scripts/run_benchmark.sh。"""

    @patch("subprocess.run")
    def test_import_calls_run_benchmark_script(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--chunk-size",
                "64",
            ]
        )
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "scripts/run_benchmark.sh" in call_args
        assert "kernels.chunk_kda_fwd" in call_args

    @patch("subprocess.run")
    def test_import_passes_shape_to_script(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
            ]
        )
        call_args = mock_run.call_args[0][0]
        assert "1,2048,4,128,128" in call_args

    @patch("subprocess.run")
    def test_import_passes_chunk_size_to_script(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--chunk-size",
                "64",
            ]
        )
        call_args = mock_run.call_args[0][0]
        assert "--chunk-size" in call_args
        assert "64" in call_args

    @patch("subprocess.run")
    def test_import_passes_tpu_type_to_script(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(
            [
                "import",
                "kernels.chunk_kda_fwd",
                "--shape",
                "1,2048,4,128,128",
                "--tpu-type",
                "v6e",
            ]
        )
        call_args = mock_run.call_args[0][0]
        assert "--tpu-type" in call_args
        assert "v6e" in call_args

    @patch("subprocess.run")
    def test_import_raises_on_script_failure(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="benchmark failed"
        )
        with pytest.raises(SystemExit):
            main(
                [
                    "import",
                    "kernels.chunk_kda_fwd",
                    "--shape",
                    "1,2048,4,128,128",
                ]
            )
