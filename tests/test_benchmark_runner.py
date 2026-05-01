"""Tests for benchmark_runner.py (Issue #211).

Acceptance criteria:
1. benchmark_runner.py accepts --kernel, --shape, --job-name params (+ env var fallback)
2. IR dump writes to /tmp/ir_dumps/{hlo,llo,mosaic}/
3. Benchmark results write to /tmp/benchmark_result.json (with timing stats)
4. tar.gz package and upload to gs://poc_profile/<job_name>/
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import benchmark_runner from scripts/ (not a Python package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER_PATH = _REPO_ROOT / "scripts" / "benchmark_runner.py"


def _import_runner():
    """Import benchmark_runner.py as a module from its file path."""
    spec = importlib.util.spec_from_file_location("benchmark_runner", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================
# Part 1: Argument parsing
# ===================================================================


class TestCliArgParsing:
    """--kernel, --shape, --job-name CLI arguments are parsed correctly."""

    def test_parses_kernel_arg(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "kernels.chunk_kda_fwd", "--shape", "1,2048"])
        assert args.kernel == "kernels.chunk_kda_fwd"

    def test_parses_shape_arg(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "kernels.test", "--shape", "1,2048,4,128,128"])
        assert args.shape == "1,2048,4,128,128"

    def test_parses_job_name_arg(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "kernels.test", "--shape", "1,2048", "--job-name", "my-job"]
        )
        assert args.job_name == "my-job"

    def test_job_name_defaults_to_benchmark(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "kernels.test", "--shape", "1,2048"])
        assert args.job_name == "benchmark"

    def test_parses_chunk_size(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "k", "--shape", "1", "--chunk-size", "64"]
        )
        assert args.chunk_size == 64

    def test_parses_gcs_bucket(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "k", "--shape", "1", "--gcs-bucket", "gs://my-bucket/"]
        )
        assert args.gcs_bucket == "gs://my-bucket/"

    def test_gcs_bucket_default(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.gcs_bucket == "gs://poc_profile/"

    def test_parses_num_runs(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "k", "--shape", "1", "--num-runs", "20"]
        )
        assert args.num_runs == 20

    def test_parses_num_warmup(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "k", "--shape", "1", "--num-warmup", "5"]
        )
        assert args.num_warmup == 5

    def test_num_runs_default(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.num_runs == 10

    def test_num_warmup_default(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.num_warmup == 3


class TestEnvVarFallback:
    """Environment variables are used when CLI args are missing."""

    def test_kernel_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"KERNEL_MODULE": "kernels.from_env"}):
            args = runner.parse_args(["--shape", "1,2048"])
        assert args.kernel == "kernels.from_env"

    def test_shape_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"SHAPE": "4,128"}):
            args = runner.parse_args(["--kernel", "k"])
        assert args.shape == "4,128"

    def test_job_name_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"JOB_NAME": "env-job"}):
            args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.job_name == "env-job"

    def test_chunk_size_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"CHUNK_SIZE": "128"}):
            args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.chunk_size == 128

    def test_gcs_bucket_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"GCS_BUCKET": "gs://other/"}):
            args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.gcs_bucket == "gs://other/"

    def test_cli_overrides_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"KERNEL_MODULE": "from_env"}):
            args = runner.parse_args(["--kernel", "from_cli", "--shape", "1"])
        assert args.kernel == "from_cli"

    def test_kernel_required_when_no_env(self):
        runner = _import_runner()
        env = {k: v for k, v in os.environ.items() if k != "KERNEL_MODULE"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit):
                runner.parse_args(["--shape", "1"])

    def test_shape_required_when_no_env(self):
        runner = _import_runner()
        env = {k: v for k, v in os.environ.items() if k != "SHAPE"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit):
                runner.parse_args(["--kernel", "k"])

    def test_no_args_reads_all_from_env(self):
        """In K8s Job context: no CLI args, everything from env."""
        runner = _import_runner()
        env_vars = {
            "KERNEL_MODULE": "kernels.moe",
            "SHAPE": "1024,128",
            "JOB_NAME": "strix-benchmark-moe-20260501",
            "CHUNK_SIZE": "64",
            "GCS_BUCKET": "gs://poc_profile/",
        }
        with patch.dict(os.environ, env_vars):
            args = runner.parse_args([])
        assert args.kernel == "kernels.moe"
        assert args.shape == "1024,128"
        assert args.job_name == "strix-benchmark-moe-20260501"
        assert args.chunk_size == 64
        assert args.gcs_bucket == "gs://poc_profile/"


class TestIrDumpDirSetup:
    """setup_ir_dump_dirs creates /tmp/ir_dumps/{hlo,llo,mosaic}/."""

    def test_creates_all_subdirs(self, tmp_path):
        runner = _import_runner()
        runner.setup_ir_dump_dirs(tmp_path)
        assert (tmp_path / "hlo").is_dir()
        assert (tmp_path / "llo").is_dir()
        assert (tmp_path / "mosaic").is_dir()

    def test_idempotent(self, tmp_path):
        runner = _import_runner()
        runner.setup_ir_dump_dirs(tmp_path)
        runner.setup_ir_dump_dirs(tmp_path)  # Should not raise
        assert (tmp_path / "hlo").is_dir()


class TestXlaFlagsSetup:
    """setup_xla_flags sets env vars only when missing."""

    def test_sets_xla_flags_when_missing(self, tmp_path):
        runner = _import_runner()
        env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
        with patch.dict(os.environ, env, clear=True):
            runner.setup_xla_flags(tmp_path)
            assert "XLA_FLAGS" in os.environ
            assert "--xla_dump_hlo_as_text" in os.environ["XLA_FLAGS"]
            assert str(tmp_path / "hlo") in os.environ["XLA_FLAGS"]

    def test_sets_libtpu_init_args_when_missing(self, tmp_path):
        runner = _import_runner()
        env = {k: v for k, v in os.environ.items() if k != "LIBTPU_INIT_ARGS"}
        with patch.dict(os.environ, env, clear=True):
            runner.setup_xla_flags(tmp_path)
            assert "LIBTPU_INIT_ARGS" in os.environ
            libtpu = os.environ["LIBTPU_INIT_ARGS"]
            assert "--xla_jf_dump_llo_text=true" in libtpu
            assert str(tmp_path / "llo") in libtpu
            assert str(tmp_path / "mosaic") in libtpu

    def test_does_not_overwrite_existing_xla_flags(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"XLA_FLAGS": "custom-flags"}):
            runner.setup_xla_flags("/tmp/ir_dumps")
            assert os.environ["XLA_FLAGS"] == "custom-flags"

    def test_does_not_overwrite_existing_libtpu_init_args(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "custom-args"}):
            runner.setup_xla_flags("/tmp/ir_dumps")
            assert os.environ["LIBTPU_INIT_ARGS"] == "custom-args"
