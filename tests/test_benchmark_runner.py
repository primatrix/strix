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
from unittest.mock import patch, MagicMock

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


# ===================================================================
# Part 2: Kernel import + benchmark execution
# ===================================================================


def _make_fake_kernel_module():
    """Create a fake kernel module with kernel_fn and config."""
    import types

    mod = types.ModuleType("fake_kernel")
    mod.config = {
        "default_shape": {"num_tokens": 32, "hidden_size": 64},
        "dtype": "bfloat16",
        "description": "test kernel",
    }
    call_count = 0

    def kernel_fn(**kwargs):
        def run():
            nonlocal call_count
            call_count += 1
            return call_count
        return run

    mod.kernel_fn = kernel_fn
    mod._get_call_count = lambda: call_count
    return mod


class TestKernelImport:
    """import_kernel loads a module and extracts kernel_fn + config."""

    def test_imports_and_returns_kernel_fn_and_config(self):
        runner = _import_runner()
        fake_mod = _make_fake_kernel_module()
        with patch("importlib.import_module", return_value=fake_mod):
            kernel_fn, config = runner.import_kernel("kernels.fake")
        assert callable(kernel_fn)
        assert config["dtype"] == "bfloat16"

    def test_raises_on_missing_kernel_fn(self):
        runner = _import_runner()
        import types
        bad_mod = types.ModuleType("bad")
        bad_mod.config = {}
        with patch("importlib.import_module", return_value=bad_mod):
            with pytest.raises(AttributeError):
                runner.import_kernel("kernels.bad")

    def test_raises_on_missing_config(self):
        runner = _import_runner()
        import types
        bad_mod = types.ModuleType("bad")
        bad_mod.kernel_fn = lambda: None
        with patch("importlib.import_module", return_value=bad_mod):
            with pytest.raises(AttributeError):
                runner.import_kernel("kernels.bad")


class TestBenchmarkExecution:
    """run_benchmark executes kernel_fn and collects timing."""

    def test_returns_correct_number_of_timings(self):
        runner = _import_runner()
        fake_mod = _make_fake_kernel_module()
        timings = runner.run_benchmark(
            fake_mod.kernel_fn, fake_mod.config, num_warmup=2, num_runs=5,
        )
        assert len(timings) == 5

    def test_all_timings_are_positive(self):
        runner = _import_runner()
        fake_mod = _make_fake_kernel_module()
        timings = runner.run_benchmark(
            fake_mod.kernel_fn, fake_mod.config, num_warmup=1, num_runs=3,
        )
        assert all(t > 0 for t in timings)

    def test_warmup_runs_not_counted_in_timings(self):
        runner = _import_runner()
        fake_mod = _make_fake_kernel_module()
        timings = runner.run_benchmark(
            fake_mod.kernel_fn, fake_mod.config, num_warmup=3, num_runs=2,
        )
        # 3 warmup + 2 timed = 5 total calls, but only 2 timings returned
        assert len(timings) == 2

    def test_calls_block_until_ready_on_jax_arrays(self):
        runner = _import_runner()
        import types

        blocked = []

        class FakeArray:
            def block_until_ready(self):
                blocked.append(True)
                return self

        mod = types.ModuleType("jax_kernel")
        mod.config = {"default_shape": {}}

        def kernel_fn(**kwargs):
            def run():
                return FakeArray()
            return run

        mod.kernel_fn = kernel_fn
        runner.run_benchmark(mod.kernel_fn, mod.config, num_warmup=1, num_runs=2)
        # 1 warmup + 2 timed = 3 block_until_ready calls
        assert len(blocked) == 3

    def test_passes_default_shape_to_kernel_fn(self):
        runner = _import_runner()
        import types

        received_kwargs = {}

        mod = types.ModuleType("shape_kernel")
        mod.config = {"default_shape": {"num_tokens": 64, "hidden_size": 128}}

        def kernel_fn(**kwargs):
            received_kwargs.update(kwargs)
            return lambda: None

        mod.kernel_fn = kernel_fn
        runner.run_benchmark(mod.kernel_fn, mod.config, num_warmup=0, num_runs=1)
        assert received_kwargs == {"num_tokens": 64, "hidden_size": 128}

    def test_passes_chunk_size_when_provided(self):
        runner = _import_runner()
        import types

        received_kwargs = {}

        mod = types.ModuleType("chunk_kernel")
        mod.config = {"default_shape": {"num_tokens": 32}}

        def kernel_fn(**kwargs):
            received_kwargs.update(kwargs)
            return lambda: None

        mod.kernel_fn = kernel_fn
        runner.run_benchmark(
            mod.kernel_fn, mod.config, num_warmup=0, num_runs=1, chunk_size=64,
        )
        assert received_kwargs["chunk_size"] == 64


# ===================================================================
# Part 3: Result JSON + tar packaging
# ===================================================================


class TestResultWriting:
    """write_benchmark_result produces correct JSON structure."""

    def test_writes_valid_json(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        runner.write_benchmark_result(
            timings=[0.1, 0.2, 0.15],
            kernel="kernels.test",
            shape="1,2048",
            job_name="test-job",
            config={"description": "test"},
            output_path=out,
        )
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_contains_timing_stats(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        runner.write_benchmark_result(
            timings=[0.1, 0.2, 0.3],
            kernel="k",
            shape="1",
            job_name="j",
            config={},
            output_path=out,
        )
        data = json.loads(out.read_text())
        stats = data["statistics"]
        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "stdev_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats

    def test_timing_stats_values_correct(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        # 100ms, 200ms, 300ms → mean=200ms, median=200ms, min=100ms, max=300ms
        runner.write_benchmark_result(
            timings=[0.1, 0.2, 0.3],
            kernel="k",
            shape="1",
            job_name="j",
            config={},
            output_path=out,
        )
        data = json.loads(out.read_text())
        stats = data["statistics"]
        assert abs(stats["mean_ms"] - 200.0) < 0.01
        assert abs(stats["median_ms"] - 200.0) < 0.01
        assert abs(stats["min_ms"] - 100.0) < 0.01
        assert abs(stats["max_ms"] - 300.0) < 0.01

    def test_includes_kernel_metadata(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        runner.write_benchmark_result(
            timings=[0.1],
            kernel="kernels.moe",
            shape="1024,128",
            job_name="strix-benchmark-moe",
            config={"description": "MoE kernel"},
            output_path=out,
        )
        data = json.loads(out.read_text())
        assert data["kernel"] == "kernels.moe"
        assert data["shape"] == "1024,128"
        assert data["job_name"] == "strix-benchmark-moe"
        assert data["config"]["description"] == "MoE kernel"

    def test_includes_raw_timings_ms(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        runner.write_benchmark_result(
            timings=[0.1, 0.2],
            kernel="k",
            shape="1",
            job_name="j",
            config={},
            output_path=out,
        )
        data = json.loads(out.read_text())
        assert data["timings_ms"] == [100.0, 200.0]

    def test_stdev_zero_for_single_run(self, tmp_path):
        import json

        runner = _import_runner()
        out = tmp_path / "result.json"
        runner.write_benchmark_result(
            timings=[0.1],
            kernel="k",
            shape="1",
            job_name="j",
            config={},
            output_path=out,
        )
        data = json.loads(out.read_text())
        assert data["statistics"]["stdev_ms"] == 0.0


class TestTarPackaging:
    """package_results creates tar.gz with correct contents."""

    def test_creates_tarball(self, tmp_path):
        runner = _import_runner()
        ir_root = tmp_path / "ir_dumps"
        (ir_root / "hlo").mkdir(parents=True)
        (ir_root / "hlo" / "dump.txt").write_text("hlo data")
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text("{}")

        tarball = runner.package_results("test-job", ir_root, result_file, tmp_path)
        assert tarball.exists()
        assert tarball.name == "test-job.tar.gz"

    def test_tarball_contains_ir_dumps(self, tmp_path):
        import tarfile

        runner = _import_runner()
        ir_root = tmp_path / "ir_dumps"
        (ir_root / "llo").mkdir(parents=True)
        (ir_root / "llo" / "module.llo").write_text("llo content")
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text("{}")

        tarball = runner.package_results("j", ir_root, result_file, tmp_path)
        with tarfile.open(tarball) as tf:
            names = tf.getnames()
        assert any("ir_dumps" in n for n in names)
        assert any("module.llo" in n for n in names)

    def test_tarball_contains_benchmark_result(self, tmp_path):
        import tarfile

        runner = _import_runner()
        ir_root = tmp_path / "ir_dumps"
        ir_root.mkdir()
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text('{"key": "value"}')

        tarball = runner.package_results("j", ir_root, result_file, tmp_path)
        with tarfile.open(tarball) as tf:
            names = tf.getnames()
        assert any("benchmark_result.json" in n for n in names)


# ===================================================================
# Part 4: GCS upload + main integration
# ===================================================================


class TestGcsUpload:
    """upload_to_gcs uploads tarball to correct GCS path."""

    def test_uploads_to_correct_blob_path(self, tmp_path):
        runner = _import_runner()
        tarball = tmp_path / "my-job.tar.gz"
        tarball.write_bytes(b"fake tarball")

        mock_blob = patch.object(
            __builtins__, "__import__", side_effect=ImportError
        )
        # Mock the google.cloud.storage module
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch.dict(sys.modules, {"google.cloud.storage": MagicMock(Client=mock_client)}):
            runner.upload_to_gcs(tarball, "gs://poc_profile/", "my-job")

        mock_bucket.blob.assert_called_once_with("my-job/my-job.tar.gz")
        mock_blob.upload_from_filename.assert_called_once_with(str(tarball))

    def test_parses_bucket_name_from_gs_url(self, tmp_path):
        runner = _import_runner()
        tarball = tmp_path / "j.tar.gz"
        tarball.write_bytes(b"data")

        from unittest.mock import MagicMock

        mock_client = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.storage": MagicMock(Client=mock_client)}):
            runner.upload_to_gcs(tarball, "gs://my-custom-bucket/", "j")

        mock_client.return_value.bucket.assert_called_once_with("my-custom-bucket")

    def test_handles_bucket_without_trailing_slash(self, tmp_path):
        runner = _import_runner()
        tarball = tmp_path / "j.tar.gz"
        tarball.write_bytes(b"data")

        from unittest.mock import MagicMock

        mock_client = MagicMock()
        with patch.dict(sys.modules, {"google.cloud.storage": MagicMock(Client=mock_client)}):
            runner.upload_to_gcs(tarball, "gs://bucket-name", "j")

        mock_client.return_value.bucket.assert_called_once_with("bucket-name")


class TestMainIntegration:
    """main() wires all steps together."""

    def test_main_runs_full_pipeline(self, tmp_path):
        runner = _import_runner()
        from unittest.mock import MagicMock

        fake_mod = _make_fake_kernel_module()

        ir_dump_root = tmp_path / "ir_dumps"
        result_path = tmp_path / "benchmark_result.json"

        mock_storage = MagicMock()
        with (
            patch("importlib.import_module", return_value=fake_mod),
            patch.dict(sys.modules, {"google.cloud.storage": mock_storage}),
            patch.dict(os.environ, {}, clear=False),
        ):
            runner.main(
                [
                    "--kernel", "kernels.test",
                    "--shape", "1,2048",
                    "--job-name", "test-job",
                    "--num-warmup", "1",
                    "--num-runs", "2",
                    "--gcs-bucket", "gs://test-bucket/",
                ],
                ir_dump_root=ir_dump_root,
                benchmark_result_path=result_path,
                output_dir=tmp_path,
            )

        # Verify IR dump dirs created
        assert (ir_dump_root / "hlo").is_dir()
        assert (ir_dump_root / "llo").is_dir()
        assert (ir_dump_root / "mosaic").is_dir()

        # Verify result JSON written
        import json
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["kernel"] == "kernels.test"
        assert len(data["timings_ms"]) == 2

        # Verify tarball created
        assert (tmp_path / "test-job.tar.gz").exists()

        # Verify GCS upload called
        mock_storage.Client.return_value.bucket.assert_called_once()

    def test_main_reads_env_vars(self, tmp_path):
        runner = _import_runner()
        from unittest.mock import MagicMock

        fake_mod = _make_fake_kernel_module()
        mock_storage = MagicMock()

        ir_dump_root = tmp_path / "ir_dumps"
        result_path = tmp_path / "benchmark_result.json"

        env = {
            "KERNEL_MODULE": "kernels.env_test",
            "SHAPE": "32,64",
            "JOB_NAME": "env-job",
            "GCS_BUCKET": "gs://env-bucket/",
        }
        with (
            patch("importlib.import_module", return_value=fake_mod),
            patch.dict(sys.modules, {"google.cloud.storage": mock_storage}),
            patch.dict(os.environ, env),
        ):
            runner.main(
                [],
                ir_dump_root=ir_dump_root,
                benchmark_result_path=result_path,
                output_dir=tmp_path,
            )

        import json
        data = json.loads(result_path.read_text())
        assert data["kernel"] == "kernels.env_test"
        assert data["job_name"] == "env-job"
