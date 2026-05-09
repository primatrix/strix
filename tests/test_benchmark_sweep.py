"""Tests for (bf, bd) sweep logic in benchmark_runner.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER_PATH = _REPO_ROOT / "scripts" / "benchmark_runner.py"


def _import_runner():
    spec = importlib.util.spec_from_file_location("benchmark_runner", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 64 MiB default total bytes; bf16 → dtype_bytes=2.
_TOTAL = 64 * 1024 * 1024  # 67108864
_DB = 2


class TestParseSweepValid:
    def test_single_pair(self):
        runner = _import_runner()
        out = runner.parse_sweep("2048:1024", _TOTAL, _DB)
        assert out == [
            {"bf": 2048, "bd": 1024, "num_loads": 16, "tile_bytes": 2048 * 1024 * 2},
        ]

    def test_multiple_pairs(self):
        runner = _import_runner()
        out = runner.parse_sweep("2048:1024,1024:512", _TOTAL, _DB)
        assert [(c["bf"], c["bd"], c["num_loads"]) for c in out] == [
            (2048, 1024, 16),
            (1024, 512, 64),
        ]

    def test_num_loads_derived_per_config(self):
        runner = _import_runner()
        out = runner.parse_sweep("512:512", _TOTAL, _DB)
        assert out[0]["num_loads"] == _TOTAL // (512 * 512 * _DB)


class TestParseSweepInvalidFormat:
    def test_empty_spec_rejected(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_sweep("", _TOTAL, _DB)

    def test_bad_separator_rejected(self):
        runner = _import_runner()
        with pytest.raises(SystemExit) as exc:
            runner.parse_sweep("2048-1024", _TOTAL, _DB)
        assert "invalid pair" in str(exc.value)

    def test_missing_second_number_rejected(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_sweep("2048:", _TOTAL, _DB)

    def test_non_integer_rejected(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_sweep("abc:1024", _TOTAL, _DB)


class TestParseSweepConstraints:
    def test_bf_must_be_multiple_of_128(self):
        runner = _import_runner()
        with pytest.raises(SystemExit) as exc:
            runner.parse_sweep("2047:1024", _TOTAL, _DB)
        msg = str(exc.value)
        assert "bf" in msg and "128" in msg

    def test_bd_must_be_multiple_of_8(self):
        runner = _import_runner()
        with pytest.raises(SystemExit) as exc:
            runner.parse_sweep("2048:15", _TOTAL, _DB)
        msg = str(exc.value)
        assert "bd" in msg and "8" in msg

    def test_bf_must_be_positive(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_sweep("0:1024", _TOTAL, _DB)

    def test_total_bytes_must_be_divisible_by_tile_bytes(self):
        runner = _import_runner()
        # 2048 * 1024 * 2 = 4194304; 4194305 is not divisible
        with pytest.raises(SystemExit) as exc:
            runner.parse_sweep("2048:1024", 4194305, _DB)
        assert "not divisible" in str(exc.value)

    def test_num_loads_zero_rejected(self):
        runner = _import_runner()
        # total_bytes smaller than a single tile
        with pytest.raises(SystemExit) as exc:
            runner.parse_sweep("2048:1024", 1024, _DB)
        msg = str(exc.value)
        assert "num_loads" in msg


class TestParseSweepAggregatesErrors:
    """All offending pairs should be reported at once."""

    def test_multiple_violations_all_reported(self):
        runner = _import_runner()
        with pytest.raises(SystemExit) as exc:
            # First: bf not multiple of 128; second: bd not multiple of 8; third: ok
            runner.parse_sweep("2047:1024,2048:15,2048:1024", _TOTAL, _DB)
        msg = str(exc.value)
        assert "sweep[0]" in msg
        assert "sweep[1]" in msg


class TestSweepCliFlags:
    def test_parses_sweep_flag(self):
        runner = _import_runner()
        args = runner.parse_args([
            "--kernel", "k", "--shape", "1", "--sweep", "2048:1024",
        ])
        assert args.sweep == "2048:1024"

    def test_parses_total_bytes(self):
        runner = _import_runner()
        args = runner.parse_args([
            "--kernel", "k", "--shape", "1", "--total-bytes", "4194304",
        ])
        assert args.total_bytes == 4194304

    def test_total_bytes_defaults_to_64mib(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.total_bytes == 64 * 1024 * 1024

    def test_sweep_env_fallback(self, monkeypatch):
        monkeypatch.setenv("SWEEP", "2048:1024,1024:512")
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.sweep == "2048:1024,1024:512"

    def test_sweep_mutex_with_bf(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_args([
                "--kernel", "k", "--shape", "1",
                "--sweep", "2048:1024", "--bf", "2048",
            ])

    def test_sweep_mutex_with_bd(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_args([
                "--kernel", "k", "--shape", "1",
                "--sweep", "2048:1024", "--bd", "1024",
            ])

    def test_total_bytes_empty_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("TOTAL_BYTES", "")
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.total_bytes == 64 * 1024 * 1024

    def test_total_bytes_zero_rejected(self):
        runner = _import_runner()
        with pytest.raises(SystemExit):
            runner.parse_args([
                "--kernel", "k", "--shape", "1", "--total-bytes", "0",
            ])


class TestCheckKernelCompat:
    def test_accepts_kernel_with_sweep_kwargs(self):
        runner = _import_runner()

        def kernel_fn(hidden_size=1, intermediate_size=1, bf=1, bd=1, num_loads=1, **_):
            return lambda: None

        # Should not raise.
        runner.check_kernel_compat(kernel_fn, module_name="fake")

    def test_accepts_kernel_with_kwargs_only(self):
        runner = _import_runner()

        def kernel_fn(**kwargs):
            return lambda: None

        runner.check_kernel_compat(kernel_fn, module_name="fake")

    def test_rejects_kernel_missing_required_kwargs(self):
        runner = _import_runner()

        def kernel_fn(hidden_size=1, intermediate_size=1):
            return lambda: None

        with pytest.raises(SystemExit) as exc:
            runner.check_kernel_compat(kernel_fn, module_name="fake")
        msg = str(exc.value)
        assert "bd" in msg and "bf" in msg and "num_loads" in msg


class TestBuildSweepRecord:
    def test_ok_record_has_timings_and_stats(self):
        runner = _import_runner()
        cfg = {"bf": 2048, "bd": 1024, "num_loads": 16, "tile_bytes": 4194304}
        rec = runner.build_sweep_record(
            kernel="k", shape="8192,2048", job_name="j",
            config_index=0, cfg=cfg, total_bytes=_TOTAL, dtype="bfloat16",
            timings=[0.001, 0.002, 0.003],
            status="ok",
            error=None,
        )
        assert rec["status"] == "ok"
        assert rec["config"] == {"bf": 2048, "bd": 1024, "num_loads": 16}
        assert rec["derived"]["tile_bytes"] == 4194304
        assert rec["num_runs"] == 3
        assert rec["statistics"]["median_ms"] == pytest.approx(2.0)
        # GiB/s = total_bytes / median_s / 1024**3
        expected = _TOTAL / 0.002 / (1024 ** 3)
        assert rec["throughput"]["gib_per_s_median"] == pytest.approx(expected)

    def test_failed_record_has_null_timings(self):
        runner = _import_runner()
        cfg = {"bf": 2048, "bd": 1024, "num_loads": 16, "tile_bytes": 4194304}
        rec = runner.build_sweep_record(
            kernel="k", shape="8192,2048", job_name="j",
            config_index=1, cfg=cfg, total_bytes=_TOTAL, dtype="bfloat16",
            timings=None,
            status="failed",
            error="RuntimeError: OOM",
        )
        assert rec["status"] == "failed"
        assert rec["timings_ms"] is None
        assert rec["statistics"] is None
        assert rec["throughput"] is None
        assert rec["error"] == "RuntimeError: OOM"


class TestRunSweep:
    def test_runs_every_config_and_records_timings(self):
        runner = _import_runner()
        calls = []

        def fake_kernel_fn(**kwargs):
            calls.append(kwargs)

            def run():
                return 0.0  # scalar without block_until_ready

            return run

        fake_config = {"default_shape": {"hidden_size": 8192, "intermediate_size": 2048}}
        sweep = [
            {"bf": 2048, "bd": 1024, "num_loads": 16, "tile_bytes": 4194304},
            {"bf": 1024, "bd": 512, "num_loads": 128, "tile_bytes": 1048576},
        ]

        records = list(runner.run_sweep(
            fake_kernel_fn, fake_config, sweep,
            num_warmup=1, num_runs=2, total_bytes=_TOTAL, dtype="bfloat16",
            kernel="k", shape="8192,2048", job_name="j",
        ))
        assert len(records) == 2
        assert all(r["status"] == "ok" for r in records)
        # kernel_fn receives bf/bd/num_loads via kwargs
        assert calls[0]["bf"] == 2048 and calls[0]["bd"] == 1024 and calls[0]["num_loads"] == 16
        assert calls[1]["bf"] == 1024 and calls[1]["bd"] == 512 and calls[1]["num_loads"] == 128

    def test_failing_config_does_not_abort_sweep(self):
        runner = _import_runner()
        attempts = []

        def fake_kernel_fn(**kwargs):
            attempts.append(kwargs)
            if kwargs["bf"] == 1024:
                raise RuntimeError("boom")
            return lambda: 0.0

        fake_config = {"default_shape": {}}
        sweep = [
            {"bf": 2048, "bd": 1024, "num_loads": 16, "tile_bytes": 4194304},
            {"bf": 1024, "bd": 512, "num_loads": 128, "tile_bytes": 1048576},
            {"bf": 2048, "bd": 512, "num_loads": 32, "tile_bytes": 2097152},
        ]

        records = list(runner.run_sweep(
            fake_kernel_fn, fake_config, sweep,
            num_warmup=1, num_runs=1, total_bytes=_TOTAL, dtype="bfloat16",
            kernel="k", shape="8192,2048", job_name="j",
        ))
        assert [r["status"] for r in records] == ["ok", "failed", "ok"]
        assert "boom" in records[1]["error"]
        assert len(attempts) == 3
