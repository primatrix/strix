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
