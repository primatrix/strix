"""Tests for project infrastructure (Issue #208)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

# Repo root = package root (strix/ is both)
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


class TestPyprojectToml:
    """pyproject.toml must exist with correct metadata and [tpu] extras."""

    def test_pyproject_exists(self):
        assert PYPROJECT_PATH.exists(), "pyproject.toml not found at repo root"

    def test_pyproject_parseable(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        assert "project" in config

    def test_project_name_is_strix(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        assert config["project"]["name"] == "strix"

    def test_tpu_extras_exist(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        optional_deps = config["project"].get("optional-dependencies", {})
        assert "tpu" in optional_deps, "Missing [tpu] extras"

    def test_tpu_extras_include_jax(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        tpu_deps = config["project"]["optional-dependencies"]["tpu"]
        assert any("jax" in d for d in tpu_deps), "jax not in [tpu] extras"

    def test_tpu_extras_include_gcs(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        tpu_deps = config["project"]["optional-dependencies"]["tpu"]
        assert any("google-cloud-storage" in d for d in tpu_deps), (
            "google-cloud-storage not in [tpu] extras"
        )

    def test_requires_python_310_plus(self):
        with open(PYPROJECT_PATH, "rb") as f:
            config = tomllib.load(f)
        assert "requires-python" in config["project"]
        assert "3.10" in config["project"]["requires-python"]


class TestMainEntry:
    """``python -m strix`` must invoke the CLI entry point."""

    def test_python_m_strix_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "strix", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()
