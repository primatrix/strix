"""Tests for fused MoE kernel wrapper (Issue #213).

Acceptance criteria:
1. kernels/fused_moe.py exports kernel_fn and config dict
2. config contains Qwen3-MoE-128E default params (num_tokens=1024, num_experts=128, top_k=8, etc.)
3. kernel_fn can be dynamically imported via importlib and returns a compilable JAX closure
4. Kernel code copied locally (no sglang-jax package dependency)
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "fused_moe.py"
IMPL_FILE = REPO_ROOT / "kernels" / "_fused_moe_impl.py"
CONFIGS_FILE = REPO_ROOT / "kernels" / "_fused_moe_configs.py"


# ---------------------------------------------------------------------------
# AC1: kernels/fused_moe.py exports kernel_fn and config dict
# ---------------------------------------------------------------------------


class TestFusedMoeFileExists:
    """Kernel wrapper and implementation files must exist."""

    def test_wrapper_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/fused_moe.py not found"

    def test_impl_file_exists(self):
        assert IMPL_FILE.is_file(), "kernels/_fused_moe_impl.py not found"

    def test_configs_file_exists(self):
        assert CONFIGS_FILE.is_file(), "kernels/_fused_moe_configs.py not found"


class TestFusedMoeModuleStructure:
    """AC1: wrapper defines kernel_fn function and config dict (AST check)."""

    @pytest.fixture(autouse=True)
    def _parse_module(self):
        self.tree = ast.parse(KERNEL_FILE.read_text())

    def test_defines_kernel_fn(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "kernel_fn" in func_names

    def test_defines_config(self):
        assign_targets = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assign_targets.append(target.id)
        assert "config" in assign_targets


# ---------------------------------------------------------------------------
# AC2: config contains Qwen3-MoE-128E default parameters
# ---------------------------------------------------------------------------


class TestFusedMoeConfig:
    """AC2: config dict has correct Qwen3-MoE-128E default parameters."""

    @pytest.fixture(autouse=True)
    def _extract_config(self):
        """Extract config dict via AST eval (no JAX imports needed)."""
        source = KERNEL_FILE.read_text()
        tree = ast.parse(source)
        config_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "config":
                        config_node = node
                        break
        assert config_node is not None, "config assignment not found in AST"
        module = ast.Module(body=[config_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, str(KERNEL_FILE), "exec")
        ns = {}
        exec(code, ns)
        self.config = ns["config"]

    def test_config_is_dict(self):
        assert isinstance(self.config, dict)

    def test_config_has_default_shape(self):
        assert "default_shape" in self.config

    def test_default_shape_num_tokens(self):
        assert self.config["default_shape"]["num_tokens"] == 1024

    def test_default_shape_num_experts(self):
        assert self.config["default_shape"]["num_experts"] == 128

    def test_default_shape_top_k(self):
        assert self.config["default_shape"]["top_k"] == 8

    def test_default_shape_hidden_size(self):
        assert self.config["default_shape"]["hidden_size"] == 4096

    def test_default_shape_intermediate_size(self):
        assert self.config["default_shape"]["intermediate_size"] == 2048

    def test_config_dtype(self):
        assert self.config["dtype"] == "bfloat16"

    def test_config_tpu_type(self):
        assert self.config["tpu_type"] == "v7x"

    def test_config_tpu_topology(self):
        assert self.config["tpu_topology"] == "2x2x1"

    def test_config_has_description(self):
        assert isinstance(self.config["description"], str)
        assert len(self.config["description"]) > 0


# ---------------------------------------------------------------------------
# AC3: kernel_fn importable via importlib (mocked JAX for CI)
# ---------------------------------------------------------------------------


class TestFusedMoeImportlib:
    """AC3: strix.kernels.fused_moe is importable and exports kernel_fn + config."""

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX required for import test")

    def test_import_fused_moe_module(self):
        from strix.kernels import fused_moe  # noqa: F401

    def test_module_has_kernel_fn(self):
        from strix.kernels import fused_moe

        assert hasattr(fused_moe, "kernel_fn")

    def test_module_has_config(self):
        from strix.kernels import fused_moe

        assert hasattr(fused_moe, "config")

    def test_kernel_fn_is_callable(self):
        from strix.kernels import fused_moe

        assert callable(fused_moe.kernel_fn)

    def test_importlib_import_module(self):
        """AC3: explicit importlib.import_module as benchmark harness would use."""
        mod = importlib.import_module("strix.kernels.fused_moe")
        assert hasattr(mod, "kernel_fn")
        assert hasattr(mod, "config")
        assert callable(mod.kernel_fn)


# ---------------------------------------------------------------------------
# AC4 (revised): no sglang-jax package dependency — code copied locally
# ---------------------------------------------------------------------------


class TestNoSglangJaxDependency:
    """Kernel code is copied locally, not imported from sglang-jax."""

    def test_impl_does_not_import_sglang(self):
        """_fused_moe_impl.py must not import from sgl_jax."""
        source = IMPL_FILE.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sgl_jax"), (
                    f"Found sgl_jax import: from {node.module}"
                )

    def test_configs_does_not_import_sglang(self):
        """_fused_moe_configs.py must not import from sgl_jax."""
        source = CONFIGS_FILE.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sgl_jax"), (
                    f"Found sgl_jax import: from {node.module}"
                )

    def test_wrapper_does_not_import_sglang(self):
        """kernels/fused_moe.py must not import from sgl_jax."""
        source = KERNEL_FILE.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sgl_jax"), (
                    f"Found sgl_jax import: from {node.module}"
                )


# ---------------------------------------------------------------------------
# Implementation file structure checks
# ---------------------------------------------------------------------------


class TestFusedMoeImplStructure:
    """Implementation file contains the ported kernel code."""

    @pytest.fixture(autouse=True)
    def _parse_impl(self):
        self.tree = ast.parse(IMPL_FILE.read_text())

    def test_defines_fused_ep_moe(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "fused_ep_moe" in func_names

    def test_defines_fused_moe_block_config(self):
        class_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        ]
        assert "FusedMoEBlockConfig" in class_names

    def test_defines_fused_ep_moe_kernel(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "_fused_ep_moe_kernel" in func_names
