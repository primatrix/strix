"""Tests for standalone expert FFN kernel (profiling tool).

Acceptance criteria:
1. kernels/expert_ffn.py exists and exports kernel_fn + config
2. config contains Ling 2.6 per-expert defaults (H=8192, I=2048)
3. Module is importable via importlib
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "expert_ffn.py"


class TestExpertFfnFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/expert_ffn.py not found"


class TestExpertFfnModuleStructure:
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

    def test_defines_expert_ffn_kernel(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "_expert_ffn_kernel" in func_names


class TestExpertFfnConfig:
    @pytest.fixture(autouse=True)
    def _extract_config(self):
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
        assert self.config["default_shape"]["num_tokens"] == 256

    def test_default_shape_hidden_size(self):
        assert self.config["default_shape"]["hidden_size"] == 8192

    def test_default_shape_intermediate_size(self):
        assert self.config["default_shape"]["intermediate_size"] == 2048

    def test_config_dtype(self):
        assert self.config["dtype"] == "bfloat16"

    def test_config_tpu_topology(self):
        assert self.config["tpu_topology"] == "1x1"

    def test_config_has_description(self):
        assert isinstance(self.config["description"], str)
        assert len(self.config["description"]) > 0

    def test_no_num_experts_in_default_shape(self):
        assert "num_experts" not in self.config["default_shape"]

    def test_no_top_k_in_default_shape(self):
        assert "top_k" not in self.config["default_shape"]


class TestExpertFfnImportlib:
    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX required for import test")

    def test_import_expert_ffn_module(self):
        from strix.kernels import expert_ffn  # noqa: F401

    def test_module_has_kernel_fn(self):
        from strix.kernels import expert_ffn
        assert hasattr(expert_ffn, "kernel_fn")

    def test_module_has_config(self):
        from strix.kernels import expert_ffn
        assert hasattr(expert_ffn, "config")

    def test_kernel_fn_is_callable(self):
        from strix.kernels import expert_ffn
        assert callable(expert_ffn.kernel_fn)

    def test_importlib_import_module(self):
        mod = importlib.import_module("strix.kernels.expert_ffn")
        assert hasattr(mod, "kernel_fn")
        assert hasattr(mod, "config")
        assert callable(mod.kernel_fn)
