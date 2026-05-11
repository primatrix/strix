"""Structural tests for kernels/double_buffer_expert.py (no TPU required).

Mirrors tests/test_expert_ffn_kernel.py — AST-based checks that the module
exposes the correct public API per
docs/superpowers/specs/2026-05-11-double-buffer-expert-design.md.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "double_buffer_expert.py"


class TestFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/double_buffer_expert.py not found"


class TestModuleStructure:
    @pytest.fixture(autouse=True)
    def _parse_module(self):
        self.tree = ast.parse(KERNEL_FILE.read_text())

    def _func_names(self):
        return [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def _assign_names(self):
        names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
        return names

    def test_defines_kernel_fn(self):
        assert "kernel_fn" in self._func_names()

    def test_defines_double_buffer_expert(self):
        assert "double_buffer_expert" in self._func_names()

    def test_defines_ref_expert_ffn(self):
        assert "_ref_expert_ffn" in self._func_names()

    def test_defines_kernel_body(self):
        assert "_double_buffer_expert_kernel" in self._func_names()

    def test_defines_config(self):
        assert "config" in self._assign_names()


class TestConfig:
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
        assert config_node is not None, "config assignment not found"
        module = ast.Module(body=[config_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, str(KERNEL_FILE), "exec")
        ns = {}
        exec(code, ns)
        self.config = ns["config"]

    def test_config_is_dict(self):
        assert isinstance(self.config, dict)

    def test_default_shape_has_ling_26_values(self):
        ds = self.config["default_shape"]
        assert ds["num_tokens"] == 256
        assert ds["hidden_size"] == 8192
        assert ds["intermediate_size"] == 2048

    def test_default_bf_is_512(self):
        assert self.config["bf"] == 512

    def test_dtype_is_bfloat16(self):
        assert self.config["dtype"] == "bfloat16"
        assert self.config["weight_dtype"] == "bfloat16"

    def test_act_fn_is_silu(self):
        assert self.config["act_fn"] == "silu"

    def test_tpu_type_is_v7x(self):
        assert self.config["tpu_type"] == "v7x"


class TestKernelFnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.double_buffer_expert")

    def test_kernel_fn_accepts_required_kwargs(self):
        sig = inspect.signature(self.module.kernel_fn)
        params = set(sig.parameters.keys())
        required = {
            "num_tokens", "hidden_size", "intermediate_size",
            "dtype", "weight_dtype", "act_fn", "bf",
        }
        missing = required - params
        assert not missing, f"kernel_fn missing kwargs: {sorted(missing)}"
