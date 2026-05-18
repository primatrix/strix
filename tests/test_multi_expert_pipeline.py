"""Structural tests for kernels/multi_expert_pipeline.py (no TPU required)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "multi_expert_pipeline.py"


class TestFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/multi_expert_pipeline.py not found"


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

    def test_defines_multi_expert_ffn(self):
        assert "multi_expert_ffn" in self._func_names()

    def test_defines_ref_multi_expert_ffn(self):
        assert "_ref_multi_expert_ffn" in self._func_names()

    def test_defines_kernel_body(self):
        assert "_multi_expert_kernel" in self._func_names()

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

    def test_default_shape_values(self):
        ds = self.config["default_shape"]
        assert ds["num_tokens"] == 256
        assert ds["hidden_size"] == 8192
        assert ds["intermediate_size"] == 2048

    def test_default_num_experts(self):
        ds = self.config["default_shape"]
        assert ds["num_experts"] == 8

    def test_default_bf(self):
        assert self.config["bf"] == 256

    def test_dtype_is_bfloat16(self):
        assert self.config["dtype"] == "bfloat16"
        assert self.config["weight_dtype"] == "bfloat16"

    def test_act_fn_is_silu(self):
        assert self.config["act_fn"] == "silu"

    def test_tpu_type_is_v7x(self):
        assert self.config["tpu_type"] == "v7x"


class TestMultiExpertFfnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_accepts_required_kwargs(self):
        sig = inspect.signature(self.module.multi_expert_ffn)
        params = set(sig.parameters.keys())
        required = {"tokens", "w1", "w2", "w3", "act_fn", "bf", "num_experts"}
        missing = required - params
        assert not missing, f"multi_expert_ffn missing params: {sorted(missing)}"

    def test_num_experts_is_keyword_only(self):
        sig = inspect.signature(self.module.multi_expert_ffn)
        p = sig.parameters["num_experts"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY


class TestKernelFnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_kernel_fn_accepts_required_kwargs(self):
        sig = inspect.signature(self.module.kernel_fn)
        params = set(sig.parameters.keys())
        required = {
            "num_tokens", "hidden_size", "intermediate_size",
            "dtype", "weight_dtype", "act_fn", "bf", "num_experts",
        }
        missing = required - params
        assert not missing, f"kernel_fn missing kwargs: {sorted(missing)}"


class TestFp8Config:
    @pytest.fixture(autouse=True)
    def _extract_config(self):
        source = KERNEL_FILE.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "config":
                        module = ast.Module(body=[node], type_ignores=[])
                        ast.fix_missing_locations(module)
                        code = compile(module, str(KERNEL_FILE), "exec")
                        ns = {}
                        exec(code, ns)
                        self.config = ns["config"]
                        return
        pytest.fail("config not found")

    def test_config_has_quant_block_k(self):
        assert "quant_block_k" in self.config
        assert self.config["quant_block_k"] == 128


class TestFp8MultiExpertFfnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_multi_expert_ffn_accepts_scale_params(self):
        sig = inspect.signature(self.module.multi_expert_ffn)
        params = set(sig.parameters.keys())
        for p in ("w1_scale", "w2_scale", "w3_scale", "quant_block_k"):
            assert p in params, f"multi_expert_ffn missing param: {p}"

    def test_kernel_fn_accepts_fp8_params(self):
        sig = inspect.signature(self.module.kernel_fn)
        params = set(sig.parameters.keys())
        for p in ("quant_block_k",):
            assert p in params, f"kernel_fn missing param: {p}"


class TestFp8RefSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib
        self.module = importlib.import_module("kernels.multi_expert_pipeline")

    def test_ref_accepts_scale_params(self):
        sig = inspect.signature(self.module._ref_multi_expert_ffn)
        params = set(sig.parameters.keys())
        for p in ("w1_scale", "w2_scale", "w3_scale", "quant_block_k"):
            assert p in params, f"_ref_multi_expert_ffn missing param: {p}"
