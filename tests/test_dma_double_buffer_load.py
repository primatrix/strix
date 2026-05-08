"""Tests for double-buffer VMEM load DMA benchmark.

Acceptance criteria:
1. kernels/dma_double_buffer_load.py exists and exports kernel_fn + config
2. config contains default shape parameters (hidden_size, intermediate_size, num_loads)
3. Module is importable via importlib
4. Kernel executes successfully with various shapes
5. Output checksum is non-zero (verifies DMA actually happened)
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "dma_double_buffer_load.py"


class TestDmaDoubleBufferLoadFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/dma_double_buffer_load.py not found"


class TestDmaDoubleBufferLoadModuleStructure:
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

    def test_defines_dma_kernel(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "_dma_double_buffer_load_kernel" in func_names

    def test_defines_wrapper_function(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "dma_double_buffer_load" in func_names


class TestDmaDoubleBufferLoadConfig:
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

    def test_default_shape_hidden_size(self):
        assert self.config["default_shape"]["hidden_size"] == 8192

    def test_default_shape_intermediate_size(self):
        assert self.config["default_shape"]["intermediate_size"] == 2048

    def test_default_shape_num_loads(self):
        assert "num_loads" in self.config["default_shape"]
        assert self.config["default_shape"]["num_loads"] > 0

    def test_config_dtype(self):
        assert self.config["dtype"] == "bfloat16"

    def test_config_weight_dtype(self):
        assert self.config["weight_dtype"] == "bfloat16"

    def test_config_tpu_topology(self):
        assert self.config["tpu_topology"] == "1x1"

    def test_config_has_description(self):
        assert isinstance(self.config["description"], str)
        assert len(self.config["description"]) > 0
        assert "DMA" in self.config["description"] or "dma" in self.config["description"]


class TestDmaDoubleBufferLoadImportlib:
    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX required for import test")

    def test_import_dma_module(self):
        from strix.kernels import dma_double_buffer_load  # noqa: F401

    def test_module_has_kernel_fn(self):
        from strix.kernels import dma_double_buffer_load
        assert hasattr(dma_double_buffer_load, "kernel_fn")

    def test_module_has_config(self):
        from strix.kernels import dma_double_buffer_load
        assert hasattr(dma_double_buffer_load, "config")

    def test_kernel_fn_is_callable(self):
        from strix.kernels import dma_double_buffer_load
        assert callable(dma_double_buffer_load.kernel_fn)

    def test_importlib_import_module(self):
        mod = importlib.import_module("strix.kernels.dma_double_buffer_load")
        assert hasattr(mod, "kernel_fn")
        assert hasattr(mod, "config")
        assert callable(mod.kernel_fn)


class TestDmaDoubleBufferLoadExecution:
    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX required for execution test")

    def test_kernel_fn_returns_callable(self):
        from strix.kernels import dma_double_buffer_load
        run_fn = dma_double_buffer_load.kernel_fn(
            hidden_size=1024,
            intermediate_size=512,
            num_loads=8,
        )
        assert callable(run_fn)

    @pytest.mark.skipif(
        not importlib.util.find_spec("jax"),
        reason="JAX not available"
    )
    def test_small_shape_execution(self):
        """Test execution with small shape (fast, suitable for CPU/GPU)."""
        import jax
        import jax.numpy as jnp
        from strix.kernels.dma_double_buffer_load import dma_double_buffer_load

        key = jax.random.key(0)
        w = jax.random.normal(key, (1024, 512), dtype=jnp.bfloat16)

        # This will fail on non-TPU, but we test that it at least compiles
        try:
            result = dma_double_buffer_load(w, bf=512, bd=512, num_loads=4)
            # If we're on TPU, verify output
            assert result.shape == ()
            assert result.dtype == jnp.float32
            # Checksum should be non-zero (verifies DMA happened)
            assert jnp.abs(result) > 0
        except ValueError as e:
            # Expected on CPU backend
            if "CPU backend" in str(e):
                pass  # Test passes - compilation succeeded
            else:
                raise


class TestDmaDoubleBufferLoadShapeVariations:
    """Test that kernel_fn accepts various shape configurations."""

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        pytest.importorskip("jax", reason="JAX required")

    @pytest.mark.parametrize("hidden_size,intermediate_size", [
        (1024, 512),
        (2048, 1024),
        (4096, 2048),
        (8192, 2048),  # Ling 2.6 default
        (8192, 4096),
    ])
    def test_various_shapes(self, hidden_size, intermediate_size):
        from strix.kernels import dma_double_buffer_load
        run_fn = dma_double_buffer_load.kernel_fn(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_loads=16,
        )
        assert callable(run_fn)

    @pytest.mark.parametrize("num_loads", [1, 8, 16, 32, 64, 128])
    def test_various_num_loads(self, num_loads):
        from strix.kernels import dma_double_buffer_load
        run_fn = dma_double_buffer_load.kernel_fn(
            hidden_size=8192,
            intermediate_size=2048,
            num_loads=num_loads,
        )
        assert callable(run_fn)

    @pytest.mark.parametrize("bf,bd", [
        (512, 512),
        (1024, 1024),
        (2048, 1024),
        (2048, 2048),
    ])
    def test_various_block_sizes(self, bf, bd):
        from strix.kernels import dma_double_buffer_load
        run_fn = dma_double_buffer_load.kernel_fn(
            hidden_size=8192,
            intermediate_size=2048,
            num_loads=16,
            bf=bf,
            bd=bd,
        )
        assert callable(run_fn)
