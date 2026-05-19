"""Tests for kernels/multi_expert_pipeline_fp8.py (no TPU required)."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "multi_expert_pipeline_fp8.py"


class TestFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file()


class TestModuleStructure:
    @pytest.fixture(autouse=True)
    def _parse_module(self):
        self.source = KERNEL_FILE.read_text()
        self.tree = ast.parse(self.source)

    def _func_names(self):
        return [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def test_dequant_functions_exist(self):
        names = self._func_names()
        for fn in ("dequant_w1", "dequant_w3", "dequant_w2"):
            assert fn in names, f"{fn} not found in kernel"

    def test_dequant_uses_reshape_not_fori_loop(self):
        """After refactoring, dequant bodies should use reshape, not fori_loop."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "dequant_w1", "dequant_w3", "dequant_w2",
            ):
                body_src = ast.dump(node)
                assert "fori_loop" not in body_src, (
                    f"{node.name} still uses fori_loop"
                )
                assert "reshape" in body_src, (
                    f"{node.name} should use reshape"
                )


class TestReshapeDequantEquivalence:
    """Verify reshape-based dequant matches the per-group loop."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.quant_block_k = 256
        self.K = 1024
        self.N = 256
        self.n_sg = self.K // self.quant_block_k

        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)
        bf16_w = jax.random.normal(k1, (self.K, self.N), dtype=jnp.bfloat16)
        self.w_fp8 = bf16_w.astype(jnp.float8_e4m3fn)
        self.scale = jax.random.uniform(
            k2, (self.n_sg, 1, self.N), dtype=jnp.float32,
            minval=0.5, maxval=2.0,
        )

    def _dequant_fori_style(self):
        """Reproduce the old fori_loop logic in pure JAX."""
        result = jnp.zeros((self.K, self.N), dtype=jnp.bfloat16)
        for sg_id in range(self.n_sg):
            sg_off = sg_id * self.quant_block_k
            w_slice = self.w_fp8[sg_off : sg_off + self.quant_block_k, :]
            s = self.scale[sg_id].reshape(1, self.N)
            dq = (w_slice.astype(jnp.float32) * jnp.broadcast_to(
                s, (self.quant_block_k, self.N)
            )).astype(jnp.bfloat16)
            result = result.at[sg_off : sg_off + self.quant_block_k, :].set(dq)
        return result

    def _dequant_reshape_style(self):
        """The proposed reshape approach."""
        w_f32 = self.w_fp8.astype(jnp.float32).reshape(
            self.n_sg, self.quant_block_k, self.N
        )
        return (w_f32 * self.scale).astype(jnp.bfloat16).reshape(self.K, self.N)

    def test_reshape_matches_fori(self):
        fori_result = self._dequant_fori_style()
        reshape_result = self._dequant_reshape_style()
        np.testing.assert_array_equal(fori_result, reshape_result)

    def test_reshape_dequant_shape(self):
        result = self._dequant_reshape_style()
        assert result.shape == (self.K, self.N)
        assert result.dtype == jnp.bfloat16
