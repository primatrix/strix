# FP8 Reshape Pre-Dequant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fori_loop-based dequant in `multi_expert_pipeline_fp8.py` with reshape-based bulk multiply to reduce dequant overhead on the critical path.

**Architecture:** Three nested dequant functions (`dequant_w1`, `dequant_w3`, `dequant_w2`) inside `_multi_expert_kernel_fp8` are changed from a `lax.fori_loop(0, n_sg, ..., unroll=n_sg)` pattern to a single reshape → broadcast-multiply → reshape expression. Everything outside these three function bodies is untouched.

**Tech Stack:** JAX, Pallas (TPU), jnp operations

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `kernels/multi_expert_pipeline_fp8.py:230-261` | Modify | Replace three dequant function bodies |
| `tests/test_multi_expert_pipeline_fp8.py` | Create | Structural + unit tests (no TPU) |

---

### Task 1: Add structural and unit tests for the FP8 kernel dequant

**Files:**
- Create: `tests/test_multi_expert_pipeline_fp8.py`

- [ ] **Step 1: Write the test file**

This test file verifies: (a) the dequant functions exist and do NOT use `lax.fori_loop` after refactoring, (b) the pure-JAX `_dequant_weight` helper produces correct results, (c) the reshape-based dequant is numerically equivalent to the fori_loop approach for a single weight tile.

```python
"""Tests for kernels/multi_expert_pipeline_fp8.py (no TPU required)."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
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
        jnp.testing.assert_array_equal(fori_result, reshape_result)

    def test_reshape_dequant_shape(self):
        result = self._dequant_reshape_style()
        assert result.shape == (self.K, self.N)
        assert result.dtype == jnp.bfloat16
```

- [ ] **Step 2: Run tests to verify they pass (structural test will FAIL — dequant still uses fori_loop)**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_multi_expert_pipeline_fp8.py -v`

Expected:
- `TestFileExists::test_kernel_file_exists` — PASS
- `TestModuleStructure::test_dequant_functions_exist` — PASS
- `TestModuleStructure::test_dequant_uses_reshape_not_fori_loop` — **FAIL** (still uses fori_loop)
- `TestReshapeDequantEquivalence::test_reshape_matches_fori` — PASS
- `TestReshapeDequantEquivalence::test_reshape_dequant_shape` — PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_multi_expert_pipeline_fp8.py
git commit -m "test(fp8): add structural and equivalence tests for reshape dequant"
```

---

### Task 2: Replace dequant_w1 and dequant_w3 with reshape pattern

**Files:**
- Modify: `kernels/multi_expert_pipeline_fp8.py:226-250`

- [ ] **Step 1: Replace dequant_w1 and dequant_w3**

Replace lines 226-250 with:

```python
    # -- Dequant: fp8 → bf16 in VMEM --
    # Reshape-based bulk multiply (pattern from gmm_v2.py).
    # Reshape is zero-cost in Pallas; gives compiler full operation graph.

    def dequant_w1(slot):
        w_fp8 = b_w1_x2_vmem[slot]
        s = b_w1_scale_x2_vmem[slot]
        w_f32 = w_fp8.astype(jnp.float32).reshape(n_sg, quant_block_k, bf)
        b_w1_dq_vmem[...] = (w_f32 * s).astype(jnp.bfloat16).reshape(d, bf)

    def dequant_w3(slot):
        w_fp8 = b_w3_x2_vmem[slot]
        s = b_w3_scale_x2_vmem[slot]
        w_f32 = w_fp8.astype(jnp.float32).reshape(n_sg, quant_block_k, bf)
        b_w3_dq_vmem[...] = (w_f32 * s).astype(jnp.bfloat16).reshape(d, bf)
```

- [ ] **Step 2: Run the structural test to check it passes now for w1/w3**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_multi_expert_pipeline_fp8.py::TestModuleStructure -v`

Expected: `test_dequant_uses_reshape_not_fori_loop` — still **FAIL** (dequant_w2 not yet changed). That's OK — we fix w2 next.

- [ ] **Step 3: Commit**

```bash
git add kernels/multi_expert_pipeline_fp8.py
git commit -m "refactor(fp8): reshape dequant for w1 and w3"
```

---

### Task 3: Replace dequant_w2 with reshape pattern

**Files:**
- Modify: `kernels/multi_expert_pipeline_fp8.py:252-261`

- [ ] **Step 1: Replace dequant_w2**

Replace lines 252-261 with:

```python
    def dequant_w2(slot):
        w_fp8 = b_w2_x2_vmem[slot]
        s = b_w2_scale_x2_vmem[slot]
        w_f32 = w_fp8.astype(jnp.float32).reshape(n_sg2, quant_block_k, d)
        b_w2_dq_vmem[...] = (w_f32 * s).astype(jnp.bfloat16).reshape(bf, d)
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_multi_expert_pipeline_fp8.py -v`

Expected: ALL PASS — including `test_dequant_uses_reshape_not_fori_loop`.

- [ ] **Step 3: Verify the `lax` import is still needed**

`lax` is still used in `expert_body` (line 390: `lax.fori_loop(0, num_experts, expert_body, ...)`), so the import stays. No cleanup needed.

- [ ] **Step 4: Commit**

```bash
git add kernels/multi_expert_pipeline_fp8.py
git commit -m "refactor(fp8): reshape dequant for w2 — completes dequant refactor"
```

---

### Task 4: Run full test suite to verify no regressions

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full FP8 test file**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_multi_expert_pipeline_fp8.py -v`

Expected: ALL PASS.

- [ ] **Step 2: Run the bf16 pipeline test to verify no accidental changes**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/test_multi_expert_pipeline.py -v`

Expected: ALL PASS.

- [ ] **Step 3: Run any other related tests**

Run: `cd /Users/xl/Code/strix && python -m pytest tests/ -v --timeout=60 2>&1 | tail -30`

Expected: No new failures.
