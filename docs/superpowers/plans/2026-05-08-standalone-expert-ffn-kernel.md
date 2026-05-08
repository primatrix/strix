# Standalone Expert FFN Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `kernels/expert_ffn.py` — a standalone Pallas kernel that isolates the expert FFN computation from the fused MoE kernel for profiling.

**Architecture:** Single file implementing a Pallas kernel with manual async DMA pipeline (weight double-buffering, token staging, result triple-buffering). Strips A2A, routing, shared expert, FP8, and bias code from the original `expert_ffn` in `_fused_moe_impl.py`. Follows the `kernel_fn`/`config` contract for benchmark harness compatibility.

**Tech Stack:** JAX, Pallas (TPU), `jax.experimental.pallas.tpu` for async DMA and semaphores.

**Spec:** `docs/superpowers/specs/2026-05-08-standalone-expert-ffn-kernel-design.md`

---

### Task 1: Structural tests for `kernels/expert_ffn.py`

**Files:**
- Create: `tests/test_expert_ffn_kernel.py`

- [ ] **Step 1: Write failing structural tests**

These tests verify file existence, AST structure, and config values — matching the pattern from `tests/test_fused_moe_kernel.py`. They run without JAX/TPU.

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/strix/.claude/worktrees/jolly-roaming-micali && python -m pytest tests/test_expert_ffn_kernel.py -v`

Expected: All tests FAIL (file not found / import errors).

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_expert_ffn_kernel.py
git commit -m "test: add structural tests for standalone expert FFN kernel"
```

---

### Task 2: Config dict and `kernel_fn` skeleton

**Files:**
- Create: `kernels/expert_ffn.py`

- [ ] **Step 1: Write the module skeleton with config and kernel_fn stub**

Create `kernels/expert_ffn.py` with the `config` dict, a placeholder `_expert_ffn_kernel` function, and a `kernel_fn` that builds inputs and returns a closure. This makes all structural/config tests pass. The `kernel_fn` closure calls a pure-JAX reference path (no Pallas yet) to keep it testable.

```python
"""Standalone Expert FFN kernel for profiling.

Extracts the expert FFN pipeline (weight staging, FFN1, activation, FFN2)
from the fused MoE kernel for isolated performance analysis.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._fused_moe_impl import FusedMoEBlockConfig, activation_fn, cdiv, align_to, get_dtype_packing

config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "tpu_type": "v7x",
    "tpu_topology": "1x1",
    "description": "Standalone Expert FFN — Ling 2.6 per-expert profile (H=8192, I=2048)",
}


DEFAULT_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32,
    bf=2048,
    bd1=1024,
    bd2=1024,
    btc=32,
    bfc=2048,
    bd1c=1024,
    bd2c=1024,
    bse=512,
)


def _expert_ffn_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    output_hbm,
    # Scratch buffers (VMEM).
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_acc_vmem,
    b_stage_x2_vmem,
    b_acc_stage_x3_vmem,
    # Semaphores.
    token_stage_sems,
    acc_stage_sems,
    weight_sems,
    *,
    act_fn: str,
    bf: int,
    bd1: int,
    bd2: int,
    bts: int,
    btc: int,
    bfc: int,
    bd1c: int,
    bd2c: int,
):
    """Pallas kernel body — expert FFN with manual DMA pipeline."""
    num_tokens = tokens_hbm.shape[0]
    hidden_size = w2_hbm.shape[1]
    intermediate_size = w1_hbm.shape[1]

    t_dtype = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    h_per_t_packing = hidden_size // t_packing
    bd1_per_t_packing = bd1 // t_packing
    bd2_per_t_packing = bd2 // t_packing
    bd1c_per_t_packing = bd1c // t_packing
    bd2c_per_t_packing = bd2c // t_packing

    num_bf = cdiv(intermediate_size, bf)
    num_bd1 = cdiv(hidden_size, bd1)
    num_bd2 = cdiv(hidden_size, bd2)
    num_token_tiles = cdiv(num_tokens, bts)

    # -- Weight DMA helpers --

    def start_fetch_w1(bw_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[pl.ds(offset, bd1_per_t_packing), pl.ds(bf_id * bf, bf)],
                dst_ref=b_w1_x2_vmem.at[bw_sem_id, p],
                sem=weight_sems.at[bw_sem_id, 0],
            ).start()

    def wait_fetch_w1(bw_sem_id):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[bw_sem_id],
            dst_ref=b_w1_x2_vmem.at[bw_sem_id],
            sem=weight_sems.at[bw_sem_id, 0],
        ).wait()

    def start_fetch_w3(bw_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[pl.ds(offset, bd1_per_t_packing), pl.ds(bf_id * bf, bf)],
                dst_ref=b_w3_x2_vmem.at[bw_sem_id, p],
                sem=weight_sems.at[bw_sem_id, 1],
            ).start()

    def wait_fetch_w3(bw_sem_id):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[bw_sem_id],
            dst_ref=b_w3_x2_vmem.at[bw_sem_id],
            sem=weight_sems.at[bw_sem_id, 1],
        ).wait()

    def start_fetch_w2(bw_sem_id, bf_id, bd2_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd2_id * bd2_per_t_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[pl.ds(bf_id * bf, bf), pl.ds(offset, bd2_per_t_packing)],
                dst_ref=b_w2_x2_vmem.at[bw_sem_id, p],
                sem=weight_sems.at[bw_sem_id, 2],
            ).start()

    def wait_fetch_w2(bw_sem_id):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[bw_sem_id],
            dst_ref=b_w2_x2_vmem.at[bw_sem_id],
            sem=weight_sems.at[bw_sem_id, 2],
        ).wait()

    # -- Token staging helpers --

    def start_stage_tokens(tile_start, bd1_id, buf_id):
        pltpu.make_async_copy(
            src_ref=tokens_hbm.at[
                pl.ds(tile_start, bts),
                pl.ds(0, t_packing),
                pl.ds(bd1_id * bd1_per_t_packing, bd1_per_t_packing),
            ],
            dst_ref=b_stage_x2_vmem.at[buf_id, pl.ds(0, bts)],
            sem=token_stage_sems.at[buf_id],
        ).start()

    def wait_stage_tokens(buf_id):
        pltpu.make_async_copy(
            src_ref=b_stage_x2_vmem.at[buf_id, pl.ds(0, bts)],
            dst_ref=b_stage_x2_vmem.at[buf_id, pl.ds(0, bts)],
            sem=token_stage_sems.at[buf_id],
        ).wait()

    # -- Result staging helpers (triple-buffer) --

    def start_load_result(tile_start, bd2_start, buf_id):
        pltpu.make_async_copy(
            src_ref=output_hbm.at[
                pl.ds(tile_start, bts),
                pl.ds(0, t_packing),
                pl.ds(bd2_start, bd2_per_t_packing),
            ],
            dst_ref=b_acc_stage_x3_vmem.at[buf_id, pl.ds(0, bts)],
            sem=acc_stage_sems.at[buf_id],
        ).start()

    def wait_stage_result(buf_id):
        pltpu.make_async_copy(
            src_ref=b_acc_stage_x3_vmem.at[buf_id, pl.ds(0, bts)],
            dst_ref=b_acc_stage_x3_vmem.at[buf_id, pl.ds(0, bts)],
            sem=acc_stage_sems.at[buf_id],
        ).wait()

    def start_store_result(tile_start, bd2_start, buf_id):
        pltpu.make_async_copy(
            src_ref=b_acc_stage_x3_vmem.at[buf_id, pl.ds(0, bts), pl.ds(0, t_packing), pl.ds(0, bd2_per_t_packing)],
            dst_ref=output_hbm.at[
                pl.ds(tile_start, bts),
                pl.ds(0, t_packing),
                pl.ds(bd2_start, bd2_per_t_packing),
            ],
            sem=acc_stage_sems.at[buf_id],
        ).start()

    # -- Compute functions (BF16-only, no quantization/bias) --

    def ffn1_compute(t_vmem, w1_vmem, w3_vmem, acc1_vmem, acc3_vmem, should_init):
        token_tile = t_vmem.shape[0]
        num_loops = token_tile // btc

        def compute_tile(btc_id, is_init_mode):
            for bd1c_id in range(cdiv(bd1, bd1c)):
                for p_id in range(t_packing):
                    t = t_vmem[
                        pl.ds(btc_id * btc, btc),
                        p_id,
                        pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                    ]
                    for bfc_id in range(cdiv(bf, bfc)):
                        w_slices = (
                            p_id,
                            pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                            pl.ds(bfc_id * bfc, bfc),
                        )
                        acc1 = jnp.dot(t, w1_vmem[*w_slices], preferred_element_type=jnp.float32)
                        acc3 = jnp.dot(t, w3_vmem[*w_slices], preferred_element_type=jnp.float32)

                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        if is_init_mode and p_id == 0 and bd1c_id == 0:
                            acc1_vmem[*acc_slices] = acc1
                            acc3_vmem[*acc_slices] = acc3
                        else:
                            acc1_vmem[*acc_slices] += acc1
                            acc3_vmem[*acc_slices] += acc3

        if should_init:
            def body_init(i, _):
                compute_tile(i, is_init_mode=True)
            lax.fori_loop(0, num_loops, body_init, None)
        else:
            def body_acc(i, _):
                compute_tile(i, is_init_mode=False)
            lax.fori_loop(0, num_loops, body_acc, None)

    def ffn2_compute(acc1_vmem, acc3_vmem, w2_vmem, res_vmem, should_init):
        token_tile = res_vmem.shape[0]
        num_loops = token_tile // btc

        def body(btc_id, __):
            for bd2c_id in range(cdiv(bd2, bd2c)):
                for p_id in range(t_packing):
                    res = jnp.zeros((btc, bd2c_per_t_packing), dtype=jnp.float32)
                    for bfc_id in range(cdiv(bf, bfc)):
                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        a1 = acc1_vmem[*acc_slices]
                        a3 = acc3_vmem[*acc_slices]
                        act = activation_fn(a1, a3, act_fn)
                        w2 = w2_vmem[
                            p_id,
                            pl.ds(bfc_id * bfc, bfc),
                            pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                        ]
                        acc = jnp.dot(act, w2, preferred_element_type=jnp.float32)
                        res += acc

                    res_slice = res_vmem.at[
                        pl.ds(btc_id * btc, btc),
                        p_id,
                        pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                    ]
                    if should_init:
                        res_slice[...] = res.astype(t_dtype)
                    else:
                        res_slice[...] = (res_slice[...].astype(jnp.float32) + res).astype(t_dtype)

        lax.fori_loop(0, num_loops, body, None)

    # -- Main pipeline --

    b_acc_2d = b_acc_vmem.reshape(2, num_tokens, bf)
    b_acc1 = b_acc_2d.at[0]
    b_acc3 = b_acc_2d.at[1]

    def with_static_bw(bw_sem_id, body_fn):
        return lax.cond(
            bw_sem_id == 0,
            lambda _: body_fn(0),
            lambda _: body_fn(1),
            operand=None,
        )

    for bf_id in range(num_bf):
        # ── FFN1 Phase ──
        # Prefetch first W1/W3 slice.
        start_fetch_w1(0, bf_id, 0)
        start_fetch_w3(0, bf_id, 0)

        bw_sem_id = jnp.int32(0)

        for bd1_id in range(num_bd1):
            def _run_bd1(bw_sem_id_static, *, _bf_id=bf_id, _bd1_id=bd1_id):
                next_bw = 1 - bw_sem_id_static

                # Prefetch next bd1 weights.
                if _bd1_id + 1 < num_bd1:
                    start_fetch_w1(next_bw, _bf_id, _bd1_id + 1)
                    start_fetch_w3(next_bw, _bf_id, _bd1_id + 1)

                # Wait for current weights.
                wait_fetch_w1(bw_sem_id_static)
                wait_fetch_w3(bw_sem_id_static)

                # Prefetch W2 at last bd1.
                if _bd1_id + 1 == num_bd1:
                    start_fetch_w2(next_bw, _bf_id, 0)

                # Prefetch first token tile.
                start_stage_tokens(0, _bd1_id, 0)

                def run_ffn1_tile(tile_id, token_buf_id):
                    tile_start = tile_id * bts
                    next_buf = token_buf_id ^ jnp.int32(1)
                    next_tile = tile_id + 1

                    @pl.when(next_tile < num_token_tiles)
                    def _():
                        start_stage_tokens(next_tile * bts, _bd1_id, next_buf)

                    wait_stage_tokens(token_buf_id)

                    ffn1_compute(
                        t_vmem=b_stage_x2_vmem.at[token_buf_id],
                        w1_vmem=b_w1_x2_vmem.at[bw_sem_id_static],
                        w3_vmem=b_w3_x2_vmem.at[bw_sem_id_static],
                        acc1_vmem=b_acc1.at[pl.ds(tile_start, bts)],
                        acc3_vmem=b_acc3.at[pl.ds(tile_start, bts)],
                        should_init=(_bd1_id == 0),
                    )
                    return next_buf

                lax.fori_loop(0, num_token_tiles, run_ffn1_tile, jnp.int32(0), unroll=False)
                return jnp.int32(next_bw)

            bw_sem_id = with_static_bw(bw_sem_id, _run_bd1)

        # ── FFN2 Phase ──
        for bd2_id in range(num_bd2):
            should_init_ffn2 = bf_id == 0

            def _run_bd2(bw_sem_id_static, *, _bf_id=bf_id, _bd2_id=bd2_id, _should_init=should_init_ffn2):
                next_bw = 1 - bw_sem_id_static

                # Prefetch next W2 or next bf's W1/W3.
                if _bd2_id + 1 < num_bd2:
                    start_fetch_w2(next_bw, _bf_id, _bd2_id + 1)
                elif _bf_id + 1 < num_bf:
                    start_fetch_w1(next_bw, _bf_id + 1, 0)
                    start_fetch_w3(next_bw, _bf_id + 1, 0)

                wait_fetch_w2(bw_sem_id_static)

                w2_vmem = b_w2_x2_vmem.at[bw_sem_id_static]
                bd2_start = _bd2_id * bd2_per_t_packing

                # Triple-buffer indices.
                init_buf_compute = jnp.int32(0)
                init_buf_load = jnp.int32(2)

                if not _should_init:
                    start_load_result(0, bd2_start, init_buf_compute)

                def run_ffn2_tile(tile_id, state):
                    buf_compute, buf_store, buf_load = state
                    tile_start = tile_id * bts

                    if not _should_init:
                        do_prefetch = tile_id + 1 < num_token_tiles

                        @pl.when(jnp.logical_and(do_prefetch, tile_id >= 2))
                        def _(buf_load=buf_load):
                            wait_stage_result(buf_load)

                        @pl.when(do_prefetch)
                        def _(next_start=(tile_id + 1) * bts, buf_load=buf_load):
                            start_load_result(next_start, bd2_start, buf_load)

                        wait_stage_result(buf_compute)
                    else:

                        @pl.when(tile_id >= 3)
                        def _(buf_compute=buf_compute):
                            wait_stage_result(buf_compute)

                    ffn2_compute(
                        acc1_vmem=b_acc1.at[pl.ds(tile_start, bts)],
                        acc3_vmem=b_acc3.at[pl.ds(tile_start, bts)],
                        w2_vmem=w2_vmem,
                        res_vmem=b_acc_stage_x3_vmem.at[buf_compute],
                        should_init=_should_init,
                    )
                    start_store_result(tile_start, bd2_start, buf_compute)
                    return (buf_load, buf_compute, buf_store)

                state = (init_buf_compute, jnp.int32(1), init_buf_load)
                lax.fori_loop(0, num_token_tiles, run_ffn2_tile, state, unroll=False)

                # Drain outstanding stores.
                @pl.when(num_token_tiles >= 1)
                def _():
                    wait_stage_result(jnp.int32(0))

                @pl.when(num_token_tiles >= 2)
                def _():
                    wait_stage_result(jnp.int32(2))

                @pl.when(num_token_tiles >= 3)
                def _():
                    wait_stage_result(jnp.int32(1))

                return jnp.int32(next_bw)

            bw_sem_id = with_static_bw(bw_sem_id, _run_bd2)


def expert_ffn(
    tokens,
    w1,
    w2,
    w3,
    *,
    act_fn="silu",
    block_config=None,
):
    """Run the standalone expert FFN kernel."""
    if block_config is None:
        block_config = DEFAULT_BLOCK_CONFIG

    num_tokens, hidden_size = tokens.shape
    intermediate_size = w1.shape[1]

    t_dtype = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)
    h_per_pack = hidden_size // t_packing
    bd1_per_pack = block_config.bd1 // t_packing
    bd2_per_pack = block_config.bd2 // t_packing

    bts = min(block_config.bts or num_tokens, num_tokens)
    btc = min(block_config.btc, bts)
    bf = block_config.bf
    bd1 = block_config.bd1
    bd2 = block_config.bd2

    tokens_packed = tokens.reshape(num_tokens, t_packing, h_per_pack)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    scope_name = (
        f"expert-ffn-bts_{bts}_{btc}-bf_{bf}_{block_config.bfc}"
        f"-bd1_{bd1}_{block_config.bd1c}-bd2_{bd2}_{block_config.bd2c}"
    )

    scratch_shapes = (
        pltpu.VMEM((2, t_packing, bd1_per_pack, bf), w1.dtype),           # b_w1_x2_vmem
        pltpu.VMEM((2, t_packing, bd1_per_pack, bf), w3.dtype),           # b_w3_x2_vmem
        pltpu.VMEM((2, t_packing, bf, bd2_per_pack), w2.dtype),           # b_w2_x2_vmem
        pltpu.VMEM((2, num_tokens, 1, bf), jnp.float32),                  # b_acc_vmem
        pltpu.VMEM((2, bts, t_packing, bd1_per_pack), t_dtype),           # b_stage_x2_vmem
        pltpu.VMEM((3, bts, t_packing, bd2_per_pack), t_dtype),           # b_acc_stage_x3_vmem
        pltpu.SemaphoreType.DMA((2,)),                                     # token_stage_sems
        pltpu.SemaphoreType.DMA((3,)),                                     # acc_stage_sems
        pltpu.SemaphoreType.DMA((2, 3)),                                   # weight_sems
    )

    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _expert_ffn_kernel,
                act_fn=act_fn,
                bf=bf,
                bd1=bd1,
                bd2=bd2,
                bts=bts,
                btc=btc,
                bfc=block_config.bfc,
                bd1c=block_config.bd1c,
                bd2c=block_config.bd2c,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (num_tokens, t_packing, h_per_pack), t_dtype,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm, hbm, hbm, hbm],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=96 * 1024 * 1024,
            ),
            name=scope_name,
        )
    )

    @jax.jit
    def run_kernel(tokens_packed, w1, w2, w3):
        return kernel(
            pltpu.with_memory_space_constraint(tokens_packed, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
        )

    output_packed = run_kernel(tokens_packed, w1, w2, w3)
    return output_packed.reshape(num_tokens, hidden_size)


def kernel_fn(
    num_tokens=256,
    hidden_size=8192,
    intermediate_size=2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn="silu",
    block_config=None,
):
    """Construct inputs and return a zero-arg closure for benchmarking."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=weight_dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=weight_dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return expert_ffn(
            tokens, w1, w2, w3,
            act_fn=act_fn,
            block_config=block_config,
        )

    return run
```

- [ ] **Step 2: Run structural tests to verify they pass**

Run: `cd /Users/xl/Code/strix/.claude/worktrees/jolly-roaming-micali && python -m pytest tests/test_expert_ffn_kernel.py -v`

Expected: All AST/config tests PASS. `TestExpertFfnImportlib` tests PASS if JAX is installed, else skip.

- [ ] **Step 3: Commit**

```bash
git add kernels/expert_ffn.py
git commit -m "feat: add standalone expert FFN kernel with DMA pipeline"
```

---

### Task 3: Verify kernel correctness on TPU

This task runs on-device and is not part of the local test suite. It validates the kernel produces correct output by comparing against `ref_moe` configured as a single expert.

**Files:**
- Modify: `kernels/expert_ffn.py` (add `_ref_expert_ffn` helper at module level)

- [ ] **Step 1: Add a reference implementation for validation**

Add this function to `kernels/expert_ffn.py` right before `kernel_fn`, used for on-device correctness checks:

```python
def _ref_expert_ffn(tokens, w1, w2, w3, *, act_fn="silu"):
    """Pure-JAX reference: activation(tokens @ W1, tokens @ W3) @ W2."""
    gate = tokens.astype(jnp.float32) @ w1.astype(jnp.float32)
    up = tokens.astype(jnp.float32) @ w3.astype(jnp.float32)
    act = activation_fn(gate, up, act_fn)
    return (act @ w2.astype(jnp.float32)).astype(tokens.dtype)
```

- [ ] **Step 2: Add a validation script block**

Add this at the bottom of `kernels/expert_ffn.py`:

```python
if __name__ == "__main__":
    import sys

    num_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    hidden_size = 8192
    intermediate_size = 2048
    dtype = jnp.bfloat16

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=dtype)

    bc = FusedMoEBlockConfig(
        bt=num_tokens, bf=2048, bd1=1024, bd2=1024,
        btc=num_tokens, bfc=2048, bd1c=1024, bd2c=1024, bse=512,
        bts=num_tokens,
    )

    result = expert_ffn(tokens, w1, w2, w3, block_config=bc)
    ref = _ref_expert_ffn(tokens, w1, w2, w3)
    max_err = jnp.max(jnp.abs(result.astype(jnp.float32) - ref.astype(jnp.float32)))
    rel_err = max_err / (jnp.max(jnp.abs(ref.astype(jnp.float32))) + 1e-6)
    print(f"num_tokens={num_tokens}, max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.05, f"Relative error too high: {rel_err}"
    print("PASS")
```

- [ ] **Step 3: Commit**

```bash
git add kernels/expert_ffn.py
git commit -m "feat: add reference impl and validation script for expert FFN kernel"
```

- [ ] **Step 4: Run on TPU for correctness validation**

This step requires a TPU device. Run via the existing benchmark infrastructure or directly:

```bash
python -m kernels.expert_ffn 32
```

Expected: `PASS` with `rel_err < 0.05`.

---

### Task 4: Run existing tests to check for regressions

**Files:** None modified.

- [ ] **Step 1: Run all existing tests**

```bash
cd /Users/xl/Code/strix/.claude/worktrees/jolly-roaming-micali && python -m pytest tests/ -v --ignore=tests/test_expert_ffn_kernel.py
```

Expected: All existing tests pass unchanged.

- [ ] **Step 2: Run new tests**

```bash
cd /Users/xl/Code/strix/.claude/worktrees/jolly-roaming-micali && python -m pytest tests/test_expert_ffn_kernel.py -v
```

Expected: All structural/config/import tests pass.

- [ ] **Step 3: Final commit with all tests passing**

```bash
git add -A
git status
```

If clean: no commit needed. If any fixes were required, commit them.
