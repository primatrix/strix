"""Ablation variants of the DMA-only double-buffer expert kernel.

Each variant removes or minimises one overhead source while keeping
the rest identical, so the measured overhead difference isolates that
single factor.

Variants
--------
baseline          – unchanged ``double_buffer_expert``
no_named_scope    – remove ``jax.named_scope`` wrapper
no_mem_constraint – remove ``pltpu.with_memory_space_constraint`` on inputs
minimal_scratch   – drop unused ``b_y_acc_vmem`` (fp32 accumulator)
minimal_sems      – collapse ``weight_sems (2,3)`` → ``(2,1)``; reuse 1 sem
no_writeback      – skip output VMEM→HBM DMA in kernel body
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .double_buffer_expert import (
    _ALLOWED_CONFIGS,
    _double_buffer_expert_kernel,
    double_buffer_expert,
)


# ---------------------------------------------------------------------------
# Helpers shared across variants
# ---------------------------------------------------------------------------

def _validate(tokens, w1, w2, w3, bf):
    bt, d = tokens.shape
    f_full = w1.shape[1]
    if d != 8192:
        raise ValueError(f"d must be 8192, got {d}")
    if f_full % bf != 0 or f_full < bf:
        raise ValueError(f"intermediate_size ({f_full}) must be a positive multiple of bf ({bf})")
    if (bt, bf) not in _ALLOWED_CONFIGS:
        raise ValueError(f"(bt={bt}, bf={bf}) not in {sorted(_ALLOWED_CONFIGS)}")
    return bt, d, f_full


def _make_pallas_call(kernel_fn, *, bt, d, bf, f_full, dtype, weight_dtype,
                      scratch_shapes, scope_name):
    """Build a pallas_call with the standard grid/block specs."""
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        functools.partial(kernel_fn, bf=bf, intermediate_size=f_full),
        out_shape=jax.ShapeDtypeStruct((bt, d), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm, hbm, hbm],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
        name=scope_name,
    )


def _standard_scratch(bt, d, bf, dtype, weight_dtype):
    """The 6-buffer + 3-semaphore scratch layout from the baseline."""
    return (
        pltpu.VMEM((2, d, bf), weight_dtype),       # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),       # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),       # b_w2_x2_vmem
        pltpu.VMEM((bt, d), dtype),                  # b_x_vmem
        pltpu.VMEM((bt, d), jnp.float32),            # b_y_acc_vmem
        pltpu.VMEM((bt, d), dtype),                  # b_y_out_vmem
        pltpu.SemaphoreType.DMA((2, 3)),             # weight_sems
        pltpu.SemaphoreType.DMA((1,)),               # x_sem
        pltpu.SemaphoreType.DMA((1,)),               # y_out_sem
    )


def _hbm_inputs(tokens, w1, w2, w3):
    return (
        pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
        pltpu.with_memory_space_constraint(w1, pltpu.HBM),
        pltpu.with_memory_space_constraint(w2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w3, pltpu.HBM),
    )


# ---------------------------------------------------------------------------
# Variant A: no_named_scope
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["bf"])
def no_named_scope(tokens, w1, w2, w3, *, bf=512):
    bt, d, f_full = _validate(tokens, w1, w2, w3, bf)
    dtype, weight_dtype = tokens.dtype, w1.dtype
    scratch = _standard_scratch(bt, d, bf, dtype, weight_dtype)
    scope_name = f"ablation-no-scope-bt{bt}-bf{bf}"
    kernel = _make_pallas_call(
        _double_buffer_expert_kernel, bt=bt, d=d, bf=bf, f_full=f_full,
        dtype=dtype, weight_dtype=weight_dtype,
        scratch_shapes=scratch, scope_name=scope_name,
    )
    # No jax.named_scope wrapper — call kernel directly
    return kernel(*_hbm_inputs(tokens, w1, w2, w3))


# ---------------------------------------------------------------------------
# Variant B: no_mem_constraint
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["bf"])
def no_mem_constraint(tokens, w1, w2, w3, *, bf=512):
    bt, d, f_full = _validate(tokens, w1, w2, w3, bf)
    dtype, weight_dtype = tokens.dtype, w1.dtype
    scratch = _standard_scratch(bt, d, bf, dtype, weight_dtype)
    scope_name = f"ablation-no-mem-bt{bt}-bf{bf}"
    kernel = _make_pallas_call(
        _double_buffer_expert_kernel, bt=bt, d=d, bf=bf, f_full=f_full,
        dtype=dtype, weight_dtype=weight_dtype,
        scratch_shapes=scratch, scope_name=scope_name,
    )
    # No with_memory_space_constraint — pass arrays directly
    return jax.named_scope(scope_name)(kernel)(tokens, w1, w2, w3)


# ---------------------------------------------------------------------------
# Variant C: minimal_scratch — drop b_y_acc_vmem (unused in DMA-only kernel)
# ---------------------------------------------------------------------------

def _kernel_minimal_scratch(
    tokens_hbm, w1_hbm, w2_hbm, w3_hbm, output_hbm,
    # 5 scratch buffers (no b_y_acc_vmem)
    b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
    b_x_vmem, b_y_out_vmem,
    # Semaphores
    weight_sems, x_sem, y_out_sem,
    *, bf, intermediate_size,
):
    """Kernel body identical to baseline but with one fewer scratch buffer."""
    n_w = intermediate_size // bf

    def start_load_x():
        pltpu.make_async_copy(src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0]).start()
    def wait_load_x():
        pltpu.make_async_copy(src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0]).wait()
    def start_fetch_w1(slot, ti):
        pltpu.make_async_copy(src_ref=w1_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).start()
    def wait_fetch_w1(slot):
        pltpu.make_async_copy(src_ref=b_w1_x2_vmem.at[slot], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).wait()
    def start_fetch_w3(slot, ti):
        pltpu.make_async_copy(src_ref=w3_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 1]).start()
    def wait_fetch_w3(slot):
        pltpu.make_async_copy(src_ref=b_w3_x2_vmem.at[slot], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 1]).wait()
    def start_fetch_w2(slot, ti):
        pltpu.make_async_copy(src_ref=w2_hbm.at[pl.ds(ti*bf, bf), :], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 2]).start()
    def wait_fetch_w2(slot):
        pltpu.make_async_copy(src_ref=b_w2_x2_vmem.at[slot], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 2]).wait()
    def drain_tile(slot):
        wait_fetch_w2(slot)

    start_load_x(); start_fetch_w1(0, 0); start_fetch_w3(0, 0); start_fetch_w2(0, 0)
    wait_load_x(); wait_fetch_w1(0); wait_fetch_w3(0)
    if n_w >= 2:
        start_fetch_w1(1, 1); start_fetch_w3(1, 1); start_fetch_w2(1, 1)
    drain_tile(slot=0)
    for tile in range(1, n_w - 1):
        slot = tile % 2; ns = 1 - slot
        start_fetch_w1(ns, tile+1); start_fetch_w3(ns, tile+1); start_fetch_w2(ns, tile+1)
        wait_fetch_w1(slot); wait_fetch_w3(slot); drain_tile(slot)
    if n_w >= 2:
        ls = (n_w-1) % 2; wait_fetch_w1(ls); wait_fetch_w3(ls); drain_tile(ls)

    pltpu.make_async_copy(src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0]).start()
    pltpu.make_async_copy(src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0]).wait()


@functools.partial(jax.jit, static_argnames=["bf"])
def minimal_scratch(tokens, w1, w2, w3, *, bf=512):
    bt, d, f_full = _validate(tokens, w1, w2, w3, bf)
    dtype, weight_dtype = tokens.dtype, w1.dtype
    scratch = (
        pltpu.VMEM((2, d, bf), weight_dtype),       # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),       # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),       # b_w2_x2_vmem
        pltpu.VMEM((bt, d), dtype),                  # b_x_vmem
        # no b_y_acc_vmem
        pltpu.VMEM((bt, d), dtype),                  # b_y_out_vmem
        pltpu.SemaphoreType.DMA((2, 3)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
    )
    scope_name = f"ablation-min-scratch-bt{bt}-bf{bf}"
    kernel = _make_pallas_call(
        _kernel_minimal_scratch, bt=bt, d=d, bf=bf, f_full=f_full,
        dtype=dtype, weight_dtype=weight_dtype,
        scratch_shapes=scratch, scope_name=scope_name,
    )
    return jax.named_scope(scope_name)(kernel)(*_hbm_inputs(tokens, w1, w2, w3))


# ---------------------------------------------------------------------------
# Variant D: minimal_sems — (2,3) → (2,1), reuse single sem for all weights
# ---------------------------------------------------------------------------

def _kernel_minimal_sems(
    tokens_hbm, w1_hbm, w2_hbm, w3_hbm, output_hbm,
    b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
    b_x_vmem, b_y_acc_vmem, b_y_out_vmem,
    weight_sems,  # (2,1) — single sem per slot
    x_sem, y_out_sem,
    *, bf, intermediate_size,
):
    """Kernel body using a single shared semaphore per slot for all 3 weights."""
    n_w = intermediate_size // bf

    def start_load_x():
        pltpu.make_async_copy(src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0]).start()
    def wait_load_x():
        pltpu.make_async_copy(src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0]).wait()

    # All 3 weights share weight_sems[slot, 0]
    def start_fetch_w1(slot, ti):
        pltpu.make_async_copy(src_ref=w1_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).start()
    def start_fetch_w3(slot, ti):
        pltpu.make_async_copy(src_ref=w3_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).start()
    def start_fetch_w2(slot, ti):
        pltpu.make_async_copy(src_ref=w2_hbm.at[pl.ds(ti*bf, bf), :], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).start()

    def wait_all_weights(slot):
        """Wait for all 3 weight DMAs on the shared semaphore (count=3)."""
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot], dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    start_load_x(); start_fetch_w1(0, 0); start_fetch_w3(0, 0); start_fetch_w2(0, 0)
    wait_load_x()
    # Must wait for all 3 weight DMAs (they all increment the same sem)
    # Semaphore counts: after 3 starts, wait needs sem count = 3
    # But pltpu.make_async_copy.wait() waits for count=1 by default.
    # We need 3 waits on the same semaphore.
    pltpu.make_async_copy(src_ref=b_w1_x2_vmem.at[0], dst_ref=b_w1_x2_vmem.at[0], sem=weight_sems.at[0, 0]).wait()
    pltpu.make_async_copy(src_ref=b_w3_x2_vmem.at[0], dst_ref=b_w3_x2_vmem.at[0], sem=weight_sems.at[0, 0]).wait()
    pltpu.make_async_copy(src_ref=b_w2_x2_vmem.at[0], dst_ref=b_w2_x2_vmem.at[0], sem=weight_sems.at[0, 0]).wait()

    if n_w >= 2:
        start_fetch_w1(1, 1); start_fetch_w3(1, 1); start_fetch_w2(1, 1)

    for tile in range(1, n_w - 1):
        slot = tile % 2; ns = 1 - slot
        start_fetch_w1(ns, tile+1); start_fetch_w3(ns, tile+1); start_fetch_w2(ns, tile+1)
        pltpu.make_async_copy(src_ref=b_w1_x2_vmem.at[slot], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).wait()
        pltpu.make_async_copy(src_ref=b_w3_x2_vmem.at[slot], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).wait()
        pltpu.make_async_copy(src_ref=b_w2_x2_vmem.at[slot], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).wait()

    if n_w >= 2:
        ls = (n_w-1) % 2
        pltpu.make_async_copy(src_ref=b_w1_x2_vmem.at[ls], dst_ref=b_w1_x2_vmem.at[ls], sem=weight_sems.at[ls, 0]).wait()
        pltpu.make_async_copy(src_ref=b_w3_x2_vmem.at[ls], dst_ref=b_w3_x2_vmem.at[ls], sem=weight_sems.at[ls, 0]).wait()
        pltpu.make_async_copy(src_ref=b_w2_x2_vmem.at[ls], dst_ref=b_w2_x2_vmem.at[ls], sem=weight_sems.at[ls, 0]).wait()

    pltpu.make_async_copy(src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0]).start()
    pltpu.make_async_copy(src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0]).wait()


@functools.partial(jax.jit, static_argnames=["bf"])
def minimal_sems(tokens, w1, w2, w3, *, bf=512):
    bt, d, f_full = _validate(tokens, w1, w2, w3, bf)
    dtype, weight_dtype = tokens.dtype, w1.dtype
    scratch = (
        pltpu.VMEM((2, d, bf), weight_dtype),
        pltpu.VMEM((2, d, bf), weight_dtype),
        pltpu.VMEM((2, bf, d), weight_dtype),
        pltpu.VMEM((bt, d), dtype),
        pltpu.VMEM((bt, d), jnp.float32),
        pltpu.VMEM((bt, d), dtype),
        pltpu.SemaphoreType.DMA((2, 1)),  # collapsed from (2,3)
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
    )
    scope_name = f"ablation-min-sems-bt{bt}-bf{bf}"
    kernel = _make_pallas_call(
        _kernel_minimal_sems, bt=bt, d=d, bf=bf, f_full=f_full,
        dtype=dtype, weight_dtype=weight_dtype,
        scratch_shapes=scratch, scope_name=scope_name,
    )
    return jax.named_scope(scope_name)(kernel)(*_hbm_inputs(tokens, w1, w2, w3))


# ---------------------------------------------------------------------------
# Variant E: no_writeback — skip output VMEM→HBM DMA
# ---------------------------------------------------------------------------

def _kernel_no_writeback(
    tokens_hbm, w1_hbm, w2_hbm, w3_hbm, output_hbm,
    b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
    b_x_vmem, b_y_acc_vmem, b_y_out_vmem,
    weight_sems, x_sem, y_out_sem,
    *, bf, intermediate_size,
):
    """Kernel body with no output write-back."""
    n_w = intermediate_size // bf

    def start_load_x():
        pltpu.make_async_copy(src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0]).start()
    def wait_load_x():
        pltpu.make_async_copy(src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0]).wait()
    def start_fetch_w1(slot, ti):
        pltpu.make_async_copy(src_ref=w1_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).start()
    def wait_fetch_w1(slot):
        pltpu.make_async_copy(src_ref=b_w1_x2_vmem.at[slot], dst_ref=b_w1_x2_vmem.at[slot], sem=weight_sems.at[slot, 0]).wait()
    def start_fetch_w3(slot, ti):
        pltpu.make_async_copy(src_ref=w3_hbm.at[:, pl.ds(ti*bf, bf)], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 1]).start()
    def wait_fetch_w3(slot):
        pltpu.make_async_copy(src_ref=b_w3_x2_vmem.at[slot], dst_ref=b_w3_x2_vmem.at[slot], sem=weight_sems.at[slot, 1]).wait()
    def start_fetch_w2(slot, ti):
        pltpu.make_async_copy(src_ref=w2_hbm.at[pl.ds(ti*bf, bf), :], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 2]).start()
    def wait_fetch_w2(slot):
        pltpu.make_async_copy(src_ref=b_w2_x2_vmem.at[slot], dst_ref=b_w2_x2_vmem.at[slot], sem=weight_sems.at[slot, 2]).wait()
    def drain_tile(slot):
        wait_fetch_w2(slot)

    start_load_x(); start_fetch_w1(0, 0); start_fetch_w3(0, 0); start_fetch_w2(0, 0)
    wait_load_x(); wait_fetch_w1(0); wait_fetch_w3(0)
    if n_w >= 2:
        start_fetch_w1(1, 1); start_fetch_w3(1, 1); start_fetch_w2(1, 1)
    drain_tile(slot=0)
    for tile in range(1, n_w - 1):
        slot = tile % 2; ns = 1 - slot
        start_fetch_w1(ns, tile+1); start_fetch_w3(ns, tile+1); start_fetch_w2(ns, tile+1)
        wait_fetch_w1(slot); wait_fetch_w3(slot); drain_tile(slot)
    if n_w >= 2:
        ls = (n_w-1) % 2; wait_fetch_w1(ls); wait_fetch_w3(ls); drain_tile(ls)

    # NO output write-back — kernel ends here


@functools.partial(jax.jit, static_argnames=["bf"])
def no_writeback(tokens, w1, w2, w3, *, bf=512):
    bt, d, f_full = _validate(tokens, w1, w2, w3, bf)
    dtype, weight_dtype = tokens.dtype, w1.dtype
    scratch = _standard_scratch(bt, d, bf, dtype, weight_dtype)
    scope_name = f"ablation-no-wb-bt{bt}-bf{bf}"
    kernel = _make_pallas_call(
        _kernel_no_writeback, bt=bt, d=d, bf=bf, f_full=f_full,
        dtype=dtype, weight_dtype=weight_dtype,
        scratch_shapes=scratch, scope_name=scope_name,
    )
    return jax.named_scope(scope_name)(kernel)(*_hbm_inputs(tokens, w1, w2, w3))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "baseline": double_buffer_expert,
    "no_named_scope": no_named_scope,
    "no_mem_constraint": no_mem_constraint,
    "minimal_scratch": minimal_scratch,
    "minimal_sems": minimal_sems,
    "no_writeback": no_writeback,
}
