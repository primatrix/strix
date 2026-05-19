"""Multi-expert double-buffer FFN pipeline — EP decode.

Extends double_buffer_expert.py to process N experts sequentially on one
device, with DMA overlap between adjacent experts.

Spec: docs/superpowers/specs/2026-05-12-multi-expert-pipeline-design.md
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._fused_moe_impl import activation_fn


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
        "num_experts": 8,
    },
    "dtype": "float8_e4m3fn",
    "weight_dtype": "float8_e4m3fn",
    "act_fn": "silu",
    "bf": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Multi-expert double-buffer FFN — EP decode "
        "(N experts sequential, expert-to-expert DMA overlap)"
    ),
}


def _multi_expert_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    output_hbm,
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_x_x2_vmem,
    b_y_acc_vmem,
    b_y_out_vmem,
    weight_sems,
    x_sem,
    y_out_sem,
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
    num_experts: int,
):
    """Pallas kernel body — multi-expert pipeline."""
    n_w = intermediate_size // bf

    # -- DMA helpers --

    def start_load_x(x_slot, expert_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=tokens_hbm.at[expert_idx],
            dst_ref=b_x_x2_vmem.at[x_slot],
            sem=x_sem.at[x_slot],
        ).start(priority=priority)

    def wait_load_x(x_slot):
        pltpu.make_async_copy(
            src_ref=b_x_x2_vmem.at[x_slot],
            dst_ref=b_x_x2_vmem.at[x_slot],
            sem=x_sem.at[x_slot],
        ).wait()

    def start_fetch_w1(slot, expert_idx, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start(priority=priority)

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w3(slot, expert_idx, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start(priority=priority)

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def start_fetch_w2(slot, expert_idx, tile_idx, priority=0):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[expert_idx, pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start(priority=priority)

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()

    def start_writeback(expert_idx, y_slot):
        pltpu.make_async_copy(
            src_ref=b_y_out_vmem.at[y_slot],
            dst_ref=output_hbm.at[expert_idx],
            sem=y_out_sem.at[y_slot],
        ).start()

    def wait_writeback(y_slot):
        pltpu.make_async_copy(
            src_ref=b_y_out_vmem.at[y_slot],
            dst_ref=b_y_out_vmem.at[y_slot],
            sem=y_out_sem.at[y_slot],
        ).wait()

    # -- Compute --

    def compute_tile(x_slot, w_slot, is_first_tile):
        x = b_x_x2_vmem[x_slot]
        w1 = b_w1_x2_vmem[w_slot]
        w3 = b_w3_x2_vmem[w_slot]
        gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)
        up = jnp.dot(x, w3, preferred_element_type=jnp.float32)
        act_up = activation_fn(gate, up, act_fn)

        wait_fetch_w2(w_slot)

        w2 = b_w2_x2_vmem[w_slot]
        partial = jnp.dot(
            act_up.astype(jnp.float8_e4m3fn), w2,
            preferred_element_type=jnp.float32,
        )
        if is_first_tile:
            b_y_acc_vmem[...] = partial
        else:
            b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial

    # -- Global Prologue --

    x_slot = 0
    start_load_x(x_slot=0, expert_idx=0)
    start_fetch_w1(0, 0, 0, priority=1)
    start_fetch_w3(0, 0, 0, priority=1)
    start_fetch_w2(0, 0, 0)

    if n_w >= 2:
        start_fetch_w1(1, 0, 1)
        start_fetch_w3(1, 0, 1)
        start_fetch_w2(1, 0, 1)

    wait_load_x(x_slot)

    # -- Expert Loop (lax.fori_loop — single trace, no unroll) --

    def expert_body(e, x_slot):
        next_xs = 1 - x_slot

        # First tile
        wait_fetch_w1(0)
        wait_fetch_w3(0)
        compute_tile(x_slot, w_slot=0, is_first_tile=True)

        # Steady state: tiles [1, n_w - 1)
        for tile in range(1, n_w - 1):
            w_slot = tile % 2
            next_w = 1 - w_slot
            start_fetch_w1(next_w, e, tile + 1)
            start_fetch_w3(next_w, e, tile + 1)
            start_fetch_w2(next_w, e, tile + 1)
            wait_fetch_w1(w_slot)
            wait_fetch_w3(w_slot)
            compute_tile(x_slot, w_slot, is_first_tile=False)

        # Epilogue: last tile — prefetch next expert's first weight
        # tile (slot 0) *before* the last compute to overlap DMA with
        # the final tile's MXU work.  Tile 1 (slot 1) cannot move here
        # because last_w == 1 and the compute still reads from that slot.
        if n_w >= 2:
            last_w = (n_w - 1) % 2
            @pl.when(e < num_experts - 1)
            def _():
                start_load_x(next_xs, e + 1, priority=1)
                start_fetch_w1(0, e + 1, 0)
                start_fetch_w3(0, e + 1, 0)
                start_fetch_w2(0, e + 1, 0)
            wait_fetch_w1(last_w)
            wait_fetch_w3(last_w)
            compute_tile(x_slot, last_w, is_first_tile=False)

        # Expert boundary: double-buffered writeback.
        # Wait for the previous user of this y_slot (expert e-2).
        @pl.when(e >= 2)
        def _():
            wait_writeback(x_slot)

        b_y_out_vmem[x_slot] = b_y_acc_vmem[...].astype(jnp.float8_e4m3fn)
        start_writeback(e, x_slot)

        @pl.when(e < num_experts - 1)
        def _():
            if n_w >= 2:
                start_fetch_w1(1, e + 1, 1)
                start_fetch_w3(1, e + 1, 1)
                start_fetch_w2(1, e + 1, 1)

        # No wait_writeback — DMA overlaps with next expert compute.

        @pl.when(e < num_experts - 1)
        def _():
            wait_load_x(next_xs)

        return next_xs

    lax.fori_loop(0, num_experts, expert_body, jnp.int32(0), unroll=False)

    # Drain outstanding writebacks.  With double-buffered output, up to
    # two slots may have in-flight DMAs (the last two experts).
    last_y_slot = (num_experts - 1) % 2
    wait_writeback(last_y_slot)
    if num_experts >= 2:
        wait_writeback(1 - last_y_slot)


@functools.partial(jax.jit, static_argnames=["act_fn", "bf", "num_experts"])
def multi_expert_ffn(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
) -> jax.Array:
    """Run the multi-expert double-buffer FFN kernel (pure fp8).

    All inputs/outputs are fp8.  MXU does fp8×fp8→f32 accumulation;
    act_up is cast back to fp8 before the down projection.

    Shapes:
      tokens: (num_experts, bt, d) fp8
      w1:     (num_experts, d, f)  fp8
      w2:     (num_experts, f, d)  fp8
      w3:     (num_experts, d, f)  fp8
    Returns:
      output: (num_experts, bt, d) fp8
    """
    n_exp, bt, d = tokens.shape
    f_full = w1.shape[2]
    if n_exp != num_experts:
        raise ValueError(
            f"tokens.shape[0]={n_exp} != num_experts={num_experts}"
        )
    if f_full % bf != 0:
        raise ValueError(
            f"intermediate_size={f_full} must be divisible by bf={bf}"
        )
    if f_full // bf < 2:
        raise ValueError(
            f"Need >= 2 weight tiles (intermediate_size/bf >= 2); "
            f"got {f_full}/{bf}={f_full // bf}"
        )
    if w1.shape != (num_experts, d, f_full):
        raise ValueError(f"w1.shape={w1.shape} must be ({num_experts}, {d}, {f_full})")
    if w2.shape != (num_experts, f_full, d):
        raise ValueError(f"w2.shape={w2.shape} must be ({num_experts}, {f_full}, {d})")
    if w3.shape != (num_experts, d, f_full):
        raise ValueError(f"w3.shape={w3.shape} must be ({num_experts}, {d}, {f_full})")

    dtype = tokens.dtype
    weight_dtype = w1.dtype

    scratch_shapes = (
        pltpu.VMEM((2, d, bf), weight_dtype),       # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),        # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),        # b_w2_x2_vmem
        pltpu.VMEM((2, bt, d), dtype),               # b_x_x2_vmem
        pltpu.VMEM((bt, d), jnp.float32),            # b_y_acc_vmem
        pltpu.VMEM((2, bt, d), dtype),               # b_y_out_vmem (double-buffered)
        pltpu.SemaphoreType.DMA((2, 3)),             # weight_sems[slot, channel]
        pltpu.SemaphoreType.DMA((2,)),               # x_sem[x_slot]
        pltpu.SemaphoreType.DMA((2,)),               # y_out_sem[y_slot]
    )

    scope_name = f"multi-expert-bt{bt}-bf{bf}-e{num_experts}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = pl.pallas_call(
        functools.partial(
            _multi_expert_kernel,
            act_fn=act_fn,
            bf=bf,
            intermediate_size=f_full,
            num_experts=num_experts,
        ),
        out_shape=jax.ShapeDtypeStruct((num_experts, bt, d), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm, hbm, hbm],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=scope_name,
    )
    return jax.named_scope(scope_name)(kernel)(
        pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
        pltpu.with_memory_space_constraint(w1, pltpu.HBM),
        pltpu.with_memory_space_constraint(w2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w3, pltpu.HBM),
    )


def _ref_multi_expert_ffn(tokens, w1, w2, w3, *, act_fn="silu"):
    """Pure-JAX reference matching kernel precision: fp8×fp8→f32, act→fp8→dot."""
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        gate = x @ w1[e].astype(jnp.float32)
        up = x @ w3[e].astype(jnp.float32)
        act = activation_fn(gate, up, act_fn)
        act_fp8 = act.astype(jnp.float8_e4m3fn).astype(jnp.float32)
        out = (act_fp8 @ w2[e].astype(jnp.float32)).astype(tokens.dtype)
        outputs.append(out)
    return jnp.stack(outputs)


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.float8_e4m3fn,
    weight_dtype=jnp.float8_e4m3fn,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int = 8,
    **_kwargs,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts, num_tokens, hidden_size), dtype=jnp.bfloat16).astype(dtype)
    w1 = jax.random.normal(k2, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w2 = jax.random.normal(k3, (num_experts, intermediate_size, hidden_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w3 = jax.random.normal(k4, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)

    def run():
        return multi_expert_ffn(
            tokens, w1, w2, w3,
            act_fn=act_fn, bf=bf, num_experts=num_experts,
        )

    return run


if __name__ == "__main__":
    num_experts_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bt = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    bf_arg = 512

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts_arg, bt, 6144), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    w1 = jax.random.normal(k2, (num_experts_arg, 6144, 2048), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    w2 = jax.random.normal(k3, (num_experts_arg, 2048, 6144), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    w3 = jax.random.normal(k4, (num_experts_arg, 6144, 2048), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)

    result = multi_expert_ffn(
        tokens, w1, w2, w3, bf=bf_arg, num_experts=num_experts_arg,
    )
    ref = _ref_multi_expert_ffn(tokens, w1, w2, w3)

    max_errs = []
    rel_errs = []
    for e in range(num_experts_arg):
        r = result[e].astype(jnp.float32)
        f = ref[e].astype(jnp.float32)
        me = jnp.max(jnp.abs(r - f))
        re = me / (jnp.max(jnp.abs(f)) + 1e-6)
        max_errs.append(float(me))
        rel_errs.append(float(re))
        print(f"  expert {e}: max_abs_err={me:.4f}, rel_err={re:.4f}")

    worst_rel = max(rel_errs)
    print(f"num_experts={num_experts_arg}, bt={bt}, bf={bf_arg}, worst_rel_err={worst_rel:.4f}")
    assert worst_rel < 0.05, f"worst rel_err too high: {worst_rel}"
    print("PASS")
