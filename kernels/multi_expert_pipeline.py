"""Multi-expert double-buffer FFN pipeline — EP decode.

Extends double_buffer_expert.py to process N experts sequentially on one
device, with DMA overlap between adjacent experts.

Supports both bf16 and fp8 (float8_e4m3fn) weights.  When fp8 scales are
provided, uses a t_packing-aligned layout: tokens are bitcast to uint32
(packing 2 bf16 per word), weights and scales carry a leading t_packing
dimension, and the inner loop processes one sub-word stripe at a time.

The `bd1c` parameter controls the contracting dimension per compute tile:
  bd1c = quant_block_k (128): exact per-block scaling, 48 dots/tile
  bd1c = 256: accumulate 2 scale groups per dot, 24 dots/tile
  bd1c = 512: accumulate 4 scale groups per dot, 12 dots/tile

Spec: docs/superpowers/specs/2026-05-18-fused-moe-v3-design.md
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
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 256,
    "quant_block_k": 128,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Multi-expert double-buffer FFN — EP decode "
        "(N experts sequential, expert-to-expert DMA overlap)"
    ),
}

# t_packing for bf16 tokens: 2 bf16 values per 32-bit VREG word
T_PACKING = 2
T_BITWIDTH = 16


def _multi_expert_kernel(
    # HBM inputs
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    w1_scale_hbm,    # None when bf16
    w2_scale_hbm,    # None when bf16
    w3_scale_hbm,    # None when bf16
    # HBM output
    output_hbm,
    # VMEM scratch
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_w1_scale_x2_vmem,  # None when bf16
    b_w3_scale_x2_vmem,  # None when bf16
    b_w2_scale_x2_vmem,  # None when bf16
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
    quant_block_k: int | None,
    bd1c: int | None,
):
    """Pallas kernel body — multi-expert pipeline."""
    n_w = intermediate_size // bf
    use_fp8 = w1_scale_hbm is not None

    if use_fp8:
        tp = T_PACKING
        d_per_tp = b_x_x2_vmem.shape[2]  # uint32 units = d // tp
        bd1c_per_tp = bd1c // tp
        n_bd1c_per_tp = d_per_tp // bd1c_per_tp
        sg_per_dot = bd1c_per_tp // quant_block_k
        n_sg_per_tp = b_w1_scale_x2_vmem.shape[2]  # d_per_tp // qbk
        bf_per_tp = bf // tp
        n_sg2_per_tp = b_w2_scale_x2_vmem.shape[2]  # bf_per_tp // qbk

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
        if use_fp8:
            for p in range(tp):
                pltpu.make_async_copy(
                    src_ref=w1_hbm.at[
                        expert_idx, p, :, pl.ds(tile_idx * bf, bf)
                    ],
                    dst_ref=b_w1_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 0],
                ).start(priority=priority)
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        expert_idx, p, :, pl.ds(0, 1),
                        pl.ds(tile_idx * bf, bf),
                    ],
                    dst_ref=b_w1_scale_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 0],
                ).start(priority=priority)
        else:
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    expert_idx, :, pl.ds(tile_idx * bf, bf)
                ],
                dst_ref=b_w1_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 0],
            ).start(priority=priority)

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()
        if use_fp8:
            pltpu.make_async_copy(
                src_ref=b_w1_scale_x2_vmem.at[slot],
                dst_ref=b_w1_scale_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 0],
            ).wait()

    def start_fetch_w3(slot, expert_idx, tile_idx, priority=1):
        if use_fp8:
            for p in range(tp):
                pltpu.make_async_copy(
                    src_ref=w3_hbm.at[
                        expert_idx, p, :, pl.ds(tile_idx * bf, bf)
                    ],
                    dst_ref=b_w3_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 1],
                ).start(priority=priority)
                pltpu.make_async_copy(
                    src_ref=w3_scale_hbm.at[
                        expert_idx, p, :, pl.ds(0, 1),
                        pl.ds(tile_idx * bf, bf),
                    ],
                    dst_ref=b_w3_scale_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 1],
                ).start(priority=priority)
        else:
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[
                    expert_idx, :, pl.ds(tile_idx * bf, bf)
                ],
                dst_ref=b_w3_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 1],
            ).start(priority=priority)

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()
        if use_fp8:
            pltpu.make_async_copy(
                src_ref=b_w3_scale_x2_vmem.at[slot],
                dst_ref=b_w3_scale_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 1],
            ).wait()

    def start_fetch_w2(slot, expert_idx, tile_idx, priority=0):
        if use_fp8:
            for p in range(tp):
                pltpu.make_async_copy(
                    src_ref=w2_hbm.at[
                        expert_idx, p,
                        pl.ds(tile_idx * bf_per_tp, bf_per_tp), :,
                    ],
                    dst_ref=b_w2_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 2],
                ).start(priority=priority)
                pltpu.make_async_copy(
                    src_ref=w2_scale_hbm.at[
                        expert_idx, p,
                        pl.ds(
                            tile_idx * bf_per_tp // quant_block_k,
                            bf_per_tp // quant_block_k,
                        ),
                        pl.ds(0, 1), :,
                    ],
                    dst_ref=b_w2_scale_x2_vmem.at[slot, p],
                    sem=weight_sems.at[slot, 2],
                ).start(priority=priority)
        else:
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    expert_idx, pl.ds(tile_idx * bf, bf), :
                ],
                dst_ref=b_w2_x2_vmem.at[slot],
                sem=weight_sems.at[slot, 2],
            ).start(priority=priority)

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()
        if use_fp8:
            pltpu.make_async_copy(
                src_ref=b_w2_scale_x2_vmem.at[slot],
                dst_ref=b_w2_scale_x2_vmem.at[slot],
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
        bt_k = b_x_x2_vmem.shape[1]

        if use_fp8:
            d_k = d_per_tp * tp  # full hidden_size

            # ---- FFN1: gate + up ----
            gate = jnp.zeros((bt_k, bf), dtype=jnp.float32)
            up = jnp.zeros_like(gate)

            x_b32_full = b_x_x2_vmem.at[x_slot][...]  # (bt, d//tp) uint32

            for p_id in range(tp):
                # Extract bf16 sub-element from packed uint32
                if p_id == 0:
                    x_b32 = x_b32_full
                else:
                    x_b32 = x_b32_full >> (p_id * T_BITWIDTH)
                t_bf16 = pltpu.bitcast(
                    x_b32.astype(jnp.int16), jnp.bfloat16
                )  # (bt, d//tp) bf16

                for bd1c_id in range(n_bd1c_per_tp):
                    bd1c_off = bd1c_id * bd1c_per_tp

                    # Accumulate across scale groups within compute tile
                    dot_acc1 = jnp.zeros((bt_k, bf), dtype=jnp.float32)
                    dot_acc3 = jnp.zeros_like(dot_acc1)

                    for sg_sub in range(sg_per_dot):
                        sg_off = bd1c_off + sg_sub * quant_block_k
                        t_slice = t_bf16[:, pl.ds(sg_off, quant_block_k)]

                        w1_tile = b_w1_x2_vmem[
                            w_slot, p_id,
                            pl.ds(sg_off, quant_block_k),
                            pl.ds(0, bf),
                        ]
                        dot_acc1 = dot_acc1 + jnp.dot(
                            t_slice, w1_tile,
                            preferred_element_type=jnp.float32,
                        )

                        w3_tile = b_w3_x2_vmem[
                            w_slot, p_id,
                            pl.ds(sg_off, quant_block_k),
                            pl.ds(0, bf),
                        ]
                        dot_acc3 = dot_acc3 + jnp.dot(
                            t_slice, w3_tile,
                            preferred_element_type=jnp.float32,
                        )

                    # Scale: use the last sub-group's scale (megablox pattern)
                    last_sg = bd1c_id * sg_per_dot + (sg_per_dot - 1)
                    s1 = b_w1_scale_x2_vmem[
                        w_slot, p_id, pl.ds(last_sg, 1), 0,
                        pl.ds(0, bf),
                    ].reshape(1, bf)
                    gate = gate + dot_acc1 * jnp.broadcast_to(
                        s1, dot_acc1.shape
                    )

                    s3 = b_w3_scale_x2_vmem[
                        w_slot, p_id, pl.ds(last_sg, 1), 0,
                        pl.ds(0, bf),
                    ].reshape(1, bf)
                    up = up + dot_acc3 * jnp.broadcast_to(
                        s3, dot_acc3.shape
                    )

            # ---- Global SiLU activation ----
            act = activation_fn(gate, up, act_fn)  # (bt, bf) f32

            # ---- FFN2: down projection ----
            wait_fetch_w2(w_slot)
            partial = jnp.zeros((bt_k, d_k), dtype=jnp.float32)

            for p_id in range(tp):
                for sg_id in range(n_sg2_per_tp):
                    sg_off_act = p_id * bf_per_tp + sg_id * quant_block_k
                    act_slice = act[:, pl.ds(sg_off_act, quant_block_k)]

                    w2_tile = b_w2_x2_vmem[
                        w_slot, p_id,
                        pl.ds(sg_id * quant_block_k, quant_block_k), :,
                    ]
                    d_val = jnp.dot(
                        act_slice, w2_tile,
                        preferred_element_type=jnp.float32,
                    )
                    s = b_w2_scale_x2_vmem[
                        w_slot, p_id, pl.ds(sg_id, 1), 0, :,
                    ].reshape(1, d_k)
                    partial = partial + d_val * jnp.broadcast_to(
                        s, d_val.shape
                    )

        else:
            # bf16 path — unchanged from original
            x = b_x_x2_vmem[x_slot]
            w1 = b_w1_x2_vmem[w_slot]
            w3 = b_w3_x2_vmem[w_slot]
            gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)
            up = jnp.dot(x, w3, preferred_element_type=jnp.float32)
            act_up = activation_fn(gate, up, act_fn)

            wait_fetch_w2(w_slot)

            w2 = b_w2_x2_vmem[w_slot]
            partial = jnp.dot(
                act_up, w2, preferred_element_type=jnp.float32,
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

        # Epilogue: last tile
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
        @pl.when(e >= 2)
        def _():
            wait_writeback(x_slot)

        b_y_out_vmem[x_slot] = b_y_acc_vmem[...].astype(jnp.bfloat16)
        start_writeback(e, x_slot)

        @pl.when(e < num_experts - 1)
        def _():
            if n_w >= 2:
                start_fetch_w1(1, e + 1, 1)
                start_fetch_w3(1, e + 1, 1)
                start_fetch_w2(1, e + 1, 1)

        @pl.when(e < num_experts - 1)
        def _():
            wait_load_x(next_xs)

        return next_xs

    lax.fori_loop(0, num_experts, expert_body, jnp.int32(0), unroll=False)

    # Drain outstanding writebacks.
    last_y_slot = (num_experts - 1) % 2
    wait_writeback(last_y_slot)
    if num_experts >= 2:
        wait_writeback(1 - last_y_slot)


def _multi_expert_kernel_fp8(
    tokens_hbm, w1_hbm, w2_hbm, w3_hbm,
    w1_scale_hbm, w2_scale_hbm, w3_scale_hbm,
    output_hbm,
    b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
    b_w1_scale_x2_vmem, b_w3_scale_x2_vmem, b_w2_scale_x2_vmem,
    b_x_x2_vmem, b_y_acc_vmem, b_y_out_vmem,
    weight_sems, x_sem, y_out_sem,
    *, act_fn, bf, intermediate_size, num_experts, quant_block_k, bd1c,
):
    _multi_expert_kernel(
        tokens_hbm, w1_hbm, w2_hbm, w3_hbm,
        w1_scale_hbm, w2_scale_hbm, w3_scale_hbm,
        output_hbm,
        b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
        b_w1_scale_x2_vmem, b_w3_scale_x2_vmem, b_w2_scale_x2_vmem,
        b_x_x2_vmem, b_y_acc_vmem, b_y_out_vmem,
        weight_sems, x_sem, y_out_sem,
        act_fn=act_fn, bf=bf, intermediate_size=intermediate_size,
        num_experts=num_experts, quant_block_k=quant_block_k, bd1c=bd1c,
    )


def _multi_expert_kernel_bf16(
    tokens_hbm, w1_hbm, w2_hbm, w3_hbm,
    output_hbm,
    b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
    b_x_x2_vmem, b_y_acc_vmem, b_y_out_vmem,
    weight_sems, x_sem, y_out_sem,
    *, act_fn, bf, intermediate_size, num_experts, quant_block_k, bd1c,
):
    _multi_expert_kernel(
        tokens_hbm, w1_hbm, w2_hbm, w3_hbm,
        None, None, None,
        output_hbm,
        b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem,
        None, None, None,
        b_x_x2_vmem, b_y_acc_vmem, b_y_out_vmem,
        weight_sems, x_sem, y_out_sem,
        act_fn=act_fn, bf=bf, intermediate_size=intermediate_size,
        num_experts=num_experts, quant_block_k=quant_block_k, bd1c=bd1c,
    )


@functools.partial(
    jax.jit,
    static_argnames=["act_fn", "bf", "num_experts", "quant_block_k", "bd1c"],
)
def multi_expert_ffn(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    quant_block_k: int | None = None,
    bd1c: int | None = None,
) -> jax.Array:
    """Run the multi-expert double-buffer FFN kernel.

    Shapes (bf16):
      tokens: (num_experts, bt, d) bf16
      w1:     (num_experts, d, f)  bf16
      w2:     (num_experts, f, d)  bf16
      w3:     (num_experts, d, f)  bf16

    Shapes (fp8 with scales):
      tokens:   (num_experts, bt, d) bf16
      w1:       (num_experts, d, f)  fp8_e4m3
      w2:       (num_experts, f, d)  fp8_e4m3
      w3:       (num_experts, d, f)  fp8_e4m3
      w1_scale: (num_experts, d//qbk, 1, f) f32
      w2_scale: (num_experts, f//qbk, 1, d) f32
      w3_scale: (num_experts, d//qbk, 1, f) f32

    Returns:
      output: (num_experts, bt, d) bf16
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

    use_fp8 = w1_scale is not None
    if use_fp8:
        if w2_scale is None or w3_scale is None:
            raise ValueError("All three scales must be provided for fp8")
        if quant_block_k is None:
            raise ValueError("quant_block_k required when scales are provided")
        if d % quant_block_k != 0:
            raise ValueError(
                f"hidden_size={d} must be divisible by quant_block_k={quant_block_k}"
            )
        if f_full % quant_block_k != 0:
            raise ValueError(
                f"intermediate_size={f_full} must be divisible by "
                f"quant_block_k={quant_block_k}"
            )
        if bf % quant_block_k != 0:
            raise ValueError(
                f"bf={bf} must be divisible by quant_block_k={quant_block_k}"
            )
        if bd1c is None:
            bd1c = quant_block_k
        if bd1c % quant_block_k != 0:
            raise ValueError(
                f"bd1c={bd1c} must be divisible by quant_block_k={quant_block_k}"
            )
        tp = T_PACKING
        if d % tp != 0:
            raise ValueError(f"hidden_size={d} must be divisible by t_packing={tp}")
        if bf % tp != 0:
            raise ValueError(f"bf={bf} must be divisible by t_packing={tp}")
        if bd1c % tp != 0:
            raise ValueError(f"bd1c={bd1c} must be divisible by t_packing={tp}")

        d_per_tp = d // tp
        f_per_tp = f_full // tp
        n_sg_per_tp = d_per_tp // quant_block_k
        bf_per_tp = bf // tp
        n_sg2_per_tp = bf_per_tp // quant_block_k  # per-tile scale groups for w2

        # Reshape weights and scales to t_packing layout
        w1 = w1.reshape(n_exp, tp, d_per_tp, f_full)
        w3 = w3.reshape(n_exp, tp, d_per_tp, f_full)
        w2 = w2.reshape(n_exp, tp, f_per_tp, d)
        w1_scale = w1_scale.reshape(n_exp, tp, n_sg_per_tp, 1, f_full)
        w3_scale = w3_scale.reshape(n_exp, tp, n_sg_per_tp, 1, f_full)
        w2_scale = w2_scale.reshape(n_exp, tp, f_per_tp // quant_block_k, 1, d)

        # Pack tokens: bf16 (E, bt, d) → uint32 (E, bt, d//tp)
        # Weight layout is contiguous-split: w1[e, p, k, :] corresponds to
        # original rows [p*d_per_tp .. (p+1)*d_per_tp-1].
        # Token packing must match: uint32[k] packs token[0*d_per_tp+k] (low)
        # and token[1*d_per_tp+k] (high), i.e. contiguous halves.
        # Reshape to (E, bt, tp, d_per_tp) then transpose to put tp last for
        # bitcast_convert_type.
        tokens = tokens.reshape(n_exp, bt, tp, d_per_tp)
        tokens = tokens.transpose(0, 1, 3, 2)  # (E, bt, d_per_tp, tp)
        tokens_packed = jax.lax.bitcast_convert_type(
            tokens, jnp.uint32
        )  # (n_exp, bt, d_per_tp) uint32

    else:
        if bd1c is None:
            bd1c = 0  # unused for bf16

    dtype = tokens.dtype if not use_fp8 else jnp.uint32
    weight_dtype = w1.dtype

    scratch_shapes = [
        pltpu.VMEM((2, tp, d_per_tp, bf), weight_dtype)
            if use_fp8 else pltpu.VMEM((2, d, bf), weight_dtype),
        pltpu.VMEM((2, tp, d_per_tp, bf), weight_dtype)
            if use_fp8 else pltpu.VMEM((2, d, bf), weight_dtype),
        pltpu.VMEM((2, tp, bf_per_tp, d), weight_dtype)
            if use_fp8 else pltpu.VMEM((2, bf, d), weight_dtype),
    ]
    if use_fp8:
        scratch_shapes.extend([
            pltpu.VMEM((2, tp, n_sg_per_tp, 1, bf), jnp.float32),
            pltpu.VMEM((2, tp, n_sg_per_tp, 1, bf), jnp.float32),
            pltpu.VMEM((2, tp, n_sg2_per_tp, 1, d), jnp.float32),
        ])
    scratch_shapes.extend([
        pltpu.VMEM((2, bt, d_per_tp), jnp.uint32)
            if use_fp8 else pltpu.VMEM((2, bt, d), tokens.dtype),
        pltpu.VMEM((bt, d), jnp.float32),
        pltpu.VMEM((2, bt, d), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2, 3)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ])

    scope_name = f"multi-expert-bt{bt}-bf{bf}-e{num_experts}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = [hbm] * (7 if use_fp8 else 4)

    kernel = pl.pallas_call(
        functools.partial(
            _multi_expert_kernel_fp8 if use_fp8 else _multi_expert_kernel_bf16,
            act_fn=act_fn,
            bf=bf,
            intermediate_size=f_full,
            num_experts=num_experts,
            quant_block_k=quant_block_k,
            bd1c=bd1c,
        ),
        out_shape=jax.ShapeDtypeStruct((num_experts, bt, d), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=scope_name,
    )

    if use_fp8:
        hbm_inputs = [
            pltpu.with_memory_space_constraint(tokens_packed, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1_scale, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2_scale, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3_scale, pltpu.HBM),
        ]
    else:
        hbm_inputs = [
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
        ]
    return jax.named_scope(scope_name)(kernel)(*hbm_inputs)


def _ref_multi_expert_ffn(
    tokens, w1, w2, w3, *,
    act_fn="silu",
    w1_scale=None, w2_scale=None, w3_scale=None,
    quant_block_k=None,
):
    """Pure-JAX reference: loop over experts, each does (silu(x@W1) * (x@W3)) @ W2."""
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        if w1_scale is not None:
            d = x.shape[-1]
            f = w1.shape[-1]
            n_sg = d // quant_block_k
            gate = jnp.zeros((x.shape[0], f), dtype=jnp.float32)
            up = jnp.zeros_like(gate)
            for sg in range(n_sg):
                sg_off = sg * quant_block_k
                x_slice = x[:, sg_off:sg_off + quant_block_k]
                w1_tile = w1[e, sg_off:sg_off + quant_block_k, :].astype(
                    jnp.float32
                )
                w3_tile = w3[e, sg_off:sg_off + quant_block_k, :].astype(
                    jnp.float32
                )
                s1 = w1_scale[e, sg:sg + 1, 0, :].reshape(1, f)
                s3 = w3_scale[e, sg:sg + 1, 0, :].reshape(1, f)
                gate += (x_slice @ w1_tile) * s1
                up += (x_slice @ w3_tile) * s3
            # Global SiLU activation on full gate/up vectors
            act = activation_fn(gate, up, act_fn)
            # FFN2
            n_sg2 = f // quant_block_k
            out = jnp.zeros((x.shape[0], d), dtype=jnp.float32)
            for sg in range(n_sg2):
                sg_off = sg * quant_block_k
                act_slice = act[:, sg_off:sg_off + quant_block_k]
                w2_tile = w2[e, sg_off:sg_off + quant_block_k, :].astype(
                    jnp.float32
                )
                s2 = w2_scale[e, sg:sg + 1, 0, :].reshape(1, d)
                out += (act_slice @ w2_tile) * s2
            outputs.append(out.astype(tokens.dtype))
        else:
            gate = x @ w1[e].astype(jnp.float32)
            up = x @ w3[e].astype(jnp.float32)
            act = activation_fn(gate, up, act_fn)
            out = (act @ w2[e].astype(jnp.float32)).astype(tokens.dtype)
            outputs.append(out)
    return jnp.stack(outputs)


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int = 8,
    quant_block_k: int = 128,
    bd1c: int | None = None,
    **_kwargs,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(
        k1, (num_experts, num_tokens, hidden_size), dtype=dtype,
    )
    w1 = jax.random.normal(
        k2, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16,
    ).astype(weight_dtype)
    w2 = jax.random.normal(
        k3, (num_experts, intermediate_size, hidden_size), dtype=jnp.bfloat16,
    ).astype(weight_dtype)
    w3 = jax.random.normal(
        k4, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16,
    ).astype(weight_dtype)

    w1_scale = w2_scale = w3_scale = None
    qbk = None
    bd1c_val = None
    if jnp.dtype(weight_dtype) == jnp.float8_e4m3fn:
        n_sg_h = hidden_size // quant_block_k
        n_sg_i = intermediate_size // quant_block_k
        w1_scale = (
            jnp.ones(
                (num_experts, n_sg_h, 1, intermediate_size), dtype=jnp.float32,
            ) * 0.01
        )
        w2_scale = (
            jnp.ones(
                (num_experts, n_sg_i, 1, hidden_size), dtype=jnp.float32,
            ) * 0.01
        )
        w3_scale = (
            jnp.ones(
                (num_experts, n_sg_h, 1, intermediate_size), dtype=jnp.float32,
            ) * 0.01
        )
        qbk = quant_block_k
        bd1c_val = bd1c if bd1c is not None else quant_block_k

    def run():
        return multi_expert_ffn(
            tokens, w1, w2, w3,
            act_fn=act_fn, bf=bf, num_experts=num_experts,
            w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
            quant_block_k=qbk, bd1c=bd1c_val,
        )

    return run


if __name__ == "__main__":
    num_experts_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bt = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    use_fp8 = "--fp8" in sys.argv
    bf_arg = 512 if use_fp8 else 256

    # Parse --bd1c
    bd1c_arg = None
    for i, a in enumerate(sys.argv):
        if a == "--bd1c" and i + 1 < len(sys.argv):
            bd1c_arg = int(sys.argv[i + 1])

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d, f = (6144, 2048) if use_fp8 else (8192, 2048)
    w_dtype = jnp.float8_e4m3fn if use_fp8 else jnp.bfloat16
    qbk = 128 if use_fp8 else None

    tokens = jax.random.normal(
        k1, (num_experts_arg, bt, d), dtype=jnp.bfloat16,
    )
    w1 = jax.random.normal(
        k2, (num_experts_arg, d, f), dtype=jnp.bfloat16,
    ).astype(w_dtype)
    w2 = jax.random.normal(
        k3, (num_experts_arg, f, d), dtype=jnp.bfloat16,
    ).astype(w_dtype)
    w3 = jax.random.normal(
        k4, (num_experts_arg, d, f), dtype=jnp.bfloat16,
    ).astype(w_dtype)

    w1_scale = w2_scale = w3_scale = None
    if use_fp8:
        n_sg_h = d // qbk
        n_sg_i = f // qbk
        w1_scale = (
            jnp.ones((num_experts_arg, n_sg_h, 1, f), dtype=jnp.float32)
            * 0.01
        )
        w2_scale = (
            jnp.ones((num_experts_arg, n_sg_i, 1, d), dtype=jnp.float32)
            * 0.01
        )
        w3_scale = (
            jnp.ones((num_experts_arg, n_sg_h, 1, f), dtype=jnp.float32)
            * 0.01
        )

    result = multi_expert_ffn(
        tokens, w1, w2, w3, bf=bf_arg, num_experts=num_experts_arg,
        w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
        quant_block_k=qbk, bd1c=bd1c_arg,
    )
    ref = _ref_multi_expert_ffn(
        tokens, w1, w2, w3,
        w1_scale=w1_scale, w2_scale=w2_scale, w3_scale=w3_scale,
        quant_block_k=qbk,
    )

    max_errs = []
    rel_errs = []
    for e in range(num_experts_arg):
        r = result[e].astype(jnp.float32)
        f_ref = ref[e].astype(jnp.float32)
        me = jnp.max(jnp.abs(r - f_ref))
        re = me / (jnp.max(jnp.abs(f_ref)) + 1e-6)
        max_errs.append(float(me))
        rel_errs.append(float(re))
        print(f"  expert {e}: max_abs_err={me:.4f}, rel_err={re:.4f}")

    worst_rel = max(rel_errs)
    mode = "fp8" if use_fp8 else "bf16"
    bd1c_str = f", bd1c={bd1c_arg}" if bd1c_arg else ""
    print(
        f"[{mode}] num_experts={num_experts_arg}, bt={bt}, bf={bf_arg}{bd1c_str}, "
        f"worst_rel_err={worst_rel:.4f}"
    )
    assert worst_rel < 0.05, f"worst rel_err too high: {worst_rel}"
    print("PASS")
