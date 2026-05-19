"""Multi-expert double-buffer FFN pipeline — FP8 e4m3 block quantization.

VMEM dequant path: load fp8 weights from HBM → dequant to bf16 in VMEM → bf16 matmul.
Per-block f32 scales with quant_block_k granularity.

Base kernel: kernels/multi_expert_pipeline.py (bf16 version)
Spec: docs/superpowers/specs/2026-05-19-multi-expert-pipeline-fp8-design.md
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
        "hidden_size": 6144,
        "intermediate_size": 2048,
        "num_experts": 8,
    },
    "dtype": "bfloat16",
    "weight_dtype": "float8_e4m3fn",
    "act_fn": "silu",
    "bf": 256,
    "quant_block_k": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Multi-expert double-buffer FFN — FP8 e4m3 block quantization "
        "(VMEM dequant path, N experts sequential)"
    ),
}


def _dequant_weight(w_fp8, scale, quant_block_k):
    """Dequant fp8 weight using per-block scale.

    Args:
        w_fp8: (K, N) float8_e4m3fn
        scale: (K // quant_block_k, 1, N) float32
        quant_block_k: block size along contracting dim

    Returns:
        (K, N) float32
    """
    w_f32 = w_fp8.astype(jnp.float32)
    s = jnp.repeat(scale.squeeze(1), quant_block_k, axis=0)
    return w_f32 * s


def _ref_multi_expert_ffn_fp8(tokens, w1, w2, w3, *,
                               w1_scale, w2_scale, w3_scale,
                               quant_block_k, act_fn="silu"):
    """Pure-JAX reference: dequant fp8 → f32, matmul, activation, matmul."""
    num_experts = tokens.shape[0]
    outputs = []
    for e in range(num_experts):
        x = tokens[e].astype(jnp.float32)
        w1_f32 = _dequant_weight(w1[e], w1_scale[e], quant_block_k)
        w3_f32 = _dequant_weight(w3[e], w3_scale[e], quant_block_k)
        w2_f32 = _dequant_weight(w2[e], w2_scale[e], quant_block_k)
        gate = x @ w1_f32
        up = x @ w3_f32
        act = activation_fn(gate, up, act_fn)
        out = (act @ w2_f32).astype(tokens.dtype)
        outputs.append(out)
    return jnp.stack(outputs)


def _multi_expert_kernel_fp8(
    tokens_hbm,       # (num_experts, bt, d) bf16
    w1_hbm,           # (num_experts, d, f) fp8
    w2_hbm,           # (num_experts, f, d) fp8
    w3_hbm,           # (num_experts, d, f) fp8
    w1_scale_hbm,     # (num_experts, d // qbk, 1, f) f32
    w2_scale_hbm,     # (num_experts, f // qbk, 1, d) f32
    w3_scale_hbm,     # (num_experts, d // qbk, 1, f) f32
    output_hbm,       # (num_experts, bt, d) bf16
    # --- VMEM scratch ---
    b_w1_x2_vmem,         # (2, d, bf) fp8
    b_w3_x2_vmem,         # (2, d, bf) fp8
    b_w2_x2_vmem,         # (2, bf, d) fp8
    b_w1_scale_vmem,      # (d // qbk, 1, f) f32
    b_w3_scale_vmem,      # (d // qbk, 1, f) f32
    b_w2_scale_vmem,      # (f // qbk, 1, d) f32
    b_x_x2_vmem,          # (2, bt, d) bf16
    b_y_acc_vmem,          # (bt, d) f32
    b_y_out_vmem,          # (2, bt, d) bf16
    # --- Semaphores ---
    weight_sems,           # DMA(2, 3)
    x_sem,                 # DMA(2,)
    y_out_sem,             # DMA(2,)
    scale_sems,            # DMA(3,)
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
    num_experts: int,
    quant_block_k: int,
    skip_dequant: bool = False,
):
    """Pallas kernel body — multi-expert FP8 pipeline with VMEM dequant."""
    _, bt, d = tokens_hbm.shape
    n_w = intermediate_size // bf
    n_sg = d // quant_block_k
    n_sg2 = bf // quant_block_k

    # -- Token DMA --

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

    # -- Scale DMA (full expert, loaded once per expert) --

    def start_load_scales(expert_idx):
        pltpu.make_async_copy(
            src_ref=w1_scale_hbm.at[expert_idx],
            dst_ref=b_w1_scale_vmem,
            sem=scale_sems.at[0],
        ).start(priority=1)
        pltpu.make_async_copy(
            src_ref=w3_scale_hbm.at[expert_idx],
            dst_ref=b_w3_scale_vmem,
            sem=scale_sems.at[1],
        ).start(priority=1)
        pltpu.make_async_copy(
            src_ref=w2_scale_hbm.at[expert_idx],
            dst_ref=b_w2_scale_vmem,
            sem=scale_sems.at[2],
        ).start(priority=0)

    def wait_load_scales():
        pltpu.make_async_copy(
            src_ref=b_w1_scale_vmem,
            dst_ref=b_w1_scale_vmem,
            sem=scale_sems.at[0],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_w3_scale_vmem,
            dst_ref=b_w3_scale_vmem,
            sem=scale_sems.at[1],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_w2_scale_vmem,
            dst_ref=b_w2_scale_vmem,
            sem=scale_sems.at[2],
        ).wait()

    # -- Weight DMA (tile-granularity, double-buffered) --

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

    # -- Output writeback --

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

    # -- Sub-tile dequant-dot fusion --
    # Instead of dequanting full (d, bf) into a VMEM scratch buffer,
    # dequant in (qbk, bf) sub-blocks and accumulate partial matmuls.
    # Eliminates 3 × (d, bf) bf16 scratch buffers, enabling larger bf.

    def dequant_dot_w13(x, w_slot, tile_idx):
        """Fused sub-tile dequant + gate/up matmuls for w1 and w3."""
        gate = jnp.zeros((bt, bf), jnp.float32)
        up = jnp.zeros((bt, bf), jnp.float32)
        w1_tile = b_w1_x2_vmem[w_slot]
        w3_tile = b_w3_x2_vmem[w_slot]
        for sg in range(n_sg):
            off = sg * quant_block_k
            w1_blk = w1_tile[off:off + quant_block_k, :]
            w3_blk = w3_tile[off:off + quant_block_k, :]
            x_blk = x[:, off:off + quant_block_k]
            if skip_dequant:
                w1_dq = w1_blk.astype(jnp.bfloat16)
                w3_dq = w3_blk.astype(jnp.bfloat16)
            else:
                s1 = b_w1_scale_vmem[sg, :, pl.ds(tile_idx * bf, bf)]
                s3 = b_w3_scale_vmem[sg, :, pl.ds(tile_idx * bf, bf)]
                w1_dq = (w1_blk.astype(jnp.float32) * s1).astype(jnp.bfloat16)
                w3_dq = (w3_blk.astype(jnp.float32) * s3).astype(jnp.bfloat16)
            gate += jnp.dot(x_blk, w1_dq, preferred_element_type=jnp.float32)
            up += jnp.dot(x_blk, w3_dq, preferred_element_type=jnp.float32)
        return gate, up

    def dequant_dot_w2(act, w_slot, tile_idx):
        """Fused sub-tile dequant + down-projection matmul for w2."""
        out = jnp.zeros((bt, d), jnp.float32)
        w2_tile = b_w2_x2_vmem[w_slot]
        for sg in range(n_sg2):
            off = sg * quant_block_k
            w2_blk = w2_tile[off:off + quant_block_k, :]
            act_blk = act[:, off:off + quant_block_k]
            if skip_dequant:
                w2_dq = w2_blk.astype(jnp.bfloat16)
            else:
                s2 = b_w2_scale_vmem[tile_idx * n_sg2 + sg, :, :]
                w2_dq = (w2_blk.astype(jnp.float32) * s2).astype(jnp.bfloat16)
            out += jnp.dot(act_blk, w2_dq, preferred_element_type=jnp.float32)
        return out

    def start_fetch_w13(slot, expert_idx, tile_idx):
        start_fetch_w1(slot, expert_idx, tile_idx)
        start_fetch_w3(slot, expert_idx, tile_idx)

    # -- Global Prologue --

    x_slot = 0
    start_load_x(x_slot=0, expert_idx=0)
    start_load_scales(0)
    start_fetch_w1(0, 0, 0, priority=1)
    start_fetch_w3(0, 0, 0, priority=1)
    start_fetch_w2(0, 0, 0)
    start_fetch_w1(1, 0, 1)
    start_fetch_w3(1, 0, 1)
    start_fetch_w2(1, 0, 1)

    wait_load_x(x_slot)
    wait_load_scales()

    # -- Expert Loop --

    def expert_body(e, x_slot):
        next_xs = 1 - x_slot

        def tile_body(tile, _):
            w_slot = tile % 2
            next_w = 1 - w_slot
            is_last = tile + 1 >= n_w
            is_penultimate = tile + 2 == n_w

            not_last = ~is_last
            pf_slot = (tile + not_last) % 2
            pf_expert = e + is_last
            pf_tile = is_last * w_slot + not_last * (tile + 1)
            should_prefetch = (tile > 0) & (pf_expert < num_experts)

            wait_fetch_w1(w_slot)
            wait_fetch_w3(w_slot)

            x = b_x_x2_vmem[x_slot]

            gate, up = dequant_dot_w13(x, w_slot, tile)

            @pl.when(should_prefetch)
            def _():
                start_fetch_w13(pf_slot, pf_expert, pf_tile)

            @pl.when(is_penultimate & (e < num_experts - 1))
            def _():
                start_load_x(next_xs, e + 1, priority=1)
                start_fetch_w13(w_slot, e + 1, w_slot)

            wait_fetch_w2(w_slot)
            act = activation_fn(gate, up, act_fn)
            partial = dequant_dot_w2(act, w_slot, tile)

            @pl.when(should_prefetch)
            def _():
                start_fetch_w2(pf_slot, pf_expert, pf_tile)

            @pl.when(is_penultimate & (e < num_experts - 1))
            def _():
                start_fetch_w2(w_slot, e + 1, w_slot)

            @pl.when(tile == 0)
            def _():
                b_y_acc_vmem[...] = partial

            @pl.when(tile > 0)
            def _():
                b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial

            return jnp.int32(0)

        lax.fori_loop(0, n_w, tile_body, jnp.int32(0), unroll=True)

        # Start next-expert scale DMA early to overlap with writeback
        @pl.when(e < num_experts - 1)
        def _():
            start_load_scales(e + 1)

        # --- Expert boundary: double-buffered writeback ---
        @pl.when(e >= 2)
        def _():
            wait_writeback(x_slot)

        b_y_out_vmem[x_slot] = b_y_acc_vmem[...].astype(jnp.bfloat16)
        start_writeback(e, x_slot)

        @pl.when(e < num_experts - 1)
        def _():
            wait_load_x(next_xs)
            wait_load_scales()

        return next_xs

    lax.fori_loop(0, num_experts, expert_body, jnp.int32(0), unroll=False)

    # -- Drain outstanding writebacks --
    last_y_slot = (num_experts - 1) % 2
    wait_writeback(last_y_slot)
    if num_experts >= 2:
        wait_writeback(1 - last_y_slot)


@functools.partial(jax.jit, static_argnames=[
    "act_fn", "bf", "num_experts", "quant_block_k", "skip_dequant",
])
def multi_expert_ffn_fp8(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    w3_scale: jax.Array,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int,
    quant_block_k: int = 256,
    skip_dequant: bool = False,
) -> jax.Array:
    """Run the multi-expert FP8 double-buffer FFN kernel (VMEM dequant path).

    Shapes:
      tokens:   (num_experts, bt, d)                 bf16
      w1:       (num_experts, d, f)                   float8_e4m3fn
      w2:       (num_experts, f, d)                   float8_e4m3fn
      w3:       (num_experts, d, f)                   float8_e4m3fn
      w1_scale: (num_experts, d // quant_block_k, 1, f) f32
      w2_scale: (num_experts, f // quant_block_k, 1, d) f32
      w3_scale: (num_experts, d // quant_block_k, 1, f) f32
    Returns:
      output:   (num_experts, bt, d)                  bf16
    """
    n_exp, bt, d = tokens.shape
    f_full = w1.shape[2]
    qbk = quant_block_k
    weight_dtype = w1.dtype

    # --- Validation ---
    if n_exp != num_experts:
        raise ValueError(f"tokens.shape[0]={n_exp} != num_experts={num_experts}")
    if f_full % bf != 0:
        raise ValueError(f"intermediate_size={f_full} must be divisible by bf={bf}")
    if f_full // bf < 2:
        raise ValueError(
            f"Need >= 2 weight tiles (intermediate_size/bf >= 2); "
            f"got {f_full}/{bf}={f_full // bf}"
        )
    if qbk % 128 != 0:
        raise ValueError(f"quant_block_k={qbk} must be 128-aligned")
    if d % qbk != 0:
        raise ValueError(f"hidden_size={d} must be divisible by quant_block_k={qbk}")
    if bf % qbk != 0:
        raise ValueError(f"bf={bf} must be divisible by quant_block_k={qbk}")
    if w1.shape != (num_experts, d, f_full):
        raise ValueError(f"w1.shape={w1.shape} must be ({num_experts}, {d}, {f_full})")
    if w2.shape != (num_experts, f_full, d):
        raise ValueError(f"w2.shape={w2.shape} must be ({num_experts}, {f_full}, {d})")
    if w3.shape != (num_experts, d, f_full):
        raise ValueError(f"w3.shape={w3.shape} must be ({num_experts}, {d}, {f_full})")

    n_sg = d // qbk
    n_sg2 = bf // qbk

    expected_w1_scale = (num_experts, n_sg, 1, f_full)
    if w1_scale.shape != expected_w1_scale:
        raise ValueError(f"w1_scale.shape={w1_scale.shape} must be {expected_w1_scale}")
    expected_w2_scale = (num_experts, f_full // qbk, 1, d)
    if w2_scale.shape != expected_w2_scale:
        raise ValueError(f"w2_scale.shape={w2_scale.shape} must be {expected_w2_scale}")
    expected_w3_scale = (num_experts, n_sg, 1, f_full)
    if w3_scale.shape != expected_w3_scale:
        raise ValueError(f"w3_scale.shape={w3_scale.shape} must be {expected_w3_scale}")

    dtype = tokens.dtype

    # --- Scratch shapes ---
    scratch_shapes = (
        pltpu.VMEM((2, d, bf), weight_dtype),              # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),              # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),              # b_w2_x2_vmem
        pltpu.VMEM((n_sg, 1, f_full), jnp.float32),       # b_w1_scale_vmem
        pltpu.VMEM((n_sg, 1, f_full), jnp.float32),       # b_w3_scale_vmem
        pltpu.VMEM((f_full // qbk, 1, d), jnp.float32),   # b_w2_scale_vmem
        pltpu.VMEM((2, bt, d), dtype),                     # b_x_x2_vmem
        pltpu.VMEM((bt, d), jnp.float32),                 # b_y_acc_vmem
        pltpu.VMEM((2, bt, d), dtype),                     # b_y_out_vmem
        pltpu.SemaphoreType.DMA((2, 3)),                   # weight_sems
        pltpu.SemaphoreType.DMA((2,)),                     # x_sem
        pltpu.SemaphoreType.DMA((2,)),                     # y_out_sem
        pltpu.SemaphoreType.DMA((3,)),                     # scale_sems
    )

    scope_name = f"multi-expert-fp8-bt{bt}-bf{bf}-e{num_experts}-qbk{qbk}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = pl.pallas_call(
        functools.partial(
            _multi_expert_kernel_fp8,
            act_fn=act_fn,
            bf=bf,
            intermediate_size=f_full,
            num_experts=num_experts,
            quant_block_k=qbk,
            skip_dequant=skip_dequant,
        ),
        out_shape=jax.ShapeDtypeStruct((num_experts, bt, d), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm, hbm, hbm, hbm, hbm, hbm],
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
        pltpu.with_memory_space_constraint(w1_scale, pltpu.HBM),
        pltpu.with_memory_space_constraint(w2_scale, pltpu.HBM),
        pltpu.with_memory_space_constraint(w3_scale, pltpu.HBM),
    )


def _make_fp8_weights_and_scales(key, shape, quant_block_k):
    """Generate random fp8 weights and matching per-block scales.

    Returns (w_fp8, scale) where:
        w_fp8: shape, float8_e4m3fn
        scale: (shape[0] // quant_block_k, 1, shape[1]), float32
    """
    w_bf16 = jax.random.normal(key, shape, dtype=jnp.bfloat16)
    w_fp8 = w_bf16.astype(jnp.float8_e4m3fn)
    K, N = shape
    n_blocks = K // quant_block_k
    w_fp8_f32 = w_fp8.astype(jnp.float32).reshape(n_blocks, quant_block_k, N)
    block_max = jnp.max(jnp.abs(w_fp8_f32), axis=1, keepdims=True)
    scale = jnp.where(block_max > 0, 1.0 / block_max, 1.0)
    return w_fp8, scale


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 6144,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.float8_e4m3fn,
    act_fn: str = "silu",
    bf: int = 256,
    num_experts: int = 8,
    quant_block_k: int = 256,
    skip_dequant: bool = False,
    **_kwargs,
) -> Callable[[], jax.Array]:
    """Build random FP8 inputs and return a zero-arg closure calling the kernel."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts, num_tokens, hidden_size), dtype=dtype)

    w1_fp8, w1_scale = _make_fp8_weights_and_scales(
        k2, (hidden_size, intermediate_size), quant_block_k)
    w2_fp8, w2_scale = _make_fp8_weights_and_scales(
        k3, (intermediate_size, hidden_size), quant_block_k)
    w3_fp8, w3_scale = _make_fp8_weights_and_scales(
        k4, (hidden_size, intermediate_size), quant_block_k)

    w1 = jnp.broadcast_to(w1_fp8[None], (num_experts,) + w1_fp8.shape).copy()
    w2 = jnp.broadcast_to(w2_fp8[None], (num_experts,) + w2_fp8.shape).copy()
    w3 = jnp.broadcast_to(w3_fp8[None], (num_experts,) + w3_fp8.shape).copy()
    w1s = jnp.broadcast_to(w1_scale[None], (num_experts,) + w1_scale.shape).copy()
    w2s = jnp.broadcast_to(w2_scale[None], (num_experts,) + w2_scale.shape).copy()
    w3s = jnp.broadcast_to(w3_scale[None], (num_experts,) + w3_scale.shape).copy()

    def run():
        return multi_expert_ffn_fp8(
            tokens, w1, w2, w3,
            w1_scale=w1s, w2_scale=w2s, w3_scale=w3s,
            act_fn=act_fn, bf=bf, num_experts=num_experts,
            quant_block_k=quant_block_k,
            skip_dequant=skip_dequant,
        )

    return run


if __name__ == "__main__":
    num_experts_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bt = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    d = int(sys.argv[3]) if len(sys.argv) > 3 else 6144
    f = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    bf_arg = 256
    qbk = 256

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts_arg, bt, d), dtype=jnp.bfloat16)

    w1_fp8, w1_scale = _make_fp8_weights_and_scales(k2, (d, f), qbk)
    w2_fp8, w2_scale = _make_fp8_weights_and_scales(k3, (f, d), qbk)
    w3_fp8, w3_scale = _make_fp8_weights_and_scales(k4, (d, f), qbk)

    w1 = jnp.stack([w1_fp8] * num_experts_arg)
    w2 = jnp.stack([w2_fp8] * num_experts_arg)
    w3 = jnp.stack([w3_fp8] * num_experts_arg)
    w1s = jnp.stack([w1_scale] * num_experts_arg)
    w2s = jnp.stack([w2_scale] * num_experts_arg)
    w3s = jnp.stack([w3_scale] * num_experts_arg)

    result = multi_expert_ffn_fp8(
        tokens, w1, w2, w3,
        w1_scale=w1s, w2_scale=w2s, w3_scale=w3s,
        bf=bf_arg, num_experts=num_experts_arg, quant_block_k=qbk,
    )
    ref = _ref_multi_expert_ffn_fp8(
        tokens, w1, w2, w3,
        w1_scale=w1s, w2_scale=w2s, w3_scale=w3s,
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
        print(f"  expert {e}: max_abs_err={me:.4f}, rel_err={re:.6f}")

    worst_rel = max(rel_errs)
    print(
        f"num_experts={num_experts_arg}, bt={bt}, d={d}, f={f}, "
        f"bf={bf_arg}, qbk={qbk}, worst_rel_err={worst_rel:.6f}"
    )
    assert worst_rel < 0.1, f"worst rel_err too high: {worst_rel}"
    print("PASS")
