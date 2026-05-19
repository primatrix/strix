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
    b_w1_scale_x2_vmem,   # (2, d // qbk, 1, bf) f32
    b_w3_scale_x2_vmem,   # (2, d // qbk, 1, bf) f32
    b_w2_scale_x2_vmem,   # (2, bf // qbk, 1, d) f32
    b_w1_dq_vmem,         # (d, bf) bf16
    b_w3_dq_vmem,         # (d, bf) bf16
    b_w2_dq_vmem,         # (bf, d) bf16
    b_x_x2_vmem,          # (2, bt, d) bf16
    b_y_acc_vmem,          # (bt, d) f32
    b_y_out_vmem,          # (2, bt, d) bf16
    # --- Semaphores ---
    weight_sems,           # DMA(2, 3)
    x_sem,                 # DMA(2,)
    y_out_sem,             # DMA(2,)
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
    num_experts: int,
    quant_block_k: int,
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

    # -- Weight + Scale DMA (co-fetched, shared semaphore) --

    def start_fetch_w1(slot, expert_idx, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start(priority=priority)
        pltpu.make_async_copy(
            src_ref=w1_scale_hbm.at[expert_idx, :, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_scale_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start(priority=priority)

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_w1_scale_x2_vmem.at[slot],
            dst_ref=b_w1_scale_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w3(slot, expert_idx, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[expert_idx, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start(priority=priority)
        pltpu.make_async_copy(
            src_ref=w3_scale_hbm.at[expert_idx, :, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_scale_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start(priority=priority)

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_w3_scale_x2_vmem.at[slot],
            dst_ref=b_w3_scale_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def start_fetch_w2(slot, expert_idx, tile_idx, priority=0):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[expert_idx, pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start(priority=priority)
        pltpu.make_async_copy(
            src_ref=w2_scale_hbm.at[expert_idx, pl.ds(tile_idx * bf // quant_block_k, bf // quant_block_k), :, :],
            dst_ref=b_w2_scale_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start(priority=priority)

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_w2_scale_x2_vmem.at[slot],
            dst_ref=b_w2_scale_x2_vmem.at[slot],
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

    # -- Dequant: fp8 → bf16 in VMEM --
    # Pattern from _fused_moe_v2_impl.py lines 1042-1082.
    # Uses lax.fori_loop with full unroll so Mosaic sees all ops statically.

    def dequant_w1(slot):
        def _dq(sg_id, _):
            sg_off = sg_id * quant_block_k
            w_fp8 = b_w1_x2_vmem[slot, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
            s = b_w1_scale_x2_vmem[slot, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
            s = s.reshape(1, bf)
            w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, bf))).astype(jnp.bfloat16)
            b_w1_dq_vmem.at[pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
            return None
        lax.fori_loop(0, n_sg, _dq, None, unroll=n_sg)

    def dequant_w3(slot):
        def _dq(sg_id, _):
            sg_off = sg_id * quant_block_k
            w_fp8 = b_w3_x2_vmem[slot, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
            s = b_w3_scale_x2_vmem[slot, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
            s = s.reshape(1, bf)
            w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, bf))).astype(jnp.bfloat16)
            b_w3_dq_vmem.at[pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
            return None
        lax.fori_loop(0, n_sg, _dq, None, unroll=n_sg)

    def dequant_w2(slot):
        def _dq(sg_id, _):
            sg_off = sg_id * quant_block_k
            w_fp8 = b_w2_x2_vmem[slot, pl.ds(sg_off, quant_block_k), pl.ds(0, d)]
            s = b_w2_scale_x2_vmem[slot, pl.ds(sg_id, 1), 0, pl.ds(0, d)]
            s = s.reshape(1, d)
            w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, d))).astype(jnp.bfloat16)
            b_w2_dq_vmem.at[pl.ds(sg_off, quant_block_k), pl.ds(0, d)][...] = w_bf16
            return None
        lax.fori_loop(0, n_sg2, _dq, None, unroll=n_sg2)
