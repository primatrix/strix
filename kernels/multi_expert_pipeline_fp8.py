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
