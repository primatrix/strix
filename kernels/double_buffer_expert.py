"""Double-buffer expert FFN kernel — §5.8 decode pipeline for Ling 2.6.

Single routed expert, B=1, persistent x, dual weight slots, low-priority
W2 DMA, fp32 y_acc accumulator. Hard-coded for d=8192, f=2048, bf16.

Spec: docs/superpowers/specs/2026-05-11-double-buffer-expert-design.md
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernels._fused_moe_impl import activation_fn


_ALLOWED_CONFIGS = {(256, 512), (512, 256)}  # (num_tokens, bf) pairs from §5.8


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 512,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Double-buffer expert FFN — §5.8 B=1 decode "
        "(persistent x, dual W slots, low-pri W2)"
    ),
}


def _double_buffer_expert_kernel(*args, **kwargs):
    """Kernel body — implemented in Task 3."""
    raise NotImplementedError("Pallas kernel body implemented in Task 3")


def double_buffer_expert(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 512,
) -> jax.Array:
    """Run the double-buffer expert FFN kernel.

    Shapes:
      tokens: (bt, d) bf16
      w1:     (d, f)  bf16
      w2:     (f, d)  bf16
      w3:     (d, f)  bf16
    Returns:
      output: (bt, d) bf16
    """
    bt, d = tokens.shape
    f_full = w1.shape[1]
    if d != 8192 or f_full != 2048:
        raise ValueError(
            f"Kernel hard-coded for d=8192, f=2048; got d={d}, f={f_full}"
        )
    if (bt, bf) not in _ALLOWED_CONFIGS:
        raise ValueError(
            f"Only (num_tokens, bf) in {sorted(_ALLOWED_CONFIGS)} supported; "
            f"got (num_tokens={bt}, bf={bf})"
        )
    if w3.shape != (d, f_full):
        raise ValueError(f"w3.shape={w3.shape} must be (d={d}, f={f_full})")
    if w2.shape != (f_full, d):
        raise ValueError(f"w2.shape={w2.shape} must be (f={f_full}, d={d})")

    # Body wired in Task 3.
    raise NotImplementedError("Pallas kernel wiring implemented in Task 3")


def _ref_expert_ffn(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
) -> jax.Array:
    """Pure-JAX fp32 reference: (silu(x @ W1) * (x @ W3)) @ W2."""
    gate = tokens.astype(jnp.float32) @ w1.astype(jnp.float32)
    up = tokens.astype(jnp.float32) @ w3.astype(jnp.float32)
    act = activation_fn(gate, up, act_fn)
    return (act @ w2.astype(jnp.float32)).astype(tokens.dtype)


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 512,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel.

    Contract matches scripts/benchmark_runner.py (see expert_ffn.py).
    """
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=weight_dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=weight_dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return double_buffer_expert(
            tokens, w1, w2, w3, act_fn=act_fn, bf=bf,
        )

    return run
