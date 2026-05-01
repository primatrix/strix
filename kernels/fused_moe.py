"""Fused Expert-Parallel MoE kernel — ported from primatrix/sglang-jax feat/hybrid-moe-prefill."""

import jax
import jax.numpy as jnp
import numpy as np

from ._fused_moe_impl import (
    FusedMoEBlockConfig,
    fused_ep_moe,
)

config = {
    "default_shape": {
        "num_tokens": 1024,
        "num_experts": 128,
        "top_k": 8,
        "hidden_size": 4096,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "ep_size": 8,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": "Fused EP MoE kernel with batch scatter optimization (Qwen3-MoE-128E config)",
}


def kernel_fn(
    num_tokens=1024,
    num_experts=128,
    top_k=8,
    hidden_size=4096,
    intermediate_size=2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn="silu",
    ep_size=8,
    block_config=None,
):
    """Construct inputs and call fused_ep_moe, returning a JAX-compilable closure."""
    available_devices = jax.devices()
    if len(available_devices) < ep_size:
        raise ValueError(
            f"Requested ep_size={ep_size} but only {len(available_devices)} devices are available."
        )
    devices = available_devices[:ep_size]
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(1, -1),
        axis_names=("data", "tensor"),
    )

    key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (num_experts, hidden_size, intermediate_size), dtype=weight_dtype)
    w2 = jax.random.normal(k3, (num_experts, intermediate_size, hidden_size), dtype=weight_dtype)
    w3 = jax.random.normal(k4, (num_experts, hidden_size, intermediate_size), dtype=weight_dtype)

    topk_weights = jnp.ones((num_tokens, top_k), dtype=jnp.float32) / top_k
    topk_ids = jax.random.randint(k5, (num_tokens, top_k), 0, num_experts)

    def run():
        return fused_ep_moe(
            mesh=mesh,
            tokens=tokens,
            w1=w1,
            w2=w2,
            w3=w3,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            act_fn=act_fn,
            block_config=block_config,
            tp_axis_name="tensor",
        )

    return run
