"""Fused Expert-Parallel MoE v2 kernel — Strix-style double-buffer pipeline.

Ported from sglang-jax fused_moe/v2. Default config targets MiMo V2 Pro
(E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32).
"""

import jax
import jax.numpy as jnp
import numpy as np

from ._fused_moe_v2_impl import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
)
from ._fused_moe_v2_configs import get_tuned_fused_moe_v2_block_config

config = {
    "default_shape": {
        "num_tokens": 256,
        "num_experts": 384,
        "top_k": 8,
        "hidden_size": 6144,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "float8_e4m3fn",
    "act_fn": "silu",
    "ep_size": 32,
    "quant_block_k": 128,
    "direct_scaled_dot": True,
    "tpu_type": "v7x",
    "tpu_topology": "4x8x1",
    "description": "Fused EP MoE v2 kernel — MiMo V2 Pro decode (384e, top_k=8, H=6144, I=2048, fp8 direct_scaled_dot)",
}


def kernel_fn(
    num_tokens=256,
    num_experts=384,
    top_k=8,
    hidden_size=6144,
    intermediate_size=2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.float8_e4m3fn,
    act_fn="silu",
    ep_size=32,
    quant_block_k=128,
    direct_scaled_dot=True,
    decode_mode=False,
    block_config=None,
):
    """Construct inputs and call fused_ep_moe_v2, returning a zero-arg closure."""
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
    ep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "tensor")))

    key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    def make_sharded(rng_key, shape, dt, scale=1.0):
        num_devices_local = len(jax.local_devices())
        local_shape = (shape[0] // ep_size * num_devices_local, *shape[1:])
        per_device = []
        for i, dev in enumerate(jax.local_devices()):
            shard_key = jax.random.fold_in(rng_key, jax.process_index() * num_devices_local + i)
            shard = jax.device_put(
                jax.random.normal(shard_key, (shape[0] // ep_size, *shape[1:]), dtype=jnp.bfloat16).astype(dt) * scale,
                dev,
            )
            per_device.append(shard)
        return jax.make_array_from_single_device_arrays(shape, ep_sharding, per_device)

    tokens = make_sharded(k1, (num_tokens, hidden_size), dtype)
    w1 = make_sharded(k2, (num_experts, hidden_size, intermediate_size), weight_dtype, 0.01)
    w2 = make_sharded(k3, (num_experts, intermediate_size, hidden_size), weight_dtype, 0.01)
    w3 = make_sharded(k4, (num_experts, hidden_size, intermediate_size), weight_dtype, 0.01)

    w1_scale = w2_scale = w3_scale = None
    qbk_arg = None
    if jnp.dtype(weight_dtype) == jnp.float8_e4m3fn:
        n_scale_groups_h = hidden_size // quant_block_k
        n_scale_groups_i = intermediate_size // quant_block_k
        w1_scale = make_sharded(k2, (num_experts, n_scale_groups_h, 1, intermediate_size), jnp.float32, 0.001)
        w2_scale = make_sharded(k3, (num_experts, n_scale_groups_i, 1, hidden_size), jnp.float32, 0.001)
        w3_scale = make_sharded(k4, (num_experts, n_scale_groups_h, 1, intermediate_size), jnp.float32, 0.001)
        qbk_arg = quant_block_k

    topk_weights = jnp.ones((num_tokens, top_k), dtype=jnp.float32) / top_k
    topk_ids = jax.random.randint(k5, (num_tokens, top_k), 0, num_experts)
    topk_weights = jax.device_put(topk_weights, ep_sharding)
    topk_ids = jax.device_put(topk_ids, ep_sharding)

    if block_config is None:
        block_config = get_tuned_fused_moe_v2_block_config(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            ep_size=ep_size,
        )

    def run():
        return fused_ep_moe_v2(
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
            quant_block_k=qbk_arg,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            decode_mode=decode_mode,
            direct_scaled_dot=direct_scaled_dot,
            tp_axis_name="tensor",
        )

    return run
