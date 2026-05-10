"""Double-buffer VMEM load benchmark - Version 3: Dynamic Address + JIT

This version calculates DMA addresses dynamically:
- DMA descriptors created each iteration with computed addresses
- Addresses calculated as (bf_id, bd_id) based on load_idx
- Static loop unrolling (NUM_LOADS=128)
- JIT compilation enabled
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

try:
    from ._fused_moe_impl import cdiv
except ImportError:
    from _fused_moe_impl import cdiv

config = {
    "default_shape": {
        "hidden_size": 8192,
        "intermediate_size": 2048,
        "num_loads": 128,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "tpu_type": "v7x",
    "tpu_topology": "1x1",
    "description": "V3: Dynamic address calculation + JIT",
}


def _dma_double_buffer_load_kernel(
    w_hbm,
    output_vmem,
    b_w_x2_vmem,
    weight_sems,
    *,
    bf: int,
    bd: int,
):
    NUM_LOADS = 128

    hidden_size = w_hbm.shape[0]
    intermediate_size = w_hbm.shape[1]

    num_bf = cdiv(intermediate_size, bf)
    num_bd = cdiv(hidden_size, bd)

    # Helper to create DMA descriptor with dynamic address
    def make_w_copy(buf, bf_id, bd_id):
        return pltpu.make_async_copy(
            src_ref=w_hbm.at[pl.ds(bd_id * bd, bd), pl.ds(bf_id * bf, bf)],
            dst_ref=b_w_x2_vmem.at[buf],
            sem=weight_sems.at[buf],
        )

    checksum = jnp.float32(0.0)

    # Prefetch tile 0 → buf 0
    copy_0 = make_w_copy(0, 0, 0)
    copy_0.start()

    # Prefetch tile 1 → buf 1
    copy_1 = make_w_copy(1, 1 % num_bf, (1 // num_bf) % num_bd)
    copy_1.start()

    # Static loop with dynamic address calculation
    for load_idx in range(NUM_LOADS):
        buf = load_idx % 2

        # Wait for current buffer
        if buf == 0:
            copy_0.wait()
        else:
            copy_1.wait()

        # Start next DMA with computed address
        if load_idx + 2 < NUM_LOADS:
            next_tile = load_idx + 2
            next_bf_id = next_tile % num_bf
            next_bd_id = (next_tile // num_bf) % num_bd
            next_copy = make_w_copy(buf, next_bf_id, next_bd_id)
            next_copy.start()

            # Update copy reference for next iteration
            if buf == 0:
                copy_0 = next_copy
            else:
                copy_1 = next_copy

        checksum = checksum + jnp.float32(1.0)

    output_vmem[...] = jnp.expand_dims(checksum, 0)


@functools.partial(jax.jit, static_argnames=("bf", "bd"))
def dma_double_buffer_load(
    w: jax.Array,
    *,
    bf: int = 2048,
    bd: int = 1024,
) -> jax.Array:
    hidden_size, intermediate_size = w.shape
    w_dtype = w.dtype

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(1,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec((1,), lambda i: (0,)),
        scratch_shapes=[
            pltpu.VMEM((2, bd, bf), w_dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )

    result = pl.pallas_call(
        functools.partial(
            _dma_double_buffer_load_kernel,
            bf=bf,
            bd=bd,
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
        ),
    )(w)

    return result[0]


def kernel_fn(
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    num_loads: int = 128,
    dtype: jnp.dtype = jnp.bfloat16,
    weight_dtype: jnp.dtype = jnp.bfloat16,
    bf: int = 2048,
    bd: int = 1024,
    **_kwargs,
):
    key = jax.random.key(42)
    w = jax.random.normal(key, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return dma_double_buffer_load(w, bf=bf, bd=bd)

    return run
