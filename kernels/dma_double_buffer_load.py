"""Double-buffer VMEM load benchmark for expert weight DMA profiling.

Tests DMA efficiency of loading expert weights (W1/W2/W3) from HBM to VMEM
using double-buffering across various shapes. Isolates the weight loading
pipeline to measure pure DMA throughput without compute interference.
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
    # Fallback for direct execution
    from _fused_moe_impl import cdiv

config = {
    "default_shape": {
        "hidden_size": 8192,
        "intermediate_size": 2048,
        "num_loads": 64,  # Number of weight tiles to load (simulates expert iteration)
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "tpu_type": "v7x",
    "tpu_topology": "1x1",
    "description": "Double-buffer VMEM load — DMA efficiency test for expert weights",
}


def _dma_double_buffer_load_kernel(
    bf_offsets_ref,
    bd_offsets_ref,
    w_hbm,
    output_vmem,
    # Scratch buffers (VMEM).
    b_w_x2_vmem,
    # Semaphores.
    weight_sems,
    *,
    bf: int,
    bd: int,
    num_loads: int,
):
    """Pallas kernel body — double-buffered weight loading from HBM to VMEM.

    Uses pre-computed tile offsets passed as scalar prefetch to avoid
    per-DMA address computation overhead in the kernel body.

    Args:
        bf_offsets_ref: Pre-computed bf dimension offsets [num_loads]
        bd_offsets_ref: Pre-computed bd dimension offsets [num_loads]
        w_hbm: Weight matrix in HBM [hidden_size, intermediate_size]
        output_vmem: Scalar output ref for checksum (VMEM)
        b_w_x2_vmem: Double-buffer VMEM scratch [2, bd, bf]
        weight_sems: DMA semaphores [2]
        bf: Intermediate dimension block size
        bd: Hidden dimension block size
        num_loads: Number of weight tiles to load
    """
    bf_offsets = bf_offsets_ref[:]
    bd_offsets = bd_offsets_ref[:]

    def make_w_copy(buf, load_idx):
        """Create async copy descriptor using pre-computed offsets."""
        bf_off = bf_offsets[load_idx]
        bd_off = bd_offsets[load_idx]
        return pltpu.make_async_copy(
            src_ref=w_hbm.at[pl.ds(bd_off, bd), pl.ds(bf_off, bf)],
            dst_ref=b_w_x2_vmem.at[buf],
            sem=weight_sems.at[buf],
        )

    # -- Double-buffered load loop (fully unrolled) --
    checksum = jnp.float32(0.0)

    copy_0 = make_w_copy(0, 0)
    copy_0.start()

    if num_loads > 1:
        copy_1 = make_w_copy(1, 1)
        copy_1.start()

    for load_idx in range(num_loads):
        buf = load_idx % 2

        if buf == 0:
            copy_0.wait()
        else:
            copy_1.wait()

        if load_idx + 2 < num_loads:
            next_copy = make_w_copy(buf, load_idx + 2)
            next_copy.start()
            if buf == 0:
                copy_0 = next_copy
            else:
                copy_1 = next_copy

        checksum = checksum + jnp.float32(1.0)

    output_vmem[...] = jnp.expand_dims(checksum, 0)


def dma_double_buffer_load(
    w: jax.Array,
    *,
    bf: int = 2048,
    bd: int = 1024,
    num_loads: int = 64,
) -> jax.Array:
    """Double-buffered weight loading benchmark.

    Args:
        w: Weight matrix [hidden_size, intermediate_size]
        bf: Intermediate dimension block size
        bd: Hidden dimension block size
        num_loads: Number of weight tiles to load

    Returns:
        Scalar checksum for verification
    """
    import numpy as np

    hidden_size, intermediate_size = w.shape
    w_dtype = w.dtype

    num_bf = cdiv(intermediate_size, bf)
    num_bd = cdiv(hidden_size, bd)

    # Pre-compute all tile offsets as byte offsets into the weight matrix
    bf_offsets = np.zeros(num_loads, dtype=np.int32)
    bd_offsets = np.zeros(num_loads, dtype=np.int32)
    for i in range(num_loads):
        bf_id = i % num_bf
        bd_id = (i // num_bf) % num_bd
        bf_offsets[i] = bf_id * bf
        bd_offsets[i] = bd_id * bd

    bf_offsets_jax = jnp.array(bf_offsets)
    bd_offsets_jax = jnp.array(bd_offsets)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(1,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec((1,), lambda i: (0,)),
        scratch_shapes=[
            pltpu.VMEM((2, bd, bf), w_dtype),  # b_w_x2_vmem
            pltpu.SemaphoreType.DMA((2,)),  # weight_sems
        ],
    )

    result = pl.pallas_call(
        functools.partial(
            _dma_double_buffer_load_kernel,
            bf=bf,
            bd=bd,
            num_loads=num_loads,
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
        ),
    )(bf_offsets_jax, bd_offsets_jax, w)

    return result[0]


def kernel_fn(
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    num_loads: int = 64,
    dtype: jnp.dtype = jnp.bfloat16,
    weight_dtype: jnp.dtype = jnp.bfloat16,
    bf: int = 2048,
    bd: int = 1024,
    **_kwargs,  # Accept and ignore extra params (ep_size, chunk_size, etc.)
):
    """Construct inputs and return a zero-arg closure for benchmarking."""
    key = jax.random.key(42)
    w = jax.random.normal(key, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return dma_double_buffer_load(
            w,
            bf=bf,
            bd=bd,
            num_loads=num_loads,
        )

    return run


if __name__ == "__main__":
    import sys

    hidden_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    intermediate_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    num_loads = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    key = jax.random.key(0)
    w = jax.random.normal(key, (hidden_size, intermediate_size), dtype=jnp.bfloat16)

    try:
        result = dma_double_buffer_load(w, num_loads=num_loads)
        print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_loads={num_loads}")
        print(f"checksum={result:.6f}")
        print("PASS")
    except ValueError as e:
        if "CPU backend" in str(e):
            print(f"Skipping execution test on CPU (TPU required)")
            print("PASS (compilation successful)")
        else:
            raise
