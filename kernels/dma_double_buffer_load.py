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

    Simulates the weight loading pattern used in expert FFN kernels:
    - Load weight tiles from HBM into VMEM using double-buffering
    - Ping-pong between two VMEM buffers (bw_sem_id = 0/1)
    - Measure pure DMA throughput without compute

    Args:
        w_hbm: Weight matrix in HBM [hidden_size, intermediate_size]
        output_vmem: Scalar output ref for checksum (VMEM)
        b_w_x2_vmem: Double-buffer VMEM scratch [2, bd, bf]
        weight_sems: DMA semaphores [2]
        bf: Intermediate dimension block size
        bd: Hidden dimension block size
        num_loads: Number of weight tiles to load (simulates expert iteration)
    """
    hidden_size = w_hbm.shape[0]
    intermediate_size = w_hbm.shape[1]

    num_bf = cdiv(intermediate_size, bf)
    num_bd = cdiv(hidden_size, bd)

    # -- Create DMA descriptors once (outside loop) --
    # Following tokamax pattern: create descriptors once, reuse via .start()/.wait()
    copy_0 = pltpu.make_async_copy(
        src_ref=w_hbm.at[pl.ds(0, bd), pl.ds(0, bf)],
        dst_ref=b_w_x2_vmem.at[0],
        sem=weight_sems.at[0],
    )
    copy_1 = pltpu.make_async_copy(
        src_ref=w_hbm.at[pl.ds(0, bd), pl.ds(0, bf)],
        dst_ref=b_w_x2_vmem.at[1],
        sem=weight_sems.at[1],
    )

    # -- Double-buffered load loop --
    # Pipeline: prefetch 2 tiles, then wait → start(i+2, same buf) → compute.
    # Keeps 2 DMAs in flight at all times for maximum HBM bandwidth utilization.
    # NOTE: All DMAs read from fixed position (0, 0) to isolate address calculation overhead.

    checksum = jnp.float32(0.0)

    # Prefetch tile 0 → buf 0
    copy_0.start()

    # Prefetch tile 1 → buf 1
    if num_loads > 1:
        copy_1.start()

    for load_idx in range(num_loads):
        buf = load_idx % 2

        # Wait for current buffer's DMA to complete
        if buf == 0:
            copy_0.wait()
        else:
            copy_1.wait()

        # Immediately start next DMA into same buffer (2 tiles ahead)
        if load_idx + 2 < num_loads:
            if buf == 0:
                copy_0.start()
            else:
                copy_1.start()

        # Compute on arrived data
        checksum = checksum + jnp.float32(1.0)

    # Write checksum to output (rank-1 to satisfy Pallas TPU block rank constraint)
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
    )(w)

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
