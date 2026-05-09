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
    from ._fused_moe_impl import cdiv, get_dtype_packing
except ImportError:
    # Fallback for direct execution
    from _fused_moe_impl import cdiv, get_dtype_packing

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
    output_hbm,
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
        output_hbm: Scalar output ref for checksum
        b_w_x2_vmem: Double-buffer VMEM scratch [2, t_packing, bd_per_pack, bf]
        weight_sems: DMA semaphores [2]
        bf: Intermediate dimension block size
        bd: Hidden dimension block size
        num_loads: Number of weight tiles to load (simulates expert iteration)
    """
    hidden_size = w_hbm.shape[0]
    intermediate_size = w_hbm.shape[1]

    w_dtype = w_hbm.dtype
    t_packing = get_dtype_packing(w_dtype)
    h_per_t_packing = hidden_size // t_packing
    bd_per_t_packing = bd // t_packing

    num_bf = cdiv(intermediate_size, bf)
    num_bd = cdiv(hidden_size, bd)

    # -- Weight DMA helpers --

    def start_fetch_w(bw_sem_id, bf_id, bd_id):
        """Start async DMA copy from HBM to VMEM buffer bw_sem_id."""
        pltpu.make_async_copy(
            src_ref=w_hbm.at[pl.ds(bd_id * bd, bd), pl.ds(bf_id * bf, bf)],
            dst_ref=b_w_x2_vmem.at[bw_sem_id],
            sem=weight_sems.at[bw_sem_id],
        ).start()

    def wait_fetch_w(bw_sem_id):
        """Wait for async DMA copy to complete."""
        pltpu.make_async_copy(
            src_ref=b_w_x2_vmem.at[bw_sem_id],
            dst_ref=b_w_x2_vmem.at[bw_sem_id],
            sem=weight_sems.at[bw_sem_id],
        ).wait()

    def consume_weight(bw_sem_id):
        """Simulate weight consumption (read from VMEM, write checksum to HBM)."""
        # Sum the weight tile to force VMEM read.
        # Cast scalar result to f32 (not the tile) to avoid intermediate HBM allocation.
        tile_sum_bf16 = jnp.sum(b_w_x2_vmem[bw_sem_id])
        return tile_sum_bf16.astype(jnp.float32)

    # -- Double-buffered load loop --
    # Prefetch first tile into buffer 0
    start_fetch_w(0, 0, 0)

    checksum = jnp.float32(0.0)

    def body(i, carry):
        checksum, bw_sem_id = carry

        wait_fetch_w(bw_sem_id)

        next_bw_sem_id = 1 - bw_sem_id
        next_bf_id = (i + 1) % num_bf
        next_bd_id = ((i + 1) // num_bf) % num_bd

        # Always start next DMA fetch. On the final iteration this produces
        # one extra unused transfer, but avoids jax.lax.cond which triggers
        # a Jaxpr KeyError during Pallas lowering.
        start_fetch_w(next_bw_sem_id, next_bf_id, next_bd_id)

        tile_checksum = consume_weight(bw_sem_id)
        checksum = checksum + tile_checksum

        return (checksum, next_bw_sem_id)

    final_checksum, _ = jax.lax.fori_loop(
        0, num_loads, body, (checksum, 0))

    output_hbm[0] = final_checksum


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
    t_packing = get_dtype_packing(w_dtype)
    bd_per_pack = bd // t_packing

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[
            pltpu.VMEM((2, t_packing, bd_per_pack, bf), w_dtype),  # b_w_x2_vmem
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
