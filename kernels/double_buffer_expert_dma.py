"""DMA-only double-buffer expert kernel — no compute, pure DMA pipeline.

Same DMA structure as the compute version (persistent x, dual weight
slots, W1/W3/W2 f-axis tiling) but replaces compute_tile with drain_tile
that only waits for DMA completion. Used to isolate DMA overhead.
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_ALLOWED_CONFIGS = {(256, 256), (512, 256)}


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "DMA-only double-buffer expert — §5.8 pipeline, "
        "no compute (drain_tile)"
    ),
}


def _double_buffer_expert_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    output_hbm,
    # Scratch buffers (VMEM).
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_x_vmem,
    # Semaphores.
    weight_sems,
    x_sem,
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
):
    """Pallas kernel body — §5.8 B=1 decode pipeline."""
    n_w = intermediate_size // bf  # 4 (bf=512) or 8 (bf=256)

    # -- DMA helpers --
    def start_load_x():
        pltpu.make_async_copy(
            src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).start()

    def wait_load_x():
        pltpu.make_async_copy(
            src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).wait()

    def start_fetch_w1(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start()

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w3(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start()

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def start_fetch_w2(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start()

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()

    # -- Drain function: wait for DMA only, no compute --
    def drain_tile(slot):
        """Wait for all DMA of this tile to complete, no compute."""
        wait_fetch_w2(slot)

    # -- Prologue (§5.8 stages L1..C2) --
    start_load_x()
    start_fetch_w1(0, 0)
    start_fetch_w3(0, 0)
    start_fetch_w2(0, 0)
    wait_load_x()
    wait_fetch_w1(0)
    wait_fetch_w3(0)

    if n_w >= 2:
        start_fetch_w1(1, 1)
        start_fetch_w3(1, 1)
        start_fetch_w2(1, 1)

    drain_tile(slot=0)

    # -- Steady state: Python for-loop over tile in [1, n_w - 1) --
    for tile in range(1, n_w - 1):
        slot = tile % 2
        next_slot = 1 - slot
        start_fetch_w1(next_slot, tile + 1)
        start_fetch_w3(next_slot, tile + 1)
        start_fetch_w2(next_slot, tile + 1)
        wait_fetch_w1(slot)
        wait_fetch_w3(slot)
        drain_tile(slot)

    # -- Epilogue: last tile, no more loads --
    if n_w >= 2:
        last_slot = (n_w - 1) % 2
        wait_fetch_w1(last_slot)
        wait_fetch_w3(last_slot)
        drain_tile(last_slot)


@functools.partial(jax.jit, static_argnames=["act_fn", "bf"])
def double_buffer_expert(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 256,
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

    dtype = tokens.dtype
    weight_dtype = w1.dtype

    scratch_shapes = (
        pltpu.VMEM((2, d, bf), weight_dtype),            # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),            # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),            # b_w2_x2_vmem
        pltpu.VMEM((bt, d), dtype),                      # b_x_vmem
        pltpu.SemaphoreType.DMA((2, 3)),                 # weight_sems[slot, channel]
        pltpu.SemaphoreType.DMA((1,)),                   # x_sem
    )

    scope_name = f"double-buffer-expert-bt{bt}-bf{bf}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = pl.pallas_call(
        functools.partial(
            _double_buffer_expert_kernel,
            act_fn=act_fn,
            bf=bf,
            intermediate_size=f_full,
        ),
        out_shape=jax.ShapeDtypeStruct((bt, d), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm, hbm, hbm],
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
    )


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 256,
    **_kwargs,
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


if __name__ == "__main__":
    bt = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    bf_arg = 256

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, 8192), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (8192, 2048), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (2048, 8192), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (8192, 2048), dtype=jnp.bfloat16)

    result = double_buffer_expert(tokens, w1, w2, w3, bf=bf_arg)
    print(f"bt={bt}, bf={bf_arg}, output_shape={result.shape}")
    print("PASS")
