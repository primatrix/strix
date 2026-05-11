"""DMA-only double-buffer expert kernel — f-axis tiling, no compute.

Isolates the DMA pipeline of the §5.8 f-axis tiling pattern:
persistent x, dual weight slots (W1/W3/W2), prologue/steady/epilogue.
All dot products and activation removed to measure pure DMA throughput.

Based on: §5.8 B=1 decode pipeline, d=8192, f=2048, bf16.
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_ALLOWED_CONFIGS = {(256, 512), (512, 256)}


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
        "DMA-only double-buffer expert — f-axis tiling, no compute "
        "(pure DMA pipeline throughput)"
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
    b_y_acc_vmem,
    b_y_out_vmem,
    # Semaphores.
    weight_sems,
    x_sem,
    y_out_sem,
    *,
    bf: int,
    intermediate_size: int,
):
    """Pallas kernel body — DMA pipeline only, no compute."""
    n_w = intermediate_size // bf

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

    # -- Wait-only tile (no compute) --
    def drain_tile(slot):
        """Wait for all DMA of this tile to complete, no compute."""
        wait_fetch_w2(slot)

    # -- Prologue --
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

    # -- Steady state --
    for tile in range(1, n_w - 1):
        slot = tile % 2
        next_slot = 1 - slot
        start_fetch_w1(next_slot, tile + 1)
        start_fetch_w3(next_slot, tile + 1)
        start_fetch_w2(next_slot, tile + 1)
        wait_fetch_w1(slot)
        wait_fetch_w3(slot)
        drain_tile(slot)

    # -- Epilogue --
    if n_w >= 2:
        last_slot = (n_w - 1) % 2
        wait_fetch_w1(last_slot)
        wait_fetch_w3(last_slot)
        drain_tile(last_slot)

    # -- Write-back: zero output --
    b_y_acc_vmem[...] = jnp.zeros_like(b_y_acc_vmem)
    b_y_out_vmem[...] = b_y_acc_vmem[...].astype(b_y_out_vmem.dtype)
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0],
    ).start()
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0],
    ).wait()


@functools.partial(jax.jit, static_argnames=["bf"])
def double_buffer_expert(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    bf: int = 512,
) -> jax.Array:
    """Run the DMA-only double-buffer expert kernel (no compute).

    Shapes:
      tokens: (bt, d) bf16
      w1:     (d, f)  bf16
      w2:     (f, d)  bf16
      w3:     (d, f)  bf16
    Returns:
      output: (bt, d) bf16  (zeros — no computation performed)
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
        pltpu.VMEM((bt, d), jnp.float32),                # b_y_acc_vmem (fp32)
        pltpu.VMEM((bt, d), dtype),                      # b_y_out_vmem (bf16 stage)
        pltpu.SemaphoreType.DMA((2, 3)),                 # weight_sems[slot, channel]
        pltpu.SemaphoreType.DMA((1,)),                   # x_sem
        pltpu.SemaphoreType.DMA((1,)),                   # y_out_sem
    )

    scope_name = f"dma-only-expert-bt{bt}-bf{bf}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = pl.pallas_call(
        functools.partial(
            _double_buffer_expert_kernel,
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
            vmem_limit_bytes=96 * 1024 * 1024,
        ),
        name=scope_name,
    )
    return jax.named_scope(scope_name)(kernel)(
        pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
        pltpu.with_memory_space_constraint(w1, pltpu.HBM),
        pltpu.with_memory_space_constraint(w2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w3, pltpu.HBM),
    )


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
    from ._fused_moe_impl import activation_fn
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
    """Build random inputs and return a zero-arg closure calling the kernel."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=weight_dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=weight_dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return double_buffer_expert(
            tokens, w1, w2, w3, bf=bf,
        )

    return run


if __name__ == "__main__":
    bt = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    bf_arg = 512 if bt == 256 else 256

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, 8192), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (8192, 2048), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (2048, 8192), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (8192, 2048), dtype=jnp.bfloat16)

    result = double_buffer_expert(tokens, w1, w2, w3, bf=bf_arg)
    print(f"bt={bt}, bf={bf_arg}, output_shape={result.shape}")
    print(f"output is zeros: {jnp.allclose(result, 0.0)}")
    print("PASS")
