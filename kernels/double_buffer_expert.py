"""Double-buffer expert FFN kernel — §5.8 decode pipeline for Ling 2.6.

Single routed expert, B=1, persistent x, dual weight slots, low-priority
W2 DMA, fp32 y_acc accumulator. Hard-coded for d=8192, f=2048, bf16 tokens dynamically quantized
to fp8, fp8×fp8 MXU matmul.

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

from ._fused_moe_impl import activation_fn


_ALLOWED_CONFIGS = {(256, 256), (512, 256)}


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "float8_e4m3fn",
    "act_fn": "silu",
    "bf": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Double-buffer expert FFN — §5.8 B=1 decode "
        "(bf16→fp8 tokens, fp8×fp8 MXU, dual W slots, low-pri W2)"
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
    act_fn: str,
    bf: int,
    intermediate_size: int,
):
    """Pallas kernel body — §5.8 B=1 decode pipeline."""
    n_w = intermediate_size // bf  # 4 (bf=512) or 8 (bf=256)

    # -- DMA helpers --
    def start_load_x(priority=1):
        pltpu.make_async_copy(
            src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).start(priority=priority)

    def wait_load_x():
        pltpu.make_async_copy(
            src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).wait()

    def start_fetch_w1(slot, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start(priority=priority)

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w3(slot, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start(priority=priority)

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def start_fetch_w2(slot, tile_idx, priority=0):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start(priority=priority)

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()

    # -- Compute function (W2 wait deferred inside) --
    def compute_tile(slot, is_first_tile):
        """Gate/up compute (W1, W3), then wait W2 (low-pri overlap), then accumulate."""
        x = b_x_vmem[...]
        x_fp8 = x.astype(jnp.float8_e4m3fn)
        w1 = b_w1_x2_vmem[slot]
        w3 = b_w3_x2_vmem[slot]
        gate = jnp.dot(x_fp8, w1, preferred_element_type=jnp.float32)
        up = jnp.dot(x_fp8, w3, preferred_element_type=jnp.float32)
        act_up = activation_fn(gate, up, act_fn)

        wait_fetch_w2(slot)  # deferred — overlaps with gate/up MXU

        act_up_fp8 = act_up.astype(jnp.float8_e4m3fn)
        w2 = b_w2_x2_vmem[slot]
        partial = jnp.dot(act_up_fp8, w2, preferred_element_type=jnp.float32)
        if is_first_tile:
            b_y_acc_vmem[...] = partial
        else:
            b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial

    # -- Prologue (§5.8 stages L1..C2) --
    start_load_x(priority=1)
    start_fetch_w1(0, 0, priority=1)
    start_fetch_w3(0, 0, priority=1)
    start_fetch_w2(0, 0, priority=1)
    
    if n_w >= 2:
        start_fetch_w1(1, 1)
        start_fetch_w3(1, 1)
        start_fetch_w2(1, 1)

    wait_load_x()
    wait_fetch_w1(0)
    wait_fetch_w3(0)
    compute_tile(slot=0, is_first_tile=True)

    # -- Steady state: Python for-loop over tile in [1, n_w - 1) --
    for tile in range(1, n_w - 1):
        slot = tile % 2
        next_slot = 1 - slot
        start_fetch_w1(next_slot, tile + 1)
        start_fetch_w3(next_slot, tile + 1)
        start_fetch_w2(next_slot, tile + 1)
        wait_fetch_w1(slot)
        wait_fetch_w3(slot)
        compute_tile(slot, is_first_tile=False)

    # -- Epilogue: last tile, no more loads --
    if n_w >= 2:
        last_slot = (n_w - 1) % 2
        wait_fetch_w1(last_slot)
        wait_fetch_w3(last_slot)
        compute_tile(last_slot, is_first_tile=False)

    # -- Write-back: y_acc (fp32) → b_y_out_vmem (bf16) → output_hbm --
    b_y_out_vmem[...] = b_y_acc_vmem[...].astype(b_y_out_vmem.dtype)
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0],
    ).start()
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0],
    ).wait()


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
      tokens: (bt, d) bf16 (quantized to fp8 in kernel)
      w1:     (d, f)  fp8
      w2:     (f, d)  fp8
      w3:     (d, f)  fp8
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
        pltpu.VMEM((bt, d), jnp.float32),                # b_y_acc_vmem (fp32)
        pltpu.VMEM((bt, d), dtype),                      # b_y_out_vmem (bf16 stage)
        pltpu.SemaphoreType.DMA((2, 3)),                 # weight_sems[slot, channel]
        pltpu.SemaphoreType.DMA((1,)),                   # x_sem
        pltpu.SemaphoreType.DMA((1,)),                   # y_out_sem
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
    bf: int = 256,
    **_kwargs,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel.

    Contract matches scripts/benchmark_runner.py (see expert_ffn.py).
    """
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=jnp.bfloat16).astype(weight_dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=jnp.bfloat16).astype(weight_dtype)

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
    w1 = jax.random.normal(k2, (8192, 2048), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    w2 = jax.random.normal(k3, (2048, 8192), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)
    w3 = jax.random.normal(k4, (8192, 2048), dtype=jnp.bfloat16).astype(jnp.float8_e4m3fn)

    result = double_buffer_expert(tokens, w1, w2, w3, bf=bf_arg)
    ref = _ref_expert_ffn(tokens, w1, w2, w3)
    result_f32 = result.astype(jnp.float32)
    ref_f32 = ref.astype(jnp.float32)
    max_err = jnp.max(jnp.abs(result_f32 - ref_f32))
    rel_err = max_err / (jnp.max(jnp.abs(ref_f32)) + 1e-6)
    print(f"bt={bt}, bf={bf_arg}, max_abs_err={max_err:.4f}, rel_err={rel_err:.4f}")
    assert rel_err < 0.05, f"rel_err too high: {rel_err}"
    print("PASS")
