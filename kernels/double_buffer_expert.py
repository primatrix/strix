"""Double-buffer expert FFN kernel — d-axis tiled, three-phase pipeline.

Single routed expert, B=1, persistent x, dual weight slots, d-axis tiling
for smaller tiles and lower VMEM. Hard-coded for d=8192, f=2048, bf16.

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


_ALLOWED_CONFIGS = {(256, 256), (256, 512), (256, 1024), (512, 256)}


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
        "n_stages": 4,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bd": 256,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Double-buffer expert FFN — §5.8 B=1 decode "
        "(persistent x, d-axis tiling, N-stage weight buffer)"
    ),
}


def _double_buffer_expert_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    output_hbm,
    # Scratch buffers (VMEM).
    b_w13_x2_vmem,
    b_w2_x2_vmem,
    b_x_vmem,
    gate_acc_vmem,
    up_acc_vmem,
    b_y_out_vmem,
    # Semaphores.
    weight_sems,
    x_sem,
    y_out_sem,
    *,
    act_fn: str,
    bd: int,
    intermediate_size: int,
    hidden_size: int,
    n_stages: int,
):
    """Pallas kernel body — d-axis tiled, three-phase pipeline.

    Phase 1: accumulate gate/up across d-chunks.
    Phase 2: element-wise activation.
    Phase 3: compute independent W2 output d-chunks, DMA each to HBM.
    """
    n_d = hidden_size // bd
    bt = b_x_vmem.shape[0]

    # -- DMA helpers --
    def start_load_x():
        pltpu.make_async_copy(
            src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).start()

    def wait_load_x():
        pltpu.make_async_copy(
            src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).wait()

    def start_fetch_w13(slot, tile_idx):
        sem = weight_sems.at[slot, 0]
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[pl.ds(tile_idx * bd, bd), :],
            dst_ref=b_w13_x2_vmem.at[slot, 0],
            sem=sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[pl.ds(tile_idx * bd, bd), :],
            dst_ref=b_w13_x2_vmem.at[slot, 1],
            sem=sem,
        ).start()

    def wait_fetch_w13(slot):
        pltpu.make_async_copy(
            src_ref=b_w13_x2_vmem.at[slot],
            dst_ref=b_w13_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w2(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[:, pl.ds(tile_idx * bd, bd)],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start()

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def wait_y_out():
        pltpu.make_async_copy(
            src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0],
        ).wait()

    # ==================== Phase 1: Accumulate gate/up ====================
    start_load_x()
    for s in range(min(n_stages - 1, n_d)):
        start_fetch_w13(s, s)
    wait_load_x()
    gate_acc_vmem[...] = jnp.zeros(gate_acc_vmem.shape, dtype=jnp.float32)
    up_acc_vmem[...] = jnp.zeros(up_acc_vmem.shape, dtype=jnp.float32)

    @pl.loop(0, n_d, unroll=False)
    def _phase1(tile_idx):
        slot = tile_idx % n_stages

        @pl.when(tile_idx + n_stages - 1 < n_d)
        def _prefetch():
            pf_slot = (tile_idx + n_stages - 1) % n_stages
            start_fetch_w13(pf_slot, tile_idx + n_stages - 1)

        wait_fetch_w13(slot)

        x_chunk = b_x_vmem[:, pl.ds(tile_idx * bd, bd)]
        w1_tile = b_w13_x2_vmem[slot, 0]
        w3_tile = b_w13_x2_vmem[slot, 1]

        gate_acc_vmem[...] = gate_acc_vmem[...] + jnp.dot(
            x_chunk, w1_tile, preferred_element_type=jnp.float32
        )
        up_acc_vmem[...] = up_acc_vmem[...] + jnp.dot(
            x_chunk, w3_tile, preferred_element_type=jnp.float32
        )

    # ==================== Phase 2: Activation ====================
    gate_acc_vmem[...] = activation_fn(
        gate_acc_vmem[...], up_acc_vmem[...], act_fn
    )

    # ==================== Phase 3: W2 output chunks ====================
    for s in range(min(n_stages - 1, n_d)):
        start_fetch_w2(s, s)

    @pl.loop(0, n_d, unroll=False)
    def _phase3(tile_idx):
        slot = tile_idx % n_stages

        @pl.when(tile_idx + n_stages - 1 < n_d)
        def _prefetch():
            pf_slot = (tile_idx + n_stages - 1) % n_stages
            start_fetch_w2(pf_slot, tile_idx + n_stages - 1)

        wait_fetch_w2(slot)

        w2_tile = b_w2_x2_vmem[slot]
        y_chunk = jnp.dot(
            gate_acc_vmem[...], w2_tile, preferred_element_type=jnp.float32
        )

        @pl.when(tile_idx > 0)
        def _wait_prev_write():
            wait_y_out()

        b_y_out_vmem[...] = y_chunk.astype(b_y_out_vmem.dtype)
        pltpu.make_async_copy(
            src_ref=b_y_out_vmem,
            dst_ref=output_hbm.at[:, pl.ds(tile_idx * bd, bd)],
            sem=y_out_sem.at[0],
        ).start()

    wait_y_out()


@functools.partial(jax.jit, static_argnames=["act_fn", "bd", "n_stages"])
def double_buffer_expert(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bd: int = 256,
    n_stages: int = 2,
) -> jax.Array:
    """Run the double-buffer expert FFN kernel (d-axis tiled).

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
    if (bt, bd) not in _ALLOWED_CONFIGS:
        raise ValueError(
            f"Only (num_tokens, bd) in {sorted(_ALLOWED_CONFIGS)} supported; "
            f"got (num_tokens={bt}, bd={bd})"
        )
    if d % bd != 0:
        raise ValueError(f"bd={bd} must divide d={d}")
    if w3.shape != (d, f_full):
        raise ValueError(f"w3.shape={w3.shape} must be (d={d}, f={f_full})")
    if w2.shape != (f_full, d):
        raise ValueError(f"w2.shape={w2.shape} must be (f={f_full}, d={d})")

    dtype = tokens.dtype
    weight_dtype = w1.dtype

    scratch_shapes = (
        pltpu.VMEM((n_stages, 2, bd, f_full), weight_dtype),  # b_w13_vmem[slot, {W1,W3}]
        pltpu.VMEM((n_stages, f_full, bd), weight_dtype),      # b_w2_vmem[slot]
        pltpu.VMEM((bt, d), dtype),                            # b_x_vmem (persistent)
        pltpu.VMEM((bt, f_full), jnp.float32),                 # gate_acc_vmem (fp32)
        pltpu.VMEM((bt, f_full), jnp.float32),                 # up_acc_vmem (fp32)
        pltpu.VMEM((bt, bd), dtype),                           # b_y_out_vmem (bf16 stage)
        pltpu.SemaphoreType.DMA((n_stages, 2)),                # weight_sems[slot, {W13,W2}]
        pltpu.SemaphoreType.DMA((1,)),                         # x_sem
        pltpu.SemaphoreType.DMA((1,)),                         # y_out_sem
    )

    scope_name = f"double-buffer-expert-bt{bt}-bd{bd}-n{n_stages}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = pl.pallas_call(
        functools.partial(
            _double_buffer_expert_kernel,
            act_fn=act_fn,
            bd=bd,
            intermediate_size=f_full,
            hidden_size=d,
            n_stages=n_stages,
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
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
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
    bd: int = 256,
    n_stages: int = 2,
    **_extra,
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
            tokens, w1, w2, w3, act_fn=act_fn, bd=bd, n_stages=n_stages,
        )

    return run


if __name__ == "__main__":
    bt = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    bd_arg = 256
    n_stages_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, 8192), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (8192, 2048), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (2048, 8192), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (8192, 2048), dtype=jnp.bfloat16)

    result = double_buffer_expert(tokens, w1, w2, w3, bd=bd_arg, n_stages=n_stages_arg)
    ref = _ref_expert_ffn(tokens, w1, w2, w3)
    result_f32 = result.astype(jnp.float32)
    ref_f32 = ref.astype(jnp.float32)
    max_err = jnp.max(jnp.abs(result_f32 - ref_f32))
    rel_err = max_err / (jnp.max(jnp.abs(ref_f32)) + 1e-6)
    print(f"bt={bt}, bd={bd_arg}, n_stages={n_stages_arg}, max_abs_err={max_err:.4f}, rel_err={rel_err:.4f}")
    assert rel_err < 0.05, f"rel_err too high: {rel_err}"
    print("PASS")
