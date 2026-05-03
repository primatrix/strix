#!/usr/bin/env python3
"""Cross-compile fused MoE kernel for a target TPU topology and export LLO.

Runs on a small TPU slice (e.g. 4-chip v7x) to compile the fused MoE kernel
targeting a *larger* topology (e.g. 2x8x8 = 128 chips, 256 devices with
2 TensorCores/chip).  The compilation uses libtpu on the local TPU, but the
generated SPMD program (and its LLO) targets the virtual topology.

Usage (on a TPU VM):
    python scripts/cross_compile.py --topology 2x8x8

Prerequisites:
    - Must run on a TPU VM (needs libtpu.so for compilation)
    - JAX with jax.experimental.topologies support (JAX >= 0.5.x recommended)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-compile fused MoE for a target TPU topology"
    )
    p.add_argument(
        "--topology",
        default="2x8x8",
        help="Target TPU chip topology, e.g. '2x8x8' (default: 2x8x8)",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/cross_compile",
        help="Root output directory for IR dumps (default: /tmp/cross_compile)",
    )
    # Ling 2.6 1T decode defaults
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=8192)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--se-intermediate-size", type=int, default=2048)
    return p.parse_args()


def setup_ir_dump_flags(ir_root: pathlib.Path):
    """Set XLA_FLAGS and LIBTPU_INIT_ARGS for IR dump collection.

    MUST be called before ``import jax``.
    """
    for sub in ("hlo", "llo", "mosaic"):
        (ir_root / sub).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_dump_hlo_as_text --xla_dump_to={ir_root / 'hlo'}",
    )
    os.environ.setdefault(
        "LIBTPU_INIT_ARGS",
        " ".join([
            "--xla_enable_custom_call_region_trace=true",
            "--xla_xprof_register_llo_debug_info=true",
            f"--xla_jf_dump_to={ir_root / 'llo'}",
            "--xla_jf_dump_hlo_text=true",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_emit_annotations=true",
            f"--xla_mosaic_dump_to={ir_root / 'mosaic'}",
            "--xla_mosaic_enable_llo_source_annotations=true",
        ]),
    )


def create_virtual_topology(topology_str: str):
    """Create a virtual TPU topology using jax.experimental.topologies.

    Returns the topology descriptor whose ``.devices`` array contains virtual
    devices for the requested chip grid.
    """
    from jax.experimental import topologies  # noqa: E402

    # The API may accept (platform, topology=...) or (name, platform, topology=...).
    # Try the kwargs-only form first, then the positional form.
    try:
        return topologies.get_topology_desc(
            platform="tpu",
            topology=topology_str,
        )
    except TypeError:
        return topologies.get_topology_desc(
            "cross_compile",
            "tpu",
            topology=topology_str,
        )


def report_ir_dumps(ir_root: pathlib.Path):
    """Print a summary of dumped IR files."""
    print(f"\n{'=' * 60}")
    print(f"IR dumps → {ir_root}")
    for sub in ("llo", "hlo", "mosaic"):
        path = ir_root / sub
        files = sorted(f for f in path.rglob("*") if f.is_file())
        print(f"\n  {sub}/ — {len(files)} files")
        for f in files[:8]:
            size = f.stat().st_size
            rel = f.relative_to(ir_root)
            print(f"    {rel}  ({size:,} bytes)")
        if len(files) > 8:
            print(f"    … and {len(files) - 8} more")


def main():
    args = parse_args()
    ir_root = pathlib.Path(args.output_dir) / args.topology

    # ── Step 0: IR dump flags (BEFORE import jax) ──
    setup_ir_dump_flags(ir_root)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    print(f"JAX version : {jax.__version__}")
    print(f"Local devices: {len(jax.local_devices())} × {jax.local_devices()[0].device_kind}")

    # ── Step 1: Virtual topology ──
    print(f"\nCreating virtual topology: {args.topology}")
    topo = create_virtual_topology(args.topology)
    virtual_devices = topo.devices.flatten()
    num_devices = len(virtual_devices)
    print(f"Virtual devices: {num_devices}")

    # ── Step 2: Virtual mesh ──
    # fused_ep_moe expects a 2D mesh ("data", "tensor") where
    # ep_size = data_size * tensor_size = num_devices.
    virtual_mesh = Mesh(
        np.array(virtual_devices).reshape(1, -1),
        axis_names=("data", "tensor"),
    )
    ep_size = num_devices
    print(f"Virtual mesh : {dict(virtual_mesh.shape)}, EP={ep_size}")

    # ── Step 3: Import kernel ──
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kernels._fused_moe_impl import FusedMoEBlockConfig, fused_ep_moe

    # ── Step 4: Input arrays (zeros — values irrelevant for compilation) ──
    print(f"\nAllocating inputs on local TPU …")
    tokens = jnp.zeros((args.num_tokens, args.hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.zeros((args.num_experts, args.hidden_size, args.intermediate_size), dtype=jnp.bfloat16)
    w2 = jnp.zeros((args.num_experts, args.intermediate_size, args.hidden_size), dtype=jnp.bfloat16)
    w3 = jnp.zeros((args.num_experts, args.hidden_size, args.intermediate_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((args.num_tokens, args.top_k), dtype=jnp.float32) / args.top_k
    topk_ids = jnp.zeros((args.num_tokens, args.top_k), dtype=jnp.int32)
    w1_shared = jnp.zeros((args.hidden_size, args.se_intermediate_size), dtype=jnp.bfloat16)
    w2_shared = jnp.zeros((args.se_intermediate_size, args.hidden_size), dtype=jnp.bfloat16)
    w3_shared = jnp.zeros((args.hidden_size, args.se_intermediate_size), dtype=jnp.bfloat16)

    # ── Step 5: Block config ──
    # bt will be clamped to local_num_tokens (= num_tokens // ep_size) by
    # effective_for().  Use the closest tuned config as starting point.
    block_config = FusedMoEBlockConfig(
        bt=8, btc=2048, bf=2048, bfc=2048,
        bd1=8, bd1c=8, bd2=2048, bd2c=2048,
        bse=2048, bts=256,
    )

    # ── Step 6: Compile (compile_only=True → returns (kernel_fn, args)) ──
    print(f"\nBuilding SPMD kernel for {ep_size} devices …")
    t0 = time.monotonic()
    kernel_fn, kernel_args = fused_ep_moe(
        mesh=virtual_mesh,
        tokens=tokens,
        w1=w1, w2=w2, w3=w3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=args.top_k,
        act_fn="silu",
        w1_shared=w1_shared,
        w2_shared=w2_shared,
        w3_shared=w3_shared,
        block_config=block_config,
        compile_only=True,
    )
    t_build = time.monotonic() - t0
    print(f"Kernel built in {t_build:.1f}s")

    # ── Step 7: Lower ──
    print(f"Lowering (tracing SPMD for {ep_size}-device mesh) …")
    t0 = time.monotonic()
    lowered = kernel_fn.lower(*kernel_args)
    t_lower = time.monotonic() - t0
    print(f"Lowered in {t_lower:.1f}s")

    # ── Step 8: Compile → LLO is generated here ──
    print(f"Compiling (libtpu XLA → LLO for {args.topology} topology) …")
    t0 = time.monotonic()
    compiled = lowered.compile()
    t_compile = time.monotonic() - t0
    print(f"Compiled in {t_compile:.1f}s")

    # Optional: cost analysis
    try:
        cost = compiled.cost_analysis()
        if cost:
            flops = cost[0].get("flops", "N/A") if isinstance(cost, list) else cost.get("flops", "N/A")
            print(f"Cost analysis — flops: {flops}")
    except Exception:
        pass

    # ── Step 9: Report ──
    report_ir_dumps(ir_root)
    print(f"\nDone. Total wall time: {t_build + t_lower + t_compile:.1f}s")


if __name__ == "__main__":
    main()
