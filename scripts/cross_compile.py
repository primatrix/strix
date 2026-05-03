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
    p.add_argument(
        "--tpu-type",
        default="v7x",
        help="TPU type for topology name prefix (default: v7x)",
    )
    p.add_argument(
        "--job-name",
        default=os.environ.get("JOB_NAME", ""),
        help="Job name for GCS upload (default: from JOB_NAME env var)",
    )
    p.add_argument(
        "--gcs-bucket",
        default=os.environ.get("GCS_BUCKET", ""),
        help="GCS bucket for result upload (default: from GCS_BUCKET env var)",
    )
    return p.parse_args()


def _parse_flag(env_value: str, flag_name: str) -> str | None:
    """Extract a flag value from a space-separated flags string."""
    import re
    m = re.search(rf"{re.escape(flag_name)}=(\S+)", env_value)
    return m.group(1) if m else None


def setup_ir_dump_flags(ir_root: pathlib.Path) -> pathlib.Path:
    """Set XLA_FLAGS and LIBTPU_INIT_ARGS for IR dump collection.

    If the env vars are already set (e.g. by K8s Job YAML), detects the
    actual dump root from the existing flags and returns it.  Otherwise
    configures dumps under *ir_root*.

    MUST be called before ``import jax``.

    Returns the effective IR root directory (may differ from *ir_root*
    when env vars were pre-set).
    """
    # If env vars are already set, derive ir_root from them.
    existing_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
    if existing_libtpu:
        llo_dir = _parse_flag(existing_libtpu, "--xla_jf_dump_to")
        if llo_dir:
            # llo_dir is e.g. "/tmp/ir_dumps/llo" → parent is the ir_root
            effective_root = pathlib.Path(llo_dir).parent
            for sub in ("hlo", "llo", "mosaic"):
                (effective_root / sub).mkdir(parents=True, exist_ok=True)
            print(f"Using pre-set IR dump root: {effective_root}")
            return effective_root

    # No pre-existing env vars — set our own.
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
    return ir_root


def create_virtual_topology(topology_str: str, tpu_name: str):
    """Create a virtual TPU topology using jax.experimental.topologies.

    Returns the topology descriptor whose ``.devices`` array contains virtual
    devices for the requested chip grid.
    """
    from jax.experimental import topologies  # noqa: E402

    # TPU topology_name format: "<name>:<chip_grid>", e.g. "TPU7x:2x8x8"
    # The name part must match libtpu's known external names.
    topology_name = f"{tpu_name}:{topology_str}"
    print(f"Using topology_name: {topology_name}")

    return topologies.get_topology_desc(
        topology_name,
        platform="tpu",
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
    ir_root = setup_ir_dump_flags(ir_root)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    print(f"JAX version : {jax.__version__}")
    local_device = jax.local_devices()[0]
    print(f"Local devices: {len(jax.local_devices())} × {local_device.device_kind}")

    # ── Step 1: Virtual topology ──
    # Use the device_kind (e.g. "TPU7x") as the topology name prefix,
    # since it matches libtpu's known external names.
    tpu_name = local_device.device_kind
    print(f"\nCreating virtual topology: {tpu_name}:{args.topology}")
    topo = create_virtual_topology(args.topology, tpu_name=tpu_name)
    virtual_devices = np.array(topo.devices).flatten()
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
    # Ensure num_tokens is large enough: local_num_tokens = num_tokens / ep_size
    # must be >= t_packing (2 for bf16). Auto-scale if needed.
    num_tokens = args.num_tokens
    t_packing = 32 // (jnp.dtype(jnp.bfloat16).itemsize * 8)  # 2 for bf16
    min_tokens = num_devices * t_packing  # each device needs >= t_packing tokens
    if num_tokens < min_tokens:
        print(f"Adjusting num_tokens from {num_tokens} to {min_tokens} "
              f"(EP={num_devices} × t_packing={t_packing})")
        num_tokens = min_tokens

    print(f"\nAllocating inputs on local TPU …")
    tokens = jnp.zeros((num_tokens, args.hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.zeros((args.num_experts, args.hidden_size, args.intermediate_size), dtype=jnp.bfloat16)
    w2 = jnp.zeros((args.num_experts, args.intermediate_size, args.hidden_size), dtype=jnp.bfloat16)
    w3 = jnp.zeros((args.num_experts, args.hidden_size, args.intermediate_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((num_tokens, args.top_k), dtype=jnp.float32) / args.top_k
    topk_ids = jnp.zeros((num_tokens, args.top_k), dtype=jnp.int32)
    w1_shared = jnp.zeros((args.hidden_size, args.se_intermediate_size), dtype=jnp.bfloat16)
    w2_shared = jnp.zeros((args.se_intermediate_size, args.hidden_size), dtype=jnp.bfloat16)
    w3_shared = jnp.zeros((args.hidden_size, args.se_intermediate_size), dtype=jnp.bfloat16)

    # ── Step 5: Block config ──
    # Adjust block config for the target EP size.
    # local_num_tokens = num_tokens / ep_size; bt must divide it.
    # bd1/bd2/bd1c/bd2c must be aligned to tile_align = t_packing * 128 = 256 (bf16).
    local_num_tokens = num_tokens // ep_size
    bt = min(8, local_num_tokens)  # clamp bt to local_num_tokens
    bt = max(bt, t_packing)  # must be >= t_packing
    block_config = FusedMoEBlockConfig(
        bt=bt, btc=2048, bf=2048, bfc=2048,
        bd1=256, bd1c=256, bd2=2048, bd2c=2048,
        bse=2048, bts=256,
    )

    # ── Step 6: Lower (trace + SPMD partitioning) ──
    # Use .lower() on the jitted fused_ep_moe to get the lowered form
    # without executing. This works with virtual devices.
    print(f"\nLowering SPMD kernel for {ep_size} devices …")
    t0 = time.monotonic()
    lowered = fused_ep_moe.lower(
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
    )
    t_lower = time.monotonic() - t0
    print(f"Lowered in {t_lower:.1f}s")

    # ── Step 7: Compile → LLO is generated here ──
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
    print(f"\nDone. Total wall time: {t_lower + t_compile:.1f}s")

    # ── Step 10: Package and upload to GCS (when running as K8s Job) ──
    if args.job_name and args.gcs_bucket:
        import tarfile
        tarball_path = pathlib.Path(f"/tmp/{args.job_name}.tar.gz")
        print(f"\nPackaging IR dumps to {tarball_path} ...")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(str(ir_root), arcname="ir_dumps")
        print(f"Tarball size: {tarball_path.stat().st_size:,} bytes")

        from google.cloud import storage
        gcs_path = args.gcs_bucket.removeprefix("gs://").strip("/")
        parts = gcs_path.split("/", 1)
        bucket_name = parts[0]
        bucket_prefix = parts[1] if len(parts) > 1 else ""
        blob_path = f"{args.job_name}/{tarball_path.name}"
        if bucket_prefix:
            blob_path = f"{bucket_prefix}/{blob_path}"
        print(f"Uploading to gs://{bucket_name}/{blob_path} ...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(tarball_path))
        print("Upload complete")


if __name__ == "__main__":
    main()
