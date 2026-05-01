#!/usr/bin/env python3
"""benchmark_runner.py — Execute kernel benchmark on TPU Pod.

Runs inside a K8s Job: dynamically imports a kernel module, executes
JAX compile + micro-benchmark, collects IR dumps, and uploads results
to GCS.
"""

from __future__ import annotations

import argparse
import os
import pathlib

IR_DUMP_SUBDIRS = ("hlo", "llo", "mosaic")


def parse_args(argv=None):
    """Parse CLI arguments with env var fallbacks."""
    p = argparse.ArgumentParser(description="Run kernel benchmark on TPU Pod")

    p.add_argument(
        "--kernel",
        default=os.environ.get("KERNEL_MODULE"),
        help="Kernel module path (e.g. kernels.chunk_kda_fwd)",
    )
    p.add_argument(
        "--shape",
        default=os.environ.get("SHAPE"),
        help="Comma-separated shape dimensions",
    )
    p.add_argument(
        "--job-name",
        default=os.environ.get("JOB_NAME", "benchmark"),
        help="Job name for output paths",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=_int_or_none(os.environ.get("CHUNK_SIZE")),
        help="Chunk size for the kernel",
    )
    p.add_argument(
        "--gcs-bucket",
        default=os.environ.get("GCS_BUCKET", "gs://poc_profile/"),
        help="GCS bucket for result upload",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    p.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )

    args = p.parse_args(argv)

    if args.kernel is None:
        p.error("--kernel is required (or set KERNEL_MODULE env var)")
    if args.shape is None:
        p.error("--shape is required (or set SHAPE env var)")

    return args


def _int_or_none(val):
    """Convert string to int, returning None for empty/missing values."""
    if val is None or val == "":
        return None
    return int(val)


def setup_ir_dump_dirs(root):
    """Create IR dump directories under *root*."""
    root = pathlib.Path(root)
    for subdir in IR_DUMP_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def setup_xla_flags(ir_dump_root):
    """Set XLA_FLAGS and LIBTPU_INIT_ARGS if not already present."""
    ir_dump_root = pathlib.Path(ir_dump_root)

    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = (
            f"--xla_dump_hlo_as_text --xla_dump_to={ir_dump_root / 'hlo'}"
        )

    if "LIBTPU_INIT_ARGS" not in os.environ:
        os.environ["LIBTPU_INIT_ARGS"] = " ".join([
            "--xla_enable_custom_call_region_trace=true",
            "--xla_xprof_register_llo_debug_info=true",
            f"--xla_jf_dump_to={ir_dump_root / 'llo'}",
            "--xla_jf_dump_hlo_text=true",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_emit_annotations=true",
            f"--xla_mosaic_dump_to={ir_dump_root / 'mosaic'}",
            "--xla_mosaic_enable_llo_source_annotations=true",
        ])


if __name__ == "__main__":
    main()
