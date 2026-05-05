#!/usr/bin/env python3
"""benchmark_runner.py — Execute kernel benchmark on TPU Pod.

Runs inside a K8s Job: dynamically imports a kernel module, executes
JAX compile + micro-benchmark, collects IR dumps, and uploads results
to GCS.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import statistics
import sys
import tarfile
import time

IR_DUMP_SUBDIRS = ("hlo", "llo", "mosaic")


def is_coordinator():
    """Return True if this process is the coordinator (process_index == 0).

    In multi-host TPU Pod runs, only the coordinator (process_index 0) should
    perform dump file generation and GCS upload to avoid redundancy.
    """
    import jax
    return jax.process_index() == 0


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
        help="Comma-separated shape dimensions (metadata; kernel uses config defaults)",
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
        "--ep-size",
        type=int,
        default=_int_or_none(os.environ.get("EP_SIZE")),
        help="Expert parallelism size (overrides kernel config default)",
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


def import_kernel(module_path):
    """Dynamically import a kernel module and return (kernel_fn, config)."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)
    mod = importlib.import_module(module_path)
    return mod.kernel_fn, mod.config


def run_benchmark(kernel_fn, config, num_warmup, num_runs, chunk_size=None, ep_size=None):
    """Execute kernel benchmark and return list of timing values (seconds)."""
    kwargs = dict(config.get("default_shape", {}))
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    kwargs["ep_size"] = ep_size if ep_size is not None else config.get("ep_size", 4)

    run_fn = kernel_fn(**kwargs)

    # Warmup
    for _ in range(num_warmup):
        result = run_fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = run_fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        timings.append(time.perf_counter() - start)

    return timings


def write_benchmark_result(timings, kernel, shape, job_name, config, output_path):
    """Write benchmark results to JSON file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "kernel": kernel,
        "shape": shape,
        "job_name": job_name,
        "num_runs": len(timings),
        "timings_ms": [t * 1000 for t in timings],
        "statistics": {
            "mean_ms": statistics.mean(timings) * 1000,
            "median_ms": statistics.median(timings) * 1000,
            "stdev_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
            "min_ms": min(timings) * 1000,
            "max_ms": max(timings) * 1000,
        },
        "config": config,
    }
    output_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def package_results(job_name, ir_dump_root, benchmark_result_path, output_dir):
    """Create tar.gz archive of IR dumps and benchmark results."""
    ir_dump_root = pathlib.Path(ir_dump_root)
    benchmark_result_path = pathlib.Path(benchmark_result_path)
    output_dir = pathlib.Path(output_dir)
    tarball_path = output_dir / f"{job_name}.tar.gz"

    with tarfile.open(tarball_path, "w:gz") as tar:
        if ir_dump_root.exists():
            tar.add(str(ir_dump_root), arcname="ir_dumps")
        if benchmark_result_path.exists():
            tar.add(str(benchmark_result_path), arcname="benchmark_result.json")

    return tarball_path


def upload_to_gcs(tarball_path, gcs_bucket, job_name):
    """Upload tarball to GCS."""
    from google.cloud import storage

    gcs_path = gcs_bucket.removeprefix("gs://").strip("/")
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    bucket_prefix = parts[1] if len(parts) > 1 else ""

    blob_path = f"{job_name}/{pathlib.Path(tarball_path).name}"
    if bucket_prefix:
        blob_path = f"{bucket_prefix}/{blob_path}"

    print(f"[benchmark] Uploading to gs://{bucket_name}/{blob_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(tarball_path))
    print(f"[benchmark] Upload complete")


_DEFAULT_IR_DUMP_ROOT = pathlib.Path("/tmp/ir_dumps")
_DEFAULT_RESULT_PATH = pathlib.Path("/tmp/benchmark_result.json")
_DEFAULT_OUTPUT_DIR = pathlib.Path("/tmp")


def main(argv=None, ir_dump_root=None, benchmark_result_path=None, output_dir=None):
    """Run the full benchmark pipeline."""
    ir_dump_root = ir_dump_root or _DEFAULT_IR_DUMP_ROOT
    benchmark_result_path = benchmark_result_path or _DEFAULT_RESULT_PATH
    output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    args = parse_args(argv)

    print(f"[benchmark] Starting benchmark for kernel: {args.kernel}")
    print(f"[benchmark] Shape: {args.shape}, Job: {args.job_name}")

    setup_ir_dump_dirs(ir_dump_root)
    setup_xla_flags(ir_dump_root)

    kernel_fn, config = import_kernel(args.kernel)

    timings = run_benchmark(
        kernel_fn, config, args.num_warmup, args.num_runs,
        chunk_size=args.chunk_size,
        ep_size=args.ep_size,
    )

    if not is_coordinator():
        print("[benchmark] Non-coordinator process, skipping dump/upload")
        return

    write_benchmark_result(
        timings, args.kernel, args.shape, args.job_name, config,
        benchmark_result_path,
    )

    tarball = package_results(
        args.job_name, ir_dump_root, benchmark_result_path, output_dir,
    )

    try:
        upload_to_gcs(tarball, args.gcs_bucket, args.job_name)
    except Exception as exc:
        print(f"[benchmark] ERROR: GCS upload failed: {exc}", file=sys.stderr)
        print(f"[benchmark] Tarball saved locally at: {tarball}", file=sys.stderr)
        raise

    print(f"[benchmark] Done!")


if __name__ == "__main__":
    main()
