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
import time

IR_DUMP_SUBDIRS = ("hlo", "llo", "mosaic")
_GIB = 1024 ** 3
_VMEM_LIMIT_KIB = 64 * 1024  # 64 MiB
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float32": 4,
}


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
    p.add_argument(
        "--bf",
        type=int,
        default=_int_or_none(os.environ.get("BF")),
        help="Intermediate dimension block size (overrides kernel default)",
    )
    p.add_argument(
        "--bd",
        type=int,
        default=_int_or_none(os.environ.get("BD")),
        help="Hidden dimension block size (overrides kernel default)",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR") or os.environ.get("ARTIFACT_LOCAL_DIR") or "/tmp/operator-artifact",
        help="Operator-optimization artifact root directory",
    )
    p.add_argument(
        "--no-ir-dump",
        action="store_true",
        default=os.environ.get("NO_IR_DUMP", "").lower() in ("1", "true", "yes"),
        help="Disable HLO/LLO/Mosaic IR dump (default: enabled)",
    )
    p.add_argument(
        "--sweep",
        default=os.environ.get("SWEEP"),
        help="Comma-separated bf:bd pairs, e.g. '2048:1024,1024:512'",
    )
    p.add_argument(
        "--total-bytes",
        type=int,
        default=_int_or_none(os.environ.get("TOTAL_BYTES")) or 64 * 1024 * 1024,
        help="Target total DMA bytes per sweep config (default: 64 MiB)",
    )

    args = p.parse_args(argv)

    if args.kernel is None:
        p.error("--kernel is required (or set KERNEL_MODULE env var)")

    if args.sweep and (args.bf is not None or args.bd is not None):
        p.error("--sweep is mutually exclusive with --bf/--bd")
    if args.total_bytes <= 0:
        p.error("--total-bytes must be positive")

    return args


def _int_or_none(val):
    """Convert string to int, returning None for empty/missing values."""
    if val is None or val == "":
        return None
    return int(val)


def parse_sweep(spec: str, total_bytes: int, dtype_bytes: int) -> list[dict]:
    """Parse --sweep string into a validated list of config dicts.

    Spec grammar: "bf:bd,bf:bd,...". For each pair compute
    num_loads = total_bytes // (bf * bd * dtype_bytes), enforce TPU tile
    alignment (bd % 8 == 0, bf % 128 == 0), and reject non-integral
    divisions. Collects all errors before raising so the user sees every
    offending pair at once.
    """
    if not spec or not spec.strip():
        raise SystemExit("--sweep: no configs specified")

    errors: list[str] = []
    configs: list[dict] = []

    for idx, raw in enumerate(spec.split(",")):
        raw = raw.strip()
        parts = raw.split(":")
        if len(parts) != 2:
            errors.append(f"sweep[{idx}]: invalid pair '{raw}' (expected 'bf:bd')")
            continue
        try:
            bf = int(parts[0])
            bd = int(parts[1])
        except ValueError:
            errors.append(f"sweep[{idx}]: invalid pair '{raw}' (bf and bd must be integers)")
            continue

        if bf <= 0 or bd <= 0:
            errors.append(
                f"sweep[{idx}] (bf={bf}, bd={bd}): bf and bd must be positive integers"
            )
            continue

        pair_errors: list[str] = []
        if bf % 128 != 0:
            pair_errors.append(
                f"sweep[{idx}] (bf={bf}, bd={bd}): bf must be divisible by 128 (TPU lane alignment)"
            )
        if bd % 8 != 0:
            pair_errors.append(
                f"sweep[{idx}] (bf={bf}, bd={bd}): bd must be divisible by 8 (TPU sublane alignment for bf16)"
            )

        tile_bytes = bf * bd * dtype_bytes
        if total_bytes % tile_bytes != 0:
            pair_errors.append(
                f"sweep[{idx}] (bf={bf}, bd={bd}): total_bytes={total_bytes} "
                f"not divisible by tile_bytes={tile_bytes}; adjust --total-bytes or tile dims"
            )
        num_loads = total_bytes // tile_bytes
        if num_loads < 1:
            pair_errors.append(
                f"sweep[{idx}] (bf={bf}, bd={bd}): derived num_loads=0; "
                f"tile too large for total_bytes={total_bytes}"
            )

        if pair_errors:
            errors.extend(pair_errors)
            continue

        configs.append({
            "bf": bf,
            "bd": bd,
            "num_loads": num_loads,
            "tile_bytes": tile_bytes,
        })
        if bf != bd and bd % 128 == 0 and bf % 8 == 0:
            configs.append({
                "bf": bd,
                "bd": bf,
                "num_loads": num_loads,
                "tile_bytes": tile_bytes,
            })

    if errors:
        raise SystemExit("\n".join(errors))
    return configs


def setup_ir_dump_dirs(root):
    """Create IR dump directories under *root*."""
    root = pathlib.Path(root)
    for subdir in IR_DUMP_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def setup_xla_flags(ir_dump_root):
    """Unconditionally set XLA_FLAGS and LIBTPU_INIT_ARGS.

    Python is the sole source of these flags within the runner; the Job
    YAML no longer predefines them. Must be called before JAX or libtpu
    is imported — these libraries snapshot XLA_FLAGS / LIBTPU_INIT_ARGS
    at init time, so later mutations are ignored.
    """
    ir_dump_root = pathlib.Path(ir_dump_root)

    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_hlo_as_text --xla_dump_to={ir_dump_root / 'hlo'}"
    )

    os.environ["LIBTPU_INIT_ARGS"] = " ".join([
        "--xla_enable_custom_call_region_trace=true",
        "--xla_xprof_register_llo_debug_info=true",
        f"--xla_jf_dump_to={ir_dump_root / 'llo'}",
        "--xla_jf_dump_hlo_text=true",
        "--xla_jf_dump_llo_text=true",
        "--xla_jf_emit_annotations=true",
        f"--xla_mosaic_dump_to={ir_dump_root / 'mosaic'}",
        "--xla_mosaic_enable_llo_source_annotations=true",
        f"--xla_tpu_scoped_vmem_limit_kib={_VMEM_LIMIT_KIB}",
    ])


def import_kernel(module_path):
    """Dynamically import a kernel module and return (kernel_fn, config)."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)
    mod = importlib.import_module(module_path)
    return mod.kernel_fn, mod.config


def check_kernel_compat(kernel_fn, *, module_name: str) -> None:
    """Verify kernel_fn accepts the kwargs sweep mode will pass."""
    import inspect

    sig = inspect.signature(kernel_fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return
    required = {"bf", "bd", "num_loads"}
    missing = required - set(params.keys())
    if missing:
        raise SystemExit(
            f"kernel {module_name}.kernel_fn does not accept {sorted(missing)}; "
            f"sweep mode requires these kwargs or **kwargs"
        )


def run_benchmark(kernel_fn, config, num_warmup, num_runs, chunk_size=None, ep_size=None, bf=None, bd=None):
    """Execute kernel benchmark and return list of timing values (seconds)."""
    kwargs = dict(config.get("default_shape", {}))
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    kwargs["ep_size"] = ep_size if ep_size is not None else config.get("ep_size", 4)
    if bf is not None:
        kwargs["bf"] = bf
    if bd is not None:
        kwargs["bd"] = bd

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


def write_sweep_summary(records, output_path, *, kernel: str, job_name: str, total_bytes: int):
    """Write sweep_summary.json aggregating ok-status records, sorted by throughput desc."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = [r for r in records if r["status"] == "ok"]
    failed_count = sum(1 for r in records if r["status"] == "failed")

    rows = [
        {
            "bf": r["config"]["bf"],
            "bd": r["config"]["bd"],
            "num_loads": r["config"]["num_loads"],
            "tile_bytes": r["derived"]["tile_bytes"],
            "median_ms": r["statistics"]["median_ms"],
            "stdev_ms": r["statistics"]["stdev_ms"],
            "gib_per_s_median": r["throughput"]["gib_per_s_median"],
        }
        for r in ok
    ]
    rows.sort(key=lambda row: row["gib_per_s_median"], reverse=True)

    summary = {
        "schema_version": 1,
        "job_name": job_name,
        "kernel": kernel,
        "total_bytes": total_bytes,
        "num_configs": len(records),
        "failed_count": failed_count,
        "rows": rows,
        "sorted_by": "gib_per_s_median_desc",
    }
    output_path.write_text(json.dumps(summary, indent=2, default=str))


def build_sweep_record(
    *,
    kernel: str,
    shape: str,
    job_name: str,
    config_index: int,
    cfg: dict,
    total_bytes: int,
    dtype: str,
    timings: list[float] | None,
    status: str,
    error: str | None,
) -> dict:
    """Construct one JSONL record for a sweep config (status ok or failed)."""
    record = {
        "schema_version": 2,
        "kernel": kernel,
        "shape": shape,
        "job_name": job_name,
        "config_index": config_index,
        "config": {"bf": cfg["bf"], "bd": cfg["bd"], "num_loads": cfg["num_loads"]},
        "derived": {
            "tile_bytes": cfg["tile_bytes"],
            "total_bytes": total_bytes,
            "dtype": dtype,
        },
        "status": status,
        "num_runs": len(timings) if timings else 0,
        "timings_ms": [t * 1000 for t in timings] if timings else None,
        "statistics": None,
        "throughput": None,
        "error": error,
    }
    if timings:
        median_s = statistics.median(timings)
        min_s = min(timings)
        record["statistics"] = {
            "mean_ms": statistics.mean(timings) * 1000,
            "median_ms": median_s * 1000,
            "stdev_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
            "min_ms": min_s * 1000,
            "max_ms": max(timings) * 1000,
        }
        record["throughput"] = {
            "gib_per_s_median": total_bytes / median_s / _GIB,
            "gib_per_s_max": total_bytes / min_s / _GIB,
        }
    return record


def run_sweep(
    kernel_fn,
    config: dict,
    sweep: list[dict],
    *,
    num_warmup: int,
    num_runs: int,
    total_bytes: int,
    dtype: str,
    kernel: str,
    shape: str,
    job_name: str,
):
    """Execute each sweep config in sequence, yielding one JSONL record per config.

    A failing config (compile error, runtime exception) does not abort
    the rest of the sweep; its record is emitted with status='failed'.
    """
    for idx, cfg in enumerate(sweep):
        kwargs = dict(config.get("default_shape", {}))
        kwargs["bf"] = cfg["bf"]
        kwargs["bd"] = cfg["bd"]
        kwargs["num_loads"] = cfg["num_loads"]
        try:
            run_fn = kernel_fn(**kwargs)

            for _ in range(num_warmup):
                result = run_fn()
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()

            timings = []
            for _ in range(num_runs):
                start = time.perf_counter()
                result = run_fn()
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()
                timings.append(time.perf_counter() - start)

            record = build_sweep_record(
                kernel=kernel, shape=shape, job_name=job_name,
                config_index=idx, cfg=cfg, total_bytes=total_bytes, dtype=dtype,
                timings=timings, status="ok", error=None,
            )
        except Exception as e:  # noqa: BLE001
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[benchmark] sweep[{idx}] (bf={cfg['bf']}, bd={cfg['bd']}) FAILED: {err_msg}", file=sys.stderr)
            record = build_sweep_record(
                kernel=kernel, shape=shape, job_name=job_name,
                config_index=idx, cfg=cfg, total_bytes=total_bytes, dtype=dtype,
                timings=None, status="failed", error=err_msg,
            )
        yield record


def _write_artifact_manifest(output_dir, args, config, sweep=None):
    """Write manifest.json to the artifact root (operator-optimization contract).

    sweep=None → single-config mode (bf/bd are scalars, no 'sweep' key).
    sweep=[{bf, bd, num_loads, tile_bytes}, ...] → sweep mode (bf/bd null).
    """
    import jax

    is_sweep = sweep is not None
    manifest = {
        "schema_version": 2,
        "workflow": "operator-optimization",
        "operator_family": "dma",
        "operator_name": args.kernel,
        "run_id": args.job_name,
        "shape": args.shape,
        "mode": "sweep" if is_sweep else "single",
        "bf": None if is_sweep else args.bf,
        "bd": None if is_sweep else args.bd,
        "hardware": {
            "device_type": os.environ.get("FALCON_DEVICE_TYPE", ""),
            "device_count": jax.device_count(),
            "device_topo": os.environ.get("FALCON_DEVICE_TOPO", ""),
        },
        "dimensions": {},
    }
    if is_sweep:
        manifest["sweep"] = {
            "total_bytes": args.total_bytes,
            "num_configs": len(sweep),
            "entries": [
                {"bf": c["bf"], "bd": c["bd"], "num_loads": c["num_loads"]}
                for c in sweep
            ],
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def _write_profiling_meta(output_dir):
    """Write profiling/env.txt and profiling/python-packages.txt to artifact root."""
    profiling_dir = output_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)

    env_path = profiling_dir / "env.txt"
    env_vars = {
        k: v for k, v in sorted(os.environ.items())
        if k.startswith(("LIBTPU", "XLA_", "JAX_", "FALCON_", "TF_XLA", "SGLANG_JAX"))
    }
    with open(env_path, "w") as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

    import importlib.metadata as md
    pkgs_path = profiling_dir / "python-packages.txt"
    with open(pkgs_path, "w") as f:
        for pkg in ("jax", "jaxlib", "libtpu"):
            try:
                f.write(f"{pkg}={md.version(pkg)}\n")
            except md.PackageNotFoundError:
                f.write(f"{pkg}=MISSING\n")


def _upload_if_configured(output_dir, args):
    """Upload artifacts to GCS if --gcs-bucket is explicitly provided.

    When running under Falcon, artifacts are written to $ARTIFACT_LOCAL_DIR and
    Falcon handles collection. Only upload directly when GCS_BUCKET is set and
    differs from the artifact path.
    """
    gcs_bucket = args.gcs_bucket
    if not gcs_bucket:
        return

    import tarfile
    tarball_path = output_dir / f"{args.job_name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(str(output_dir), arcname=args.job_name)

    from google.cloud import storage
    gcs_path = gcs_bucket.removeprefix("gs://").strip("/")
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    bucket_prefix = parts[1] if len(parts) > 1 else ""
    blob_path = f"{args.job_name}/{tarball_path.name}"
    if bucket_prefix:
        blob_path = f"{bucket_prefix}/{blob_path}"

    print(f"[benchmark] Uploading to gs://{bucket_name}/{blob_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(tarball_path))
    print(f"[benchmark] Upload complete")


def main(argv=None, ir_dump_root=None, benchmark_result_path=None, output_dir=None):
    """Run the full benchmark pipeline."""
    args = parse_args(argv)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_dump_root = output_dir / "rank-0" / "compiler"
    benchmark_result_path = output_dir / "rank-0" / "benchmark" / "metrics.jsonl"
    sweep_summary_path = output_dir / "rank-0" / "benchmark" / "sweep_summary.json"

    print(f"[benchmark] Starting benchmark for kernel: {args.kernel}")
    print(f"[benchmark] Shape: {args.shape}, Job: {args.job_name}")
    if args.sweep:
        print(f"[benchmark] Sweep: {args.sweep} (total_bytes={args.total_bytes})")
    else:
        print(f"[benchmark] bf={args.bf}, bd={args.bd}")
    print(f"[benchmark] Output dir: {output_dir}")

    if not args.no_ir_dump:
        setup_ir_dump_dirs(ir_dump_root)
        setup_xla_flags(ir_dump_root)
    else:
        print("[benchmark] IR dump disabled (--no-ir-dump)")
        os.environ["LIBTPU_INIT_ARGS"] = (
            f"--xla_tpu_scoped_vmem_limit_kib={_VMEM_LIMIT_KIB}"
        )

    kernel_fn, config = import_kernel(args.kernel)

    if args.sweep:
        check_kernel_compat(kernel_fn, module_name=args.kernel)
        dtype = str(config.get("weight_dtype", "bfloat16"))
        dtype_bytes = _DTYPE_BYTES.get(dtype, 2)
        sweep_configs = parse_sweep(args.sweep, args.total_bytes, dtype_bytes)

        records = list(run_sweep(
            kernel_fn, config, sweep_configs,
            num_warmup=args.num_warmup, num_runs=args.num_runs,
            total_bytes=args.total_bytes, dtype=dtype,
            kernel=args.kernel, shape=args.shape, job_name=args.job_name,
        ))

        if not is_coordinator():
            print("[benchmark] Non-coordinator process, skipping dump/upload")
            return

        benchmark_result_path.parent.mkdir(parents=True, exist_ok=True)
        with benchmark_result_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")

        write_sweep_summary(
            records, sweep_summary_path,
            kernel=args.kernel, job_name=args.job_name, total_bytes=args.total_bytes,
        )

        _write_artifact_manifest(output_dir, args, config, sweep=sweep_configs)
        _write_profiling_meta(output_dir)
        _upload_if_configured(output_dir, args)

        if all(r["status"] == "failed" for r in records):
            print("[benchmark] All sweep configs failed — exiting 1")
            raise SystemExit(1)

        print(f"[benchmark] Done!")
        return

    timings = run_benchmark(
        kernel_fn, config, args.num_warmup, args.num_runs,
        chunk_size=args.chunk_size,
        ep_size=args.ep_size,
        bf=args.bf,
        bd=args.bd,
    )

    if not is_coordinator():
        print("[benchmark] Non-coordinator process, skipping dump/upload")
        return

    write_benchmark_result(
        timings, args.kernel, args.shape, args.job_name, config,
        benchmark_result_path,
    )

    _write_artifact_manifest(output_dir, args, config)

    if is_coordinator():
        _write_profiling_meta(output_dir)

    _upload_if_configured(output_dir, args)

    print(f"[benchmark] Done!")


if __name__ == "__main__":
    main()
