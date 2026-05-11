"""Overhead characterization sweep for DMA-only double-buffer expert kernel.

Runs the kernel at 8 different intermediate_size values (512..65536) within
a single process, measuring wall-clock time at each point.  Fits a linear
model  t = overhead + data / bandwidth  and reports the extracted overhead
and bandwidth parameters.

Designed to run inside the same K8s Job template used by benchmark_runner.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time


_GIB = 1024**3
_F_VALUES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def is_coordinator():
    import jax
    return jax.process_index() == 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DMA overhead characterization sweep")
    p.add_argument("--job-name", default=os.environ.get("JOB_NAME", "overhead-sweep"))
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/tmp/operator-artifact"))
    p.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", ""))
    p.add_argument("--no-ir-dump", action="store_true", default=bool(os.environ.get("NO_IR_DUMP")))
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-runs", type=int, default=20)
    p.add_argument("--bt", type=int, default=256, help="num_tokens (default 256)")
    p.add_argument("--bf", type=int, default=512, help="tile size along f-axis (default 512)")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation: test each kernel variant at f=512 and f=65536")
    return p.parse_args(argv)


def run_single_f(f: int, bt: int, bf: int, num_warmup: int, num_runs: int) -> dict:
    """Benchmark the DMA-only kernel at a single intermediate_size value."""
    # Ensure kernels package is importable (same trick as benchmark_runner.py)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    import jax
    import jax.numpy as jnp
    from kernels.double_buffer_expert import double_buffer_expert

    d = 8192
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, d), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (d, f), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (f, d), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (d, f), dtype=jnp.bfloat16)

    def run_fn():
        return double_buffer_expert(tokens, w1, w2, w3, bf=bf)

    # Warmup (triggers JIT compilation on first call)
    for _ in range(num_warmup):
        result = run_fn()
        result.block_until_ready()

    # Timed runs
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = run_fn()
        result.block_until_ready()
        timings.append(time.perf_counter() - start)

    n_tiles = f // bf
    # 3 weights: W1(d,f) + W3(d,f) + W2(f,d) = 3*d*f elements, bf16 = 2 bytes
    weight_bytes = 3 * d * f * 2
    # tokens: (bt, d) bf16
    token_bytes = bt * d * 2
    total_bytes = weight_bytes + token_bytes

    median_s = statistics.median(timings)
    min_s = min(timings)

    return {
        "intermediate_size": f,
        "n_tiles": n_tiles,
        "weight_bytes": weight_bytes,
        "token_bytes": token_bytes,
        "total_dma_bytes": total_bytes,
        "num_runs": num_runs,
        "timings_ms": [t * 1000 for t in timings],
        "statistics": {
            "mean_ms": statistics.mean(timings) * 1000,
            "median_ms": median_s * 1000,
            "stdev_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
            "min_ms": min_s * 1000,
            "max_ms": max(timings) * 1000,
        },
        "throughput": {
            "gib_per_s_median": total_bytes / median_s / _GIB,
            "gib_per_s_max": total_bytes / min_s / _GIB,
        },
    }


def linear_regression(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares: y = a*x + b.  Returns {a, b, r_squared}."""
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x * sum_x
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y * sum_xx - sum_x * sum_xy) / denom

    # R²
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"a": a, "b": b, "r_squared": r_squared}


def fit_overhead_model(records: list[dict]) -> dict:
    """Fit t_ms = overhead_ms + weight_bytes / bandwidth.

    Uses median_ms as the y-value (more robust than mean).
    x-axis: weight_bytes (tokens DMA is constant across configs, absorbed into intercept).
    """
    xs = [r["weight_bytes"] for r in records]
    ys = [r["statistics"]["median_ms"] for r in records]

    fit = linear_regression(xs, ys)

    # a = ms per byte → bandwidth = 1/a bytes/ms = 1/(a*1e-3) bytes/s
    a, b = fit["a"], fit["b"]
    bandwidth_bytes_per_s = 1.0 / (a * 1e-3) if a > 0 else float("inf")

    return {
        "overhead_ms": b,
        "slope_ms_per_byte": a,
        "bandwidth_gib_s": bandwidth_bytes_per_s / _GIB,
        "r_squared": fit["r_squared"],
        "model": "t_median_ms = overhead_ms + slope_ms_per_byte * weight_bytes",
        "data_points": len(records),
        "f_values": [r["intermediate_size"] for r in records],
        "median_ms_values": ys,
        "weight_bytes_values": xs,
    }


# ---------------------------------------------------------------------------
# Ablation mode
# ---------------------------------------------------------------------------

_ABLATION_F_VALUES = [512, 65536]


def run_variant_at_f(variant_fn, f: int, bt: int, bf: int,
                     num_warmup: int, num_runs: int) -> dict:
    """Benchmark a single kernel variant at a single f value."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    import jax
    import jax.numpy as jnp

    d = 8192
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, d), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (d, f), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (f, d), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (d, f), dtype=jnp.bfloat16)

    def run_fn():
        return variant_fn(tokens, w1, w2, w3, bf=bf)

    for _ in range(num_warmup):
        result = run_fn()
        result.block_until_ready()

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = run_fn()
        result.block_until_ready()
        timings.append(time.perf_counter() - start)

    weight_bytes = 3 * d * f * 2
    token_bytes = bt * d * 2
    total_bytes = weight_bytes + token_bytes
    median_s = statistics.median(timings)

    return {
        "intermediate_size": f,
        "weight_bytes": weight_bytes,
        "total_dma_bytes": total_bytes,
        "statistics": {
            "mean_ms": statistics.mean(timings) * 1000,
            "median_ms": median_s * 1000,
            "stdev_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
            "min_ms": min(timings) * 1000,
            "max_ms": max(timings) * 1000,
        },
    }


def run_ablation(args):
    """Run all ablation variants and extract overhead for each."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    from kernels.double_buffer_expert_ablation import VARIANTS

    output_dir = pathlib.Path(args.output_dir)
    metrics_dir = output_dir / "rank-0" / "benchmark"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] Starting overhead ablation study")
    print(f"[ablation] bt={args.bt}, bf={args.bf}")
    print(f"[ablation] f_values={_ABLATION_F_VALUES}")
    print(f"[ablation] variants: {list(VARIANTS.keys())}")
    print(f"[ablation] num_warmup={args.num_warmup}, num_runs={args.num_runs}")
    print()

    results = {}
    for name, variant_fn in VARIANTS.items():
        print(f"[ablation] ── variant: {name}")
        records = []
        for f in _ABLATION_F_VALUES:
            weight_mib = 3 * 8192 * f * 2 / (1024**2)
            print(f"[ablation]    f={f:>5d}  weight_dma={weight_mib:.0f} MiB ... ", end="", flush=True)
            rec = run_variant_at_f(
                variant_fn, f, args.bt, args.bf,
                args.num_warmup, args.num_runs,
            )
            records.append(rec)
            med = rec["statistics"]["median_ms"]
            print(f"median={med:.3f} ms")

        # 2-point linear interpolation to extract overhead
        xs = [r["weight_bytes"] for r in records]
        ys = [r["statistics"]["median_ms"] for r in records]
        if xs[1] != xs[0]:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            overhead = ys[0] - slope * xs[0]
            bw = 1.0 / (slope * 1e-3) / _GIB if slope > 0 else float("inf")
        else:
            overhead = ys[0]
            slope = 0.0
            bw = float("inf")

        results[name] = {
            "overhead_ms": overhead,
            "slope_ms_per_byte": slope,
            "bandwidth_gib_s": bw,
            "records": records,
        }
        print(f"[ablation]    → overhead={overhead:.4f} ms  bandwidth={bw:.0f} GiB/s")
        print()

    # Summary table
    print("[ablation] ══════════════════════════════════════════════")
    print("[ablation] Variant              overhead_ms   Δ vs baseline")
    print("[ablation] ──────────────────────────────────────────────")
    baseline_oh = results["baseline"]["overhead_ms"]
    for name, r in results.items():
        delta = r["overhead_ms"] - baseline_oh
        delta_str = f"{delta:+.4f} ms" if name != "baseline" else "  (ref)"
        print(f"[ablation] {name:<22s} {r['overhead_ms']:.4f} ms   {delta_str}")
    print("[ablation] ══════════════════════════════════════════════")

    if not is_coordinator():
        return

    # Write results
    ablation_path = metrics_dir / "ablation_results.json"
    out = {
        "config": {"bt": args.bt, "bf": args.bf, "f_values": _ABLATION_F_VALUES,
                   "num_warmup": args.num_warmup, "num_runs": args.num_runs},
        "variants": {
            name: {
                "overhead_ms": r["overhead_ms"],
                "slope_ms_per_byte": r["slope_ms_per_byte"],
                "bandwidth_gib_s": r["bandwidth_gib_s"],
                "delta_vs_baseline_ms": r["overhead_ms"] - baseline_oh,
                "f512_median_ms": r["records"][0]["statistics"]["median_ms"],
                "f65536_median_ms": r["records"][1]["statistics"]["median_ms"],
            }
            for name, r in results.items()
        },
    }
    with ablation_path.open("w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"[ablation] Results written to {ablation_path}")

    # Write manifest
    manifest = {
        "schema_version": 2,
        "workflow": "overhead-ablation",
        "operator_name": "kernels.double_buffer_expert",
        "run_id": args.job_name,
        "mode": "ablation",
        "config": out["config"],
    }
    with (output_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    _write_env_info(output_dir / "rank-0" / "profiling")
    _upload_if_configured(output_dir, args)


def main(argv=None):
    args = parse_args(argv)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Suppress IR dumps; disable bounds check to reduce launch overhead
    if args.no_ir_dump:
        os.environ["LIBTPU_INIT_ARGS"] = (
            "--xla_tpu_scoped_vmem_limit_kib=65536"
            " --xla_jf_bounds_check=false"
        )

    if args.ablation:
        run_ablation(args)
        return

    metrics_dir = output_dir / "rank-0" / "benchmark"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.jsonl"
    fit_path = metrics_dir / "overhead_fit.json"

    print(f"[overhead_sweep] Starting DMA overhead sweep")
    print(f"[overhead_sweep] bt={args.bt}, bf={args.bf}")
    print(f"[overhead_sweep] f_values={_F_VALUES}")
    print(f"[overhead_sweep] num_warmup={args.num_warmup}, num_runs={args.num_runs}")

    records = []
    for f in _F_VALUES:
        n_tiles = f // args.bf
        weight_mib = 3 * 8192 * f * 2 / (1024**2)
        print(f"[overhead_sweep] f={f:>5d}  n_tiles={n_tiles:>3d}  weight_dma={weight_mib:.0f} MiB ... ", end="", flush=True)

        rec = run_single_f(f, args.bt, args.bf, args.num_warmup, args.num_runs)
        records.append(rec)

        med = rec["statistics"]["median_ms"]
        tput = rec["throughput"]["gib_per_s_median"]
        print(f"median={med:.3f} ms  throughput={tput:.0f} GiB/s")

    # Fit linear model
    fit = fit_overhead_model(records)
    print()
    print(f"[overhead_sweep] ── Linear fit: t = {fit['overhead_ms']:.4f} ms + weight_bytes / {fit['bandwidth_gib_s']:.0f} GiB/s")
    print(f"[overhead_sweep] ── Overhead:   {fit['overhead_ms']:.4f} ms")
    print(f"[overhead_sweep] ── Bandwidth:  {fit['bandwidth_gib_s']:.1f} GiB/s")
    print(f"[overhead_sweep] ── R²:         {fit['r_squared']:.6f}")

    if not is_coordinator():
        print("[overhead_sweep] Non-coordinator, skipping file output")
        return

    # Write profiling env info
    profiling_dir = output_dir / "rank-0" / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    _write_env_info(profiling_dir)

    # Write metrics JSONL
    with metrics_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # Write fit results
    with fit_path.open("w") as fh:
        json.dump(fit, fh, indent=2)
        fh.write("\n")

    # Write manifest
    manifest = {
        "schema_version": 2,
        "workflow": "overhead-characterization",
        "operator_name": "kernels.double_buffer_expert",
        "run_id": args.job_name,
        "mode": "overhead_sweep",
        "config": {
            "bt": args.bt,
            "bf": args.bf,
            "d": 8192,
            "f_values": _F_VALUES,
            "num_warmup": args.num_warmup,
            "num_runs": args.num_runs,
        },
    }
    with (output_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"[overhead_sweep] Results written to {metrics_path}")
    print(f"[overhead_sweep] Fit written to {fit_path}")

    # Upload to GCS
    _upload_if_configured(output_dir, args)


def _write_env_info(profiling_dir: pathlib.Path):
    env_path = profiling_dir / "env.txt"
    env_vars = {
        k: v for k, v in sorted(os.environ.items())
        if k.startswith(("LIBTPU", "XLA_", "JAX_", "TF_XLA"))
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


def _upload_if_configured(output_dir: pathlib.Path, args):
    if not args.gcs_bucket:
        return
    import tarfile
    tarball_path = output_dir / f"{args.job_name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(str(output_dir), arcname=args.job_name)

    from google.cloud import storage
    gcs_path = args.gcs_bucket.removeprefix("gs://").strip("/")
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    bucket_prefix = parts[1] if len(parts) > 1 else ""
    blob_path = f"{args.job_name}/{tarball_path.name}"
    if bucket_prefix:
        blob_path = f"{bucket_prefix}/{blob_path}"

    print(f"[overhead_sweep] Uploading to gs://{bucket_name}/{blob_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(tarball_path))
    print(f"[overhead_sweep] Upload complete")


if __name__ == "__main__":
    main()
