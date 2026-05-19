#!/usr/bin/env python3
"""Sweep num_experts with device-level timing via JAX profiler trace.

Supports configurable shapes for different model profiles.

Usage:
  python scripts/sweep_multi_expert_device.py                    # Ling 2.6 defaults
  python scripts/sweep_multi_expert_device.py --profile mimo-v2      # MiMo V2 bf16 approx
  python scripts/sweep_multi_expert_device.py --profile mimo-v2-fp8  # MiMo V2 native FP8
  python scripts/sweep_multi_expert_device.py --hidden-size 6144 --intermediate-size 1024
"""
import argparse
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import statistics

import importlib

import jax
import jax.numpy as jnp
from scripts.benchmark_runner import _timed_runs

NUM_WARMUP = 3
NUM_RUNS = 10

PROFILES = {
    "ling2.6": dict(
        kernel="kernels.multi_expert_pipeline",
        hidden_size=8192, intermediate_size=2048, bf=256,
        experts=[1, 4, 8, 16, 32, 64, 256],
    ),
    "mimo-v2": dict(
        # FP8 approx: halve intermediate_size so bf16 DMA volume matches FP8.
        # Real MiMo: d=6144, f=2048, fp8 weights.
        # bf16 approx: d=6144, f=1024 (same tile count & per-tile bytes as
        #   fp8 f=2048 bf=512 when using bf16 bf=256).
        kernel="kernels.multi_expert_pipeline",
        hidden_size=6144, intermediate_size=1024, bf=256,
        experts=[1, 4, 8, 12, 16, 24, 48],
    ),
    "mimo-v2-fp8": dict(
        # Native FP8 e4m3fn weights, quant_block_k=256, bf=512.
        kernel="kernels.multi_expert_pipeline_fp8",
        hidden_size=6144, intermediate_size=2048, bf=512,
        experts=[4, 8, 12, 24, 36, 48],
    ),
}


def main():
    p = argparse.ArgumentParser(description="Sweep num_experts with device-level timing")
    p.add_argument("--profile", choices=list(PROFILES))
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--intermediate-size", type=int)
    p.add_argument("--bf", type=int)
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--experts", type=str, help="Comma-separated expert counts")
    p.add_argument("--no-ir-dump", action="store_true")
    args = p.parse_args()

    defaults = PROFILES.get(args.profile, PROFILES["ling2.6"])
    kernel_mod = importlib.import_module(defaults["kernel"])
    kernel_fn = kernel_mod.kernel_fn
    d = args.hidden_size or defaults["hidden_size"]
    f = args.intermediate_size or defaults["intermediate_size"]
    bf = args.bf or defaults["bf"]
    bt = args.num_tokens
    expert_counts = (
        [int(x) for x in args.experts.split(",")]
        if args.experts else defaults["experts"]
    )

    print(f"Config: d={d}, f={f}, bf={bf}, bt={bt}")
    print(f"{'experts':>8} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} "
          f"{'max_ms':>10} {'stdev_ms':>10} {'per_expert_us':>14}")
    print("-" * 84)

    for ne in expert_counts:
        run = kernel_fn(
            num_experts=ne, num_tokens=bt,
            hidden_size=d, intermediate_size=f, bf=bf,
        )
        for _ in range(NUM_WARMUP):
            run().block_until_ready()
        durations_s = _timed_runs(run, NUM_RUNS)
        timings = [t * 1000 for t in durations_s]
        med = statistics.median(timings)
        mean = statistics.mean(timings)
        mn, mx = min(timings), max(timings)
        sd = statistics.stdev(timings) if len(timings) > 1 else 0.0
        per_expert = med / ne * 1000
        print(f"{ne:>8} {med:>10.4f} {mean:>10.4f} {mn:>10.4f} "
              f"{mx:>10.4f} {sd:>10.4f} {per_expert:>14.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
