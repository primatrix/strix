#!/usr/bin/env python3
"""Sweep num_experts with device-level timing via JAX profiler trace.

Like sweep_multi_expert.py but uses jax.profiler.trace to extract actual
TPU device durations (device_duration_ps) instead of wall-clock timing.
This eliminates Python dispatch overhead from the measurements.

Usage: python scripts/sweep_multi_expert_device.py [--no-ir-dump]
"""
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Filter out flags (e.g. --no-ir-dump from run_benchmark.sh)
positional = [a for a in sys.argv[1:] if not a.startswith("-")]

import statistics

import jax
import jax.numpy as jnp
from kernels.multi_expert_pipeline import kernel_fn
from scripts.benchmark_runner import _timed_runs

EXPERT_COUNTS = [1, 4, 8, 16, 32, 64, 256]
NUM_WARMUP = 3
NUM_RUNS = 10


def benchmark_experts(num_experts):
    run = kernel_fn(num_experts=num_experts)

    for _ in range(NUM_WARMUP):
        result = run()
        result.block_until_ready()

    durations_s = _timed_runs(run, NUM_RUNS)
    return [d * 1000 for d in durations_s]  # convert to ms


print(f"{'experts':>8} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} "
      f"{'max_ms':>10} {'stdev_ms':>10} {'per_expert_us':>14}")
print("-" * 84)

for ne in EXPERT_COUNTS:
    timings = benchmark_experts(ne)
    med = statistics.median(timings)
    mean = statistics.mean(timings)
    mn = min(timings)
    mx = max(timings)
    sd = statistics.stdev(timings) if len(timings) > 1 else 0.0
    per_expert = med / ne * 1000
    print(f"{ne:>8} {med:>10.4f} {mean:>10.4f} {mn:>10.4f} "
          f"{mx:>10.4f} {sd:>10.4f} {per_expert:>14.2f}")

print("\nDone.")
