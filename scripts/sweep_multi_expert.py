#!/usr/bin/env python3
"""Sweep num_experts for multi-expert pipeline kernel."""
import os
import sys
import time

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import jax
import jax.numpy as jnp
from kernels.multi_expert_pipeline import multi_expert_ffn

EXPERT_COUNTS = [1, 4, 8, 16, 32, 64, 256]
BT = 256
BF = 256
D = 8192
F = 2048
NUM_WARMUP = 3
NUM_RUNS = 10


def benchmark_experts(num_experts):
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_experts, BT, D), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (num_experts, D, F), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (num_experts, F, D), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (num_experts, D, F), dtype=jnp.bfloat16)

    def run():
        return multi_expert_ffn(tokens, w1, w2, w3, bf=BF, num_experts=num_experts)

    for _ in range(NUM_WARMUP):
        result = run()
        result.block_until_ready()

    timings = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        result = run()
        result.block_until_ready()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    return timings


print(f"{'experts':>8} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10} {'per_expert_us':>14}")
print("-" * 72)

for ne in EXPERT_COUNTS:
    timings = benchmark_experts(ne)
    timings.sort()
    median = timings[len(timings) // 2]
    mean = sum(timings) / len(timings)
    mn = min(timings)
    mx = max(timings)
    per_expert = median / ne * 1000
    print(f"{ne:>8} {median:>10.4f} {mean:>10.4f} {mn:>10.4f} {mx:>10.4f} {per_expert:>14.2f}")

print("\nDone.")
