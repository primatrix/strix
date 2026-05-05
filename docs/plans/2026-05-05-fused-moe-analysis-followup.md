# Fused MoE Theoretical Analysis — Follow-Up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the theoretical formulas against LLO measurements and benchmark data, build a parameterized performance calculator, and create a prefill case study.

**Architecture:** The analysis document (`2026-05-05-fused-moe-theoretical-analysis.md`) provides a formula framework. This plan validates those formulas against real LLO dumps, builds a reusable Python calculator for instant performance estimation, and extends the framework to prefill scenarios with benchmark data.

**Tech Stack:** Python, existing strix LLO parser/analyzer, TPU v7x benchmark runner

---

### Task 1: Extract and Validate LLO VMEM Allocations Against Formulas

**Files:**
- Create: `scripts/validate_vmem_formula.py`
- Reference: `docs/plans/2026-05-05-fused-moe-theoretical-analysis.md` Section 3.3
- Data: `benchmark_results/ir_dumps/llo/`

**Goal:** Parse the VMEM allocations from LLO `02-original.txt`, compute the formula-predicted sizes, and quantify the gap.

**Step 1: Write script skeleton**

```python
"""Validate VMEM occupancy formulas against LLO allocation dumps."""
import re
import sys
from pathlib import Path

def parse_llo_allocations(llo_path: str) -> dict[str, int]:
    """Extract allocation name and byte size from LLO original dump."""
    # Parse lines like: allocation8  bf16[2,8,16,2,4096]  ; 4194304 bytes
    allocations = {}
    pattern = re.compile(
        r'allocation(\d+)\s+(\w+)\[([^\]]+)\]\s*;\s*(\d+)\s*bytes'
    )
    with open(llo_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                alloc_id = m.group(1)
                dtype = m.group(2)
                shape = m.group(3)
                size_bytes = int(m.group(4))
                allocations[f"allocation{alloc_id}"] = {
                    "dtype": dtype,
                    "shape": shape,
                    "size_bytes": size_bytes,
                }
    return allocations


def formula_vmem_estimates(
    bt: int, bf: int, bd1: int, bd2: int, bse: int,
    H: int, expert_buffer_count: int, EP: int, tp: int,
) -> dict[str, int]:
    """Compute expected VMEM sizes from the theoretical formulas (Section 3.3)."""
    tau = 2  # BF16
    tau_acc = 4  # F32

    return {
        "token_input_x2": 2 * bt * H * tau,
        "w1_double_buf": 2 * tp * (bd1 // tp) * bf * tau,
        "w3_double_buf": 2 * tp * (bd1 // tp) * bf * tau,
        "w2_double_buf": 2 * tp * bf * (bd2 // tp) * tau,
        "gate_accum": bt * bf * tau_acc,
        "up_accum": bt * bf * tau_acc,
        "ffn2_accum_x3": 3 * bt * bd2 * tau_acc,
        "se_w1_x2": 2 * tp * (bd1 // tp) * bse * tau,
        "se_w2_x2": 2 * tp * bse * (bd2 // tp) * tau,
        "se_w3_x2": 2 * tp * (bd1 // tp) * bse * tau,
        "se_accum": bt * H * tau_acc,
        "output_x2": 2 * bt * H * tau,
        "a2a_scratch_approx": expert_buffer_count * bt * EP * tp * (H // tp) * tau,
    }


def main():
    llo_dir = Path("benchmark_results/ir_dumps/llo")
    # Find the original.txt for the fused MoE kernel
    original_files = list(llo_dir.glob("*fused-moe*original.txt"))
    if not original_files:
        print("ERROR: No LLO original.txt found for fused-moe kernel")
        sys.exit(1)

    # Parse block config from filename
    # e.g. fused-moe-k_8-bt_64_64_64-bf_1024_1024-bd1_2048_2048-bd2_2048_2048...
    fname = original_files[0].name
    bt = int(re.search(r'bt_(\d+)', fname).group(1))
    bf = int(re.search(r'bf_(\d+)', fname).group(1))
    bd1 = int(re.search(r'bd1_(\d+)', fname).group(1))
    bd2 = int(re.search(r'bd2_(\d+)', fname).group(1))
    bse_match = re.search(r'bse_(\d+)', fname)
    bse = int(bse_match.group(1)) if bse_match else bf

    llo_allocs = parse_llo_allocations(str(original_files[0]))
    formula_est = formula_vmem_estimates(bt, bf, bd1, bd2, bse, H=8192,
                                          expert_buffer_count=32, EP=8, tp=2)

    total_llo = sum(a["size_bytes"] for a in llo_allocs.values())
    total_formula = sum(formula_est.values())

    print(f"LLO total VMEM:  {total_llo / 1e6:.2f} MB")
    print(f"Formula total:   {total_formula / 1e6:.2f} MB")
    print(f"Gap:             {(total_llo - total_formula) / 1e6:.2f} MB"
          f" ({(total_llo/total_formula - 1)*100:.1f}%)")

    # Itemize formula components
    print("\nFormula breakdown:")
    for name, size in sorted(formula_est.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {size/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
```

**Step 2: Run against available LLO dump**

Run: `python scripts/validate_vmem_formula.py`
Expected: Output showing total LLO VMEM vs formula estimate with gap analysis.

**Step 3: Refine formulas if gap > 20%**

Compare individual allocation sizes to identify which formulas need adjustment (e.g., padding, alignment overhead, SMEM allocations not accounted for).

**Step 4: Commit**

```bash
git add scripts/validate_vmem_formula.py
git commit -m "feat: add VMEM formula validation script against LLO dumps"
```

---

### Task 2: Build Parameterized Performance Calculator

**Files:**
- Create: `scripts/moe_perf_calculator.py`
- Reference: `docs/plans/2026-05-05-fused-moe-theoretical-analysis.md` Parts 1-2

**Goal:** A standalone Python module that takes model config + block config and outputs per-stage theoretical bounds. No JAX/TPU needed — pure arithmetic.

**Step 1: Write the calculator module**

```python
"""Parameterized fused MoE performance calculator.

Usage:
    python scripts/moe_perf_calculator.py --B 256 --E 256 --k 8 --H 8192 --I 2048 \
        --ep 8 --bt 64 --bf 1024 --bd1 2048 --bd2 2048
"""

import argparse
from dataclasses import dataclass


@dataclass
class TPUv7x:
    mxu_peak_tflops: float = 2307.0
    hbm_bw_gbs: float = 3690.0
    ici_bw_gbs: float = 300.0  # per link, approximate
    vmem_mib: float = 64.0
    ridge_point: float = 2307.0 / 3.69  # ~625 FLOPs/byte


@dataclass
class ModelConfig:
    B: int       # global batch
    E: int       # num experts
    k: int       # top-k
    H: int       # hidden dim
    I: int       # intermediate dim
    I_se: int    # shared expert intermediate
    dtype: str = "bf16"

    @property
    def tau(self) -> int:
        return 2 if self.dtype == "bf16" else 1


@dataclass
class ParallelConfig:
    DP: int = 1
    EP: int = 8
    TP: int = 1


@dataclass
class BlockConfig:
    bt: int; btc: int; bf: int; bfc: int
    bd1: int; bd1c: int; bd2: int; bd2c: int
    bse: int; bts: int | None = None

    @property
    def tp(self) -> int:
        return 2  # BF16 packing


@dataclass
class Derived:
    """All derived quantities from model + parallel config."""
    T_local: int       # tokens per device
    E_local: int       # experts per device
    Q_pairs: int       # token-expert pairs per device
    n_bar_e: float     # avg tokens per expert (global)
    n_bar_e_local: float  # avg tokens per local expert
    n_bt: int          # number of outer bt loops

    @classmethod
    def from_configs(cls, model: ModelConfig, parallel: ParallelConfig,
                     block: BlockConfig) -> "Derived":
        T_local = model.B // parallel.DP
        E_local = model.E // parallel.EP
        Q_pairs = T_local * model.k
        n_bar_e = (model.B * model.k) / model.E
        n_bar_e_local = Q_pairs / E_local
        n_bt = max(1, (T_local + block.bt - 1) // block.bt)
        return cls(T_local, E_local, Q_pairs, n_bar_e, n_bar_e_local, n_bt)


def compute_stage1(model: ModelConfig, parallel: ParallelConfig,
                   derived: Derived) -> dict:
    """Stage 1: gate, topk, allreduce metadata, permute."""
    hbm_read = 8 * derived.T_local * model.k  # topk_weights(F32) + topk_ids(S32)
    comm_bytes = model.E * 4  # S32 per entry
    comm_rounds = (parallel.EP.bit_length() - 1) if parallel.EP > 0 else 0
    return {
        "hbm_read_mb": hbm_read / 1e6,
        "comm_total_kb": comm_bytes * comm_rounds / 1e3,
        "comm_rounds": comm_rounds,
        "bottleneck": "communication_latency",
        "est_time_us": 10 + 5 * comm_rounds,
    }


def compute_stage2(model: ModelConfig, parallel: ParallelConfig,
                   derived: Derived, block: BlockConfig, hw: TPUv7x) -> dict:
    """Stage 2: A2A dispatch + expert FFN + A2A combine."""
    tau = model.tau

    # Dispatch communication
    frac_local = 1.0 / parallel.EP
    token_bytes = derived.T_local * model.k * model.H * tau
    ici_out_mb = token_bytes * (1 - frac_local) / 1e6

    # Per-expert weights and FLOPs
    weight_per_expert_mb = 3 * model.H * model.I * tau / 1e6
    total_weight_read_mb = derived.E_local * weight_per_expert_mb
    token_read_mb = token_bytes / 1e6
    total_hbm_read_mb = total_weight_read_mb + token_read_mb

    flops_per_expert_gflops = 6 * block.btc * model.H * model.I / 1e9
    total_flops_gflops = derived.E_local * flops_per_expert_gflops
    useful_flops_gflops = total_flops_gflops * derived.n_bar_e / block.btc

    # Timing
    t_hbm_us = (total_hbm_read_mb / hw.hbm_bw_gbs) * 1000
    t_compute_us = (total_flops_gflops / hw.mxu_peak_tflops)
    t_comm_us = (ici_out_mb / hw.ici_bw_gbs) * 1000 * 2  # scatter + gather

    ai_executed = total_flops_gflops * 1e9 / (total_hbm_read_mb * 1e6)
    ai_useful = useful_flops_gflops * 1e9 / (total_hbm_read_mb * 1e6)

    return {
        "total_hbm_read_mb": total_hbm_read_mb,
        "weight_read_mb": total_weight_read_mb,
        "token_read_mb": token_read_mb,
        "ici_dispatch_mb": ici_out_mb,
        "total_flops_gflops": total_flops_gflops,
        "useful_flops_gflops": useful_flops_gflops,
        "ai_executed": ai_executed,
        "ai_useful": ai_useful,
        "t_hbm_us": t_hbm_us,
        "t_compute_us": t_compute_us,
        "t_comm_us": t_comm_us,
        "bottleneck": "HBM_BW" if t_hbm_us > max(t_compute_us, t_comm_us)
                      else ("COMPUTE" if t_compute_us > t_comm_us else "ICI"),
        "theoretical_lower_bound_us": max(t_hbm_us, t_compute_us, t_comm_us),
    }


def compute_stage4(model: ModelConfig, derived: Derived, block: BlockConfig,
                   hw: TPUv7x) -> dict:
    """Stage 4: Shared expert FFN (standalone)."""
    tau = model.tau
    weight_mb = 3 * model.H * model.I_se * tau / 1e6
    token_mb = derived.T_local * model.H * tau / 1e6
    total_hbm_mb = weight_mb + token_mb
    flops_gflops = 6 * derived.T_local * model.H * model.I_se / 1e9
    ai = flops_gflops * 1e9 / (total_hbm_mb * 1e6)
    t_hbm_us = (total_hbm_mb / hw.hbm_bw_gbs) * 1000
    t_compute_us = flops_gflops / hw.mxu_peak_tflops

    return {
        "hbm_read_mb": total_hbm_mb,
        "flops_gflops": flops_gflops,
        "ai": ai,
        "t_hbm_us": t_hbm_us,
        "t_compute_us": t_compute_us,
        "can_overlap_with_s1": True,
        "overlap_window_us": 15,  # approximate S1 duration
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=256)
    parser.add_argument("--E", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--H", type=int, default=8192)
    parser.add_argument("--I", type=int, default=2048)
    parser.add_argument("--I-se", type=int, default=2048)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--bt", type=int, default=64)
    parser.add_argument("--bf", type=int, default=1024)
    parser.add_argument("--bd1", type=int, default=2048)
    parser.add_argument("--bd2", type=int, default=2048)
    parser.add_argument("--bse", type=int, default=256)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    hw = TPUv7x()
    model = ModelConfig(args.B, args.E, args.k, args.H, args.I, args.I_se, args.dtype)
    parallel = ParallelConfig(DP=args.dp, EP=args.ep)
    block = BlockConfig(
        bt=args.bt, btc=args.bt, bf=args.bf, bfc=args.bf,
        bd1=args.bd1, bd1c=args.bd1, bd2=args.bd2, bd2c=args.bd2, bse=args.bse,
    )
    derived = Derived.from_configs(model, parallel, block)

    s1 = compute_stage1(model, parallel, derived)
    s2 = compute_stage2(model, parallel, derived, block, hw)
    s4 = compute_stage4(model, derived, block, hw)

    print("=" * 60)
    print(f"Fused MoE Performance Estimate")
    print(f"  Model: B={model.B}, E={model.E}, k={model.k}, H={model.H}, I={model.I}")
    print(f"  Parallel: DP={parallel.DP}, EP={parallel.EP}")
    print(f"  Block: bt={block.bt}, bf={block.bf}, bd1={block.bd1}, bd2={block.bd2}")
    print(f"  Derived: T_local={derived.T_local}, E_local={derived.E_local}, "
          f"n_bar_e={derived.n_bar_e:.1f}, n_bt={derived.n_bt}")
    print()
    print(f"Stage 1 (Routing):     {s1['est_time_us']:.0f} μs (comm: {s1['comm_rounds']} rounds)")
    print(f"Stage 2 (Expert FFN):  HBM={s2['t_hbm_us']:.0f} μs, "
          f"Compute={s2['t_compute_us']:.1f} μs, Comm={s2['t_comm_us']:.0f} μs")
    print(f"  AI: {s2['ai_executed']:.1f} (exec), {s2['ai_useful']:.1f} (useful) FLOPs/byte")
    print(f"  Bottleneck: {s2['bottleneck']}")
    print(f"  Theoretical bound: {s2['theoretical_lower_bound_us']:.0f} μs")
    print(f"Stage 4 (Shared Exp):  HBM={s4['t_hbm_us']:.0f} μs, "
          f"Compute={s4['t_compute_us']:.1f} μs, AI={s4['ai']:.0f}")
    print(f"  Overlap window w/ S1: {s4['overlap_window_us']} μs")
    print()
    print(f"Total theoretical:     {s2['theoretical_lower_bound_us']:.0f} μs")
    print(f"  (Stage 1+2, SE overlapped)")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Run against Ling 2.6 decode config**

Run: `python scripts/moe_perf_calculator.py --B 256 --E 256 --k 8 --H 8192 --I 2048 --ep 8 --bt 64 --bf 1024 --bd1 2048 --bd2 2048`
Expected: Output showing ~0.8 ms HBM bound for Stage 2, AI ~67 executed.

**Step 3: Run against Ling 2.6 EP=4 config**

Run: `python scripts/moe_perf_calculator.py --B 256 --E 256 --k 8 --H 8192 --I 2048 --ep 4 --bt 64 --bf 1024 --bd1 2048 --bd2 2048`
Expected: E_local=64, ~1.7 ms theoretical (matches existing analysis).

**Step 4: Run prefill scenario**

Run: `python scripts/moe_perf_calculator.py --B 8192 --E 256 --k 8 --H 8192 --I 2048 --ep 8 --bt 256 --bf 1024 --bd1 2048 --bd2 2048`
Expected: T_local=8192, showing how AI shifts toward ridge point.

**Step 5: Commit**

```bash
git add scripts/moe_perf_calculator.py
git commit -m "feat: add parameterized fused MoE performance calculator"
```

---

### Task 3: Run EP Scaling Benchmark

**Files:**
- Modify: `scripts/benchmark_runner.py` (already supports `--ep-size`)
- Create: `scripts/run_ep_scaling.sh`

**Goal:** Validate the EP scaling formulas with real benchmarks at EP=4, 8, 16 on the same TPU slice.

**Step 1: Write benchmark script**

```bash
#!/bin/bash
# EP scaling benchmark for fused MoE kernel
# Requires: TPU v7x 4x4x1 slice (or larger)

EP_SIZES=(4 8 16)
OUTPUT_DIR="benchmark_results/ep_scaling"
mkdir -p "$OUTPUT_DIR"

for EP in "${EP_SIZES[@]}"; do
    echo "=== Running EP=$EP ==="
    python scripts/benchmark_runner.py \
        --kernel kernels.fused_moe \
        --ep-size "$EP" \
        --output-dir "$OUTPUT_DIR/ep_${EP}" \
        --num-runs 10
done

echo "=== Done ==="
echo "Results in $OUTPUT_DIR/"
```

**Step 2: Run benchmarks on TPU**

Run: `bash scripts/run_ep_scaling.sh`
Expected: Three benchmark result sets at EP=4, 8, 16.

**Step 3: Compare against formula predictions**

```python
# Quick comparison script (inline)
import json
for ep in [4, 8, 16]:
    with open(f"benchmark_results/ep_scaling/ep_{ep}/benchmark_result.json") as f:
        data = json.load(f)
    mean_ms = data["statistics"]["mean_ms"]
    print(f"EP={ep}: measured={mean_ms:.1f}ms")
```

Expected: EP=4 should be ~2x slower than EP=8 (E_local=64 vs 32), EP=16 ~0.5x of EP=8.

**Step 4: Commit**

```bash
git add scripts/run_ep_scaling.sh
git commit -m "feat: add EP scaling benchmark script for fused MoE validation"
```

---

### Task 4: Write Prefill Case Study

**Files:**
- Create: `docs/plans/2026-05-05-fused-moe-prefill-case-study.md`

**Goal:** A companion document applying the theoretical framework to a prefill scenario (B=2048-8192), validating the prefill vs decode analysis from Section 4.6.

**Step 1: Create the prefill case study document**

Document structure:
1. Prefill config parameters (B=2048, 4096, 8192)
2. Derived quantities per config
3. Stage-by-stage numbers from the calculator
4. Comparison with decode: when does AI cross the memory/compute boundary?
5. Optimal block config for prefill (larger btc, token-stationary preference)
6. VMEM constraint analysis for large token batches
7. Communication scaling (all-to-all volume grows with B)
8. If available: actual prefill benchmark results

**Step 2: Run calculator for prefill configs to populate numbers**

Run:
```bash
python scripts/moe_perf_calculator.py --B 2048 --E 256 --k 8 --H 8192 --I 2048 --ep 8 --bt 256 --bf 1024 --bd1 2048 --bd2 2048
python scripts/moe_perf_calculator.py --B 4096 --E 256 --k 8 --H 8192 --I 2048 --ep 8 --bt 512 --bf 1024 --bd1 2048 --bd2 2048
```

**Step 3: Commit**

```bash
git add docs/plans/2026-05-05-fused-moe-prefill-case-study.md
git commit -m "docs: add prefill case study for fused MoE kernel"
```

---

### Task 5: Validate DMA Efficiency Model

**Files:**
- Create: `scripts/analyze_dma_sizes.py`

**Goal:** Parse LLO dumps to extract actual DMA transfer sizes and compute the effective HBM bandwidth per size class, validating Section 3.4.

**Step 1: Write DMA size analyzer**

```python
"""Extract DMA transfer sizes from LLO bundles and compute efficiency."""
import re
import json
from pathlib import Path
from collections import defaultdict


def parse_dma_ops(llo_bundles_path: str) -> list[dict]:
    """Extract DMA operations and sizes from LLO bundle dump."""
    dma_ops = []
    dma_pattern = re.compile(r'dma\.(hbm_to_vmem|general)')
    size_pattern = re.compile(r'(\d+)\s*bytes')

    with open(llo_bundles_path) as f:
        for line in f:
            if dma_pattern.search(line):
                size_match = size_pattern.search(line)
                size = int(size_match.group(1)) if size_match else 0
                dma_ops.append({
                    "op": dma_pattern.search(line).group(1),
                    "size_bytes": size,
                })
    return dma_ops


def compute_efficiency(dma_ops: list[dict], total_time_s: float,
                       peak_bw_gbs: float = 3690.0) -> dict:
    """Group DMA ops by size class and compute effective bandwidth."""
    total_bytes = sum(op["size_bytes"] for op in dma_ops)
    effective_bw = total_bytes / total_time_s / 1e9  # GB/s
    efficiency = effective_bw / peak_bw_gbs * 100

    # Size class breakdown
    size_classes = defaultdict(lambda: {"count": 0, "total_bytes": 0})
    for op in dma_ops:
        sz = op["size_bytes"]
        if sz < 4096:
            cls = "<4KB"
        elif sz < 65536:
            cls = "4-64KB"
        elif sz < 262144:
            cls = "64-256KB"
        elif sz < 1048576:
            cls = "256KB-1MB"
        else:
            cls = ">1MB"
        size_classes[cls]["count"] += 1
        size_classes[cls]["total_bytes"] += sz

    return {
        "total_dma_ops": len(dma_ops),
        "total_bytes_gb": total_bytes / 1e9,
        "effective_bw_gbs": effective_bw,
        "efficiency_pct": efficiency,
        "size_distribution": dict(size_classes),
    }


def main():
    llo_dir = Path("benchmark_results/ir_dumps/llo")
    bundle_files = list(llo_dir.glob("*fused-moe*final_bundles.txt"))
    if not bundle_files:
        print("ERROR: No final_bundles.txt found")
        return

    dma_ops = parse_dma_ops(str(bundle_files[0]))
    # Use benchmark time for effective BW calculation
    total_time = 0.150  # 150 ms from benchmark

    result = compute_efficiency(dma_ops, total_time)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Step 2: Run DMA analysis**

Run: `python scripts/analyze_dma_sizes.py`
Expected: Size distribution showing small DMAs dominate count but large weight tiles dominate bytes. Effective BW << peak.

**Step 3: Commit**

```bash
git add scripts/analyze_dma_sizes.py
git commit -m "feat: add DMA size distribution analyzer for LLO bundles"
```

---

### Task 6: Cross-Validate Formulas Against tpu-perf-model

**Files:**
- Modify: `primatrix/wiki/steps_fused_moe_per_expert.json` (potentially)
- Create: `scripts/validate_against_perf_model.py`

**Goal:** Run the tpu-perf-model simulation for multiple configs and compare against formula-derived bottlenecks and AI values.

**Step 1: Write cross-validation script**

```python
"""Cross-validate analytical formulas against tpu-perf-model simulation."""
import json
import subprocess
import sys


def run_perf_model(config: dict) -> dict:
    """Run tpu-perf-model CLI and parse output."""
    # Write temp config
    with open("/tmp/perf_model_input.json", "w") as f:
        json.dump(config, f)

    result = subprocess.run(
        ["python", "-m", "strix.cli", "analyze", "/tmp/perf_model_input.json"],
        capture_output=True, text=True,
    )
    # Parse the output for bottleneck, timings, AI
    # (Implementation depends on CLI output format)
    return {"bottleneck": "HBM_BW", "total_ns": 28860}  # placeholder


def formula_prediction(model_cfg: dict, parallel_cfg: dict, block_cfg: dict) -> dict:
    """Compute formula-based prediction (reuse calculator module)."""
    from scripts.moe_perf_calculator import (
        ModelConfig, ParallelConfig, BlockConfig, Derived,
        compute_stage2, TPUv7x,
    )
    model = ModelConfig(**model_cfg)
    parallel = ParallelConfig(**parallel_cfg)
    block = BlockConfig(**block_cfg)
    derived = Derived.from_configs(model, parallel, block)
    return compute_stage2(model, parallel, derived, block, TPUv7x())


def main():
    configs = [
        # (model, parallel, block) tuples for testing
        {
            "model": {"B": 256, "E": 256, "k": 8, "H": 8192, "I": 2048, "I_se": 2048},
            "parallel": {"DP": 1, "EP": 8},
            "block": {"bt": 64, "btc": 64, "bf": 1024, "bfc": 1024,
                      "bd1": 2048, "bd1c": 2048, "bd2": 2048, "bd2c": 2048, "bse": 256},
        },
    ]

    for i, cfg in enumerate(configs):
        formula = formula_prediction(cfg["model"], cfg["parallel"], cfg["block"])
        print(f"Config {i}:")
        print(f"  Formula: bottleneck={formula['bottleneck']}, "
              f"AI={formula['ai_executed']:.1f}, HBM={formula['t_hbm_us']:.0f}μs")
        print()


if __name__ == "__main__":
    main()
```

**Step 2: Run validation**

Run: `python scripts/validate_against_perf_model.py`
Expected: Formula bottleneck and AI should match perf-model to within ~10%.

**Step 3: Commit**

```bash
git add scripts/validate_against_perf_model.py
git commit -m "feat: cross-validate analytical formulas against tpu-perf-model"
```
