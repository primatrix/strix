#!/usr/bin/env python3
"""Run multi-expert pipeline correctness test on TPU.

Usage: python scripts/test_multi_expert_correctness.py [num_experts] [bt]
"""
import runpy
import sys

sys.argv = [
    "kernels/multi_expert_pipeline.py",
    sys.argv[1] if len(sys.argv) > 1 else "4",
    sys.argv[2] if len(sys.argv) > 2 else "256",
]
runpy.run_module("kernels.multi_expert_pipeline", run_name="__main__")
