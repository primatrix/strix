#!/usr/bin/env python3
"""Run multi-expert pipeline correctness test on TPU.

Usage: python scripts/test_multi_expert_correctness.py [num_experts] [bt]

When invoked via run_benchmark.sh --runner-script, extra flags like
--no-ir-dump may be appended. This script ignores them.
"""
import os
import runpy
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

positional = [a for a in sys.argv[1:] if not a.startswith("-")]
num_experts = positional[0] if len(positional) > 0 else "4"
bt = positional[1] if len(positional) > 1 else "256"

sys.argv = ["kernels/multi_expert_pipeline.py", num_experts, bt]
runpy.run_module("kernels.multi_expert_pipeline", run_name="__main__")
