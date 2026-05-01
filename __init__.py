"""
strix
=====

A small static profiler and timeline simulator for TPU Mosaic/Pallas LLO
dumps, implemented according to the design docs in
`design_doc/tpu_pallas_kernel_static_profiler&simulator.md` and
`design_doc/implementation_detail.md`.

The main public entrypoint for now is the CLI:

    python -m strix.cli path/to/post-finalize-llo.txt

This will:
  * parse the LLO into a lightweight Instruction tree,
  * run a dual-track (VPU + DMA) simulator,
  * analyse stalls and bottlenecks,
  * print a console summary and optionally emit a Chrome trace.json.
"""

try:
    from .domain import Instruction, PerformanceMetrics
    from .hardware import HardwareSpec
except ImportError:
    # Pytest discovers root __init__.py outside of the strix package context,
    # causing relative imports to fail.  The names remain available via the
    # normal ``import strix`` path.
    pass

__all__ = [
    "Instruction",
    "PerformanceMetrics",
    "HardwareSpec",
]
