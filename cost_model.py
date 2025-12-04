from __future__ import annotations

from .domain import PerformanceMetrics

# NOTE: Matmul FLOPs calculation is now hardware-specific and handled
# dynamically in MatmulOpEvent.exec() based on the vector shape.
# For v6e with 256x256 MXU and vector<8x128x2xbf16>:
#   M = 8 * 2 = 16 rows
#   K = N = 256 (MXU width)
#   FLOPs per vmatmul.mubr = 2 * M * K * N = 2 * 16 * 256 * 256 = 2,097,152

__all__ = [
    "PerformanceMetrics",
]

