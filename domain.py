from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Instruction:
    """Lightweight AST node for a single LLO instruction or control op."""

    opcode: str  # e.g. "vmatmul.mubr", "enqueue_dma", "scf.for"

    # SSA topology.
    outputs: List[str]
    inputs: List[str]

    # Best-effort metadata extracted from the raw text.
    # Examples:
    #   {"shape": (8, 128, 128)}
    #   {"size": 4096}
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Hierarchical structure for loops / regions (e.g., scf.for bodies).
    body: List["Instruction"] = field(default_factory=list)

    # Optional source location for debugging (e.g., "filename:line").
    location: Optional[str] = None

    @property
    def is_container(self) -> bool:
        return bool(self.body)


@dataclass
class PerformanceMetrics:
    """Estimated hardware cost of a single instruction occurrence."""

    # Hardware-independent work quantities.
    flops: int = 0
    bytes_accessed: int = 0

    # Hardware-dependent time estimate (in nanoseconds).
    estimated_time_ns: int = 0

    # Classification for analysis / visualization.
    # Typical values: "Compute", "Memory", "Overhead", "Control", "Unknown".
    category: str = "Unknown"

    def __add__(self, other: "PerformanceMetrics") -> "PerformanceMetrics":
        """Support simple aggregation."""
        return PerformanceMetrics(
            flops=self.flops + other.flops,
            bytes_accessed=self.bytes_accessed + other.bytes_accessed,
            estimated_time_ns=self.estimated_time_ns + other.estimated_time_ns,
            # Category is not aggregated; the caller can decide if it matters.
            category=self.category,
        )
