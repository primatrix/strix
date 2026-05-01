from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SourceLoc:
    """Source location from a loc() annotation in a VLIW bundle."""

    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass
class BundleInstruction:
    """A single VLIW slot instruction within a Bundle."""

    opcode: str
    raw_text: str
    outputs: List[str]
    loc: Optional[SourceLoc]


@dataclass
class Bundle:
    """A VLIW Bundle — one clock cycle."""

    address: int
    control_flags: List[str]
    nesting_depth: int
    instructions: List[BundleInstruction]
    comments: List[str]
    is_empty: bool


@dataclass
class BundleProgram:
    """A parsed bundle program with source-location inverted index."""

    bundles: List[Bundle]
    source_index: Dict[SourceLoc, List[Tuple[int, int]]]
