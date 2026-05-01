from __future__ import annotations

import re
from typing import List, Optional

from .bundle_domain import BundleInstruction, SourceLoc


class BundleParser:
    """Parser for *-final_bundles.txt files produced by the LLO compiler."""

    # loc("file":L:C to :C2) or loc("file":L1:C1 to L2:C2)
    _LOC_RE = re.compile(
        r'"(?P<file>[^"]+)":(?P<sl>\d+):(?P<sc>\d+)'
        r'\s+to\s+(?:(?P<el>\d+):)?:?(?P<ec>\d+)'
    )

    def _parse_loc(self, loc_str: str) -> Optional[SourceLoc]:
        """Parse a loc() annotation body into a SourceLoc.

        Accepts:
          "file":L:C1 to :C2       (single-line)
          "file":L1:C1 to L2:C2    (cross-line)
        """
        m = self._LOC_RE.search(loc_str)
        if not m:
            return None
        start_line = int(m.group("sl"))
        end_line = int(m.group("el")) if m.group("el") else start_line
        return SourceLoc(
            file=m.group("file"),
            start_line=start_line,
            start_col=int(m.group("sc")),
            end_line=end_line,
            end_col=int(m.group("ec")),
        )

    # %name = opcode ...
    _ASSIGN_RE = re.compile(r'^\s*(?P<out>%[A-Za-z0-9_]+)\s*=\s*(?P<opcode>[A-Za-z0-9_.]+)')
    # /* loc(...) */  — find the first loc() in any comment
    _LOC_COMMENT_RE = re.compile(r'loc\((?P<body>[^)]+)\)')

    def _parse_instruction(self, text: str) -> BundleInstruction:
        """Parse a single VLIW slot instruction text."""
        text = text.strip()
        opcode = ""
        outputs: List[str] = []

        m = self._ASSIGN_RE.match(text)
        if m:
            outputs = [m.group("out")]
            opcode = m.group("opcode")

        # Extract loc from the first loc() in comments
        loc: Optional[SourceLoc] = None
        m_loc = self._LOC_COMMENT_RE.search(text)
        if m_loc:
            loc = self._parse_loc(m_loc.group("body"))

        return BundleInstruction(
            opcode=opcode,
            raw_text=text,
            outputs=outputs,
            loc=loc,
        )
