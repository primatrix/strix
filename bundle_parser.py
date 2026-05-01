from __future__ import annotations

import re
from typing import List, Optional

from .bundle_domain import Bundle, BundleInstruction, SourceLoc


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

    # Bundle line: <addr> [flags]: [>...] { ... } [comments]
    _BUNDLE_HEAD_RE = re.compile(
        r'^\s*(?P<addr>0x[0-9a-fA-F]+|0)\s*'
        r'(?P<flags>[A-Z, ]*?)\s*:\s*'
        r'(?P<nesting>>*)\s*\{'
    )
    _KNOWN_FLAGS = {"LH", "LB", "LE", "PB", "PF", "CT"}

    def _parse_bundle_line(self, full_text: str) -> Optional[Bundle]:
        """Parse a complete bundle line (possibly pre-joined from multi-line)."""
        m = self._BUNDLE_HEAD_RE.match(full_text)
        if not m:
            return None

        addr_str = m.group("addr")
        address = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)

        flags_raw = m.group("flags").strip()
        control_flags = [f for f in flags_raw.split() if f in self._KNOWN_FLAGS]

        nesting_depth = len(m.group("nesting"))

        # Extract content between first { and last }
        brace_start = full_text.index("{", m.start())
        brace_end = full_text.rindex("}")
        inner = full_text[brace_start + 1 : brace_end].strip()

        # Comments after the closing }
        after_brace = full_text[brace_end + 1 :].strip()
        comments: List[str] = []
        if after_brace:
            # Extract text inside /* ... */
            for cm in re.finditer(r'/\*\s*(.*?)\s*\*/', after_brace):
                comments.append(cm.group(1))

        # Empty bundle
        if not inner:
            return Bundle(address, control_flags, nesting_depth, [], comments, is_empty=True)

        # Split instructions by ;;
        raw_instrs = inner.split(";;")
        instructions = [self._parse_instruction(s) for s in raw_instrs if s.strip()]

        return Bundle(address, control_flags, nesting_depth, instructions, comments, is_empty=False)
