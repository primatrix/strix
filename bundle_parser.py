from __future__ import annotations

import re
from typing import Optional

from .bundle_domain import SourceLoc


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
