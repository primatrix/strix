"""Bundle analysis exporters: console table and JSON output."""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import IO, Dict, List, Optional, Tuple

from .bundle_domain import BundleProgram, SourceLoc


class BundleConsoleExporter:
    """Render a BundleProgram as a human-readable console mapping table."""

    def export(
        self,
        program: BundleProgram,
        *,
        file: IO[str] | None = None,
        line_filter: int | None = None,
        source_root: str | None = None,
    ) -> None:
        out = file or sys.stdout

        # Group source locations by file, sorted by start_line
        by_file: Dict[str, List[Tuple[SourceLoc, List[Tuple[int, int]]]]] = {}
        for loc, slots in program.source_index.items():
            if line_filter is not None:
                if not (loc.start_line <= line_filter <= loc.end_line):
                    continue
            by_file.setdefault(loc.file, []).append((loc, slots))

        for entries in by_file.values():
            entries.sort(key=lambda e: (e[0].start_line, e[0].start_col))

        # Resolve source files if --source-root provided
        source_cache: Dict[str, List[str]] = {}
        if source_root:
            source_cache = self._resolve_sources(by_file.keys(), source_root)

        # Count annotated instructions
        total_annotated = sum(
            len(slots)
            for entries in by_file.values()
            for _, slots in entries
        )

        # Count distinct source locations shown
        total_locs = sum(len(entries) for entries in by_file.values())

        # Header
        out.write("=== Bundle-Source Mapping ===\n")

        for filename in sorted(by_file.keys()):
            entries = by_file[filename]
            # Use just the basename for display
            display_name = os.path.basename(filename)
            out.write(f"File: {display_name}\n\n")

            for loc, slots in entries:
                self._write_loc_entry(out, loc, slots, program, source_cache)

        # Summary
        out.write(
            f"Summary: {total_locs} source locations, "
            f"{len(program.bundles)} bundles, "
            f"{total_annotated} annotated instrs\n"
        )

    def _write_loc_entry(
        self,
        out: IO[str],
        loc: SourceLoc,
        slots: List[Tuple[int, int]],
        program: BundleProgram,
        source_cache: Dict[str, List[str]],
    ) -> None:
        # Location label
        if loc.start_line == loc.end_line:
            loc_label = f"L{loc.start_line}:{loc.start_col}-{loc.end_col}"
        else:
            loc_label = f"L{loc.start_line}:{loc.start_col}-{loc.end_line}:{loc.end_col}"

        # Show source code line if available
        lines = source_cache.get(loc.file)
        if lines and 1 <= loc.start_line <= len(lines):
            src_line = lines[loc.start_line - 1].rstrip()
            out.write(f"  {loc_label}\n")
            out.write(f"    > {src_line}\n")
        else:
            out.write(f"  {loc_label}\n")

        # Distinct bundles
        bundle_addrs = sorted(set(addr for addr, _ in slots))
        num_bundles = len(bundle_addrs)
        num_instrs = len(slots)

        if num_bundles <= 4:
            addr_str = ", ".join(f"0x{a:02x}" for a in bundle_addrs)
        else:
            addr_str = (
                f"0x{bundle_addrs[0]:02x}..0x{bundle_addrs[-1]:02x}, non-contiguous"
            )

        out.write(
            f"    {num_instrs} instrs / {num_bundles} bundles ({addr_str})\n"
        )

        # Opcode frequency
        opcodes: Counter[str] = Counter()
        bundle_map = {b.address: b for b in program.bundles}
        for addr, slot_idx in slots:
            bundle = bundle_map.get(addr)
            if bundle and slot_idx < len(bundle.instructions):
                op = bundle.instructions[slot_idx].opcode
                if op:
                    opcodes[op] += 1

        if opcodes:
            parts = []
            for op, count in opcodes.most_common():
                if count > 1:
                    parts.append(f"{op}\u00d7{count}")
                else:
                    parts.append(op)
            out.write(f"    {' '.join(parts)}\n")

        out.write("\n")

    @staticmethod
    def _resolve_sources(
        filenames: "Iterable[str]",
        source_root: str,
    ) -> Dict[str, List[str]]:
        """Map source file paths to local file contents via suffix matching."""
        cache: Dict[str, List[str]] = {}
        for filepath in filenames:
            # Try direct basename match first, then suffix match
            candidate = _find_local_file(filepath, source_root)
            if candidate:
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        cache[filepath] = f.readlines()
                except OSError:
                    pass
        return cache


def _find_local_file(pod_path: str, source_root: str) -> Optional[str]:
    """Find a local file matching the given pod path by suffix match.

    Tries progressively shorter suffixes of *pod_path* until a match is
    found under *source_root*.
    """
    parts = pod_path.replace("\\", "/").split("/")
    # Try full basename first, then longer suffixes
    for i in range(len(parts) - 1, -1, -1):
        suffix = os.path.join(*parts[i:])
        candidate = os.path.join(source_root, suffix)
        if os.path.isfile(candidate):
            return candidate
    return None
