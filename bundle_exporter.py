"""Bundle analysis exporters: console table and JSON output."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import IO, Dict, Iterable, List, Optional, Tuple

from .bundle_domain import Bundle, BundleProgram, SourceLoc


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

        entries = _collect_mappings(program, line_filter)

        # Group by file
        by_file: Dict[str, List[Tuple[SourceLoc, List[Tuple[int, int]]]]] = {}
        for loc, slots in entries:
            by_file.setdefault(loc.file, []).append((loc, slots))

        # Resolve source files if --source-root provided
        source_cache: Dict[str, List[str]] = {}
        if source_root:
            source_cache = self._resolve_sources(by_file.keys(), source_root)

        # Count annotated instructions
        total_annotated = sum(len(slots) for _, slots in entries)
        total_locs = len(entries)

        # Build address→bundle map once for opcode lookups
        bundle_map = {b.address: b for b in program.bundles}

        # Header
        out.write("=== Bundle-Source Mapping ===\n")

        for filename in sorted(by_file.keys()):
            file_entries = by_file[filename]
            display_name = os.path.basename(filename)
            out.write(f"File: {display_name}\n\n")

            for loc, slots in file_entries:
                self._write_loc_entry(out, loc, slots, bundle_map, source_cache)

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
        bundle_map: Dict[int, "Bundle"],
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
        opcodes = _count_opcodes(slots, bundle_map)

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
        filenames: Iterable[str],
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

    Tries progressively shorter suffixes of *pod_path* (longest first)
    to find the most specific match under *source_root*.
    """
    parts = pod_path.replace("\\", "/").split("/")
    # Filter empty parts (from leading /)
    parts = [p for p in parts if p]
    # Try longest suffix first for most specific match
    for i in range(len(parts)):
        suffix = os.path.join(*parts[i:])
        candidate = os.path.join(source_root, suffix)
        if os.path.isfile(candidate):
            return candidate
    return None


def _collect_mappings(
    program: BundleProgram,
    line_filter: int | None = None,
) -> List[Tuple[SourceLoc, List[Tuple[int, int]]]]:
    """Collect and filter source-index entries, sorted by line number."""
    entries: List[Tuple[SourceLoc, List[Tuple[int, int]]]] = []
    for loc, slots in program.source_index.items():
        if line_filter is not None:
            if not (loc.start_line <= line_filter <= loc.end_line):
                continue
        entries.append((loc, slots))
    entries.sort(key=lambda e: (e[0].file, e[0].start_line, e[0].start_col))
    return entries


def _count_opcodes(
    slots: List[Tuple[int, int]],
    bundle_map: Dict[int, "Bundle"],
) -> Counter[str]:
    """Count opcode frequency across the given slots."""
    opcodes: Counter[str] = Counter()
    for addr, slot_idx in slots:
        bundle = bundle_map.get(addr)
        if bundle and slot_idx < len(bundle.instructions):
            op = bundle.instructions[slot_idx].opcode
            if op:
                opcodes[op] += 1
    return opcodes


class BundleJsonExporter:
    """Export a BundleProgram as structured JSON."""

    def export(
        self,
        program: BundleProgram,
        output_path: str,
        line_filter: int | None = None,
    ) -> None:
        entries = _collect_mappings(program, line_filter)

        total_annotated = sum(len(slots) for _, slots in entries)

        bundle_map = {b.address: b for b in program.bundles}

        mappings = []
        for loc, slots in entries:
            opcodes = _count_opcodes(slots, bundle_map)
            mappings.append({
                "loc": {
                    "file": loc.file,
                    "start_line": loc.start_line,
                    "start_col": loc.start_col,
                    "end_line": loc.end_line,
                    "end_col": loc.end_col,
                },
                "slots": [
                    {"bundle": f"0x{addr:02x}", "slot": slot_idx}
                    for addr, slot_idx in slots
                ],
                "opcodes": opcodes,
            })

        data = {
            "total_bundles": len(program.bundles),
            "annotated_instructions": total_annotated,
            "mappings": mappings,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
