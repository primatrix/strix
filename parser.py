from __future__ import annotations

import os
import re
from collections import deque
from typing import Deque, Dict, List, Optional

from .domain import Instruction


class LLOParser:
    """
    Very lightweight, regex-based parser for Mosaic LLO dumps.

    It is intentionally conservative: the goal is to recover enough structure
    (opcodes, SSA inputs/outputs, loop nesting) to support static cost
    modeling and simple dependency-driven simulation, not to be a full MLIR
    parser.
    """

    def __init__(self) -> None:
        # Internal state for establishing a synthetic DMA dependency chain.
        self._next_dma_token_id: int = 0
        self._dma_token_queue: Deque[str] = deque()

        # Regexes reused across lines.
        self._op_re = re.compile(r"\bllo\.([a-zA-Z0-9_.]+)\b")
        self._result_re = re.compile(
            r"^\s*(?P<results>(?:%[A-Za-z0-9_]+(?::\d+)?(?:\s*,\s*%[A-Za-z0-9_]+(?::\d+)?)*)?)\s*=\s*"
        )
        self._ssa_re = re.compile(r"%[A-Za-z0-9_]+")
        self._type_suffix_re = re.compile(r":\s*(.+)")

    # ------------------------------------------------------------------ public

    def parse_file(self, path: str, exclude_instructions: List[str]) -> Instruction:
        """
        Parse a post-finalize-llo .txt/.mlir file into a root Instruction.

        The root's opcode is "module" and its body contains the top-level
        instructions found in the first func.func body.
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the first func.func body as our entry point.
        func_start = None
        for idx, line in enumerate(lines):
            if "func.func" in line:
                func_start = idx
                break

        if func_start is None:
            # Fallback: parse the whole file as a flat block.
            body = self._parse_block(lines, 0, len(lines) - 1, os.path.basename(path), exclude_instructions=exclude_instructions)
        else:
            func_line = lines[func_start]
            if "{" not in func_line:
                # Look ahead for the opening brace.
                brace_idx = func_start + 1
                while brace_idx < len(lines) and "{" not in lines[brace_idx]:
                    brace_idx += 1
                if brace_idx >= len(lines):
                    body_start = func_start + 1
                    body_end = len(lines) - 1
                else:
                    body_start = brace_idx + 1
                    body_end = self._find_matching_brace(lines, brace_idx) - 1
            else:
                body_start = func_start + 1
                body_end = self._find_matching_brace(lines, func_start) - 1

            body = self._parse_block(lines, body_start, body_end, os.path.basename(path), exclude_instructions=exclude_instructions)

        return Instruction(
            opcode="module",
            outputs=[],
            inputs=[],
            body=body,
            location=os.path.basename(path),
        )

    # ----------------------------------------------------------------- parsing

    def _find_matching_brace(self, lines: List[str], start_idx: int) -> int:
        """
        Given a line index whose line contains an opening '{', find the index
        of the line containing the matching closing '}'.
        """
        depth = lines[start_idx].count("{") - lines[start_idx].count("}")
        idx = start_idx + 1
        while idx < len(lines) and depth > 0:
            depth += lines[idx].count("{") - lines[idx].count("}")
            idx += 1
        return max(idx - 1, start_idx)

    def _parse_block(
        self,
        lines: List[str],
        start_idx: int,
        end_idx: int,
        filename: str,
        exclude_instructions: List[str],
    ) -> List[Instruction]:
        """
        Parse a region of lines into a *flat* instruction stream.

        Control-flow constructs like scf.for/scf.if are represented by
        explicit header instructions plus synthetic ".end" markers:

            scf.for        -> Instruction(opcode="scf.for", ...)
            ...body...
            scf.for end   -> Instruction(opcode="scf.for.end", ...)

        The Instruction.body field is left empty for all nodes; the
        simulator is responsible for reconstructing block structure from
        this flat stream using the start/end markers.
        """
        insts: List[Instruction] = []
        idx = start_idx
        while idx <= end_idx:
            line = lines[idx]
            stripped = line.strip()

            if not stripped:
                idx += 1
                continue

            # scf.for header + body + synthetic end marker.
            if "scf.for" in line and "{" in line:
                loop_end = self._find_matching_brace(lines, idx)

                # Header instruction (no nested body attached).
                header_inst = self._parse_for_header(
                    line=line,
                    line_idx=idx,
                    filename=filename,
                )
                insts.append(header_inst)

                # Body region is parsed recursively but flattened into
                # the same instruction stream.
                if idx + 1 <= loop_end - 1:
                    body_insts = self._parse_block(
                        lines,
                        idx + 1,
                        loop_end - 1,
                        filename,
                        exclude_instructions
                    )
                    insts.extend(body_insts)

                # Synthetic end marker to delimit the loop body.
                end_inst = Instruction(
                    opcode="scf.for.end",
                    outputs=[],
                    inputs=[],
                    attributes={
                        "line_idx": loop_end,
                        "parent_line_idx": idx,
                    },
                    body=[],
                    location=f"{filename}:{loop_end + 1}",
                )
                insts.append(end_inst)

                idx = loop_end + 1
                continue

            # scf.if header + body + synthetic end marker (single-region).
            if stripped.startswith("scf.if") and "{" in line:
                if_end = self._find_matching_brace(lines, idx)

                header_inst = self._parse_if_header(
                    line=line,
                    line_idx=idx,
                    filename=filename,
                )
                insts.append(header_inst)

                if idx + 1 <= if_end - 1:
                    body_insts = self._parse_block(
                        lines,
                        idx + 1,
                        if_end - 1,
                        filename,
                        exclude_instructions,
                    )
                    insts.extend(body_insts)

                end_inst = Instruction(
                    opcode="scf.if.end",
                    outputs=[],
                    inputs=[],
                    attributes={
                        "line_idx": if_end,
                        "parent_line_idx": idx,
                    },
                    body=[],
                    location=f"{filename}:{if_end + 1}",
                )
                insts.append(end_inst)

                idx = if_end + 1
                continue

            # Regular LLO instruction line.
            if "llo." in line:
                inst = self._parse_op_line(
                    line=line,
                    line_idx=idx,
                    filename=filename,
                )
                if inst is not None and (not exclude_instructions or inst.opcode not in exclude_instructions):
                    insts.append(inst)

            idx += 1

        return insts

    def _parse_for_header(
        self,
        line: str,
        line_idx: int,
        filename: str,
    ) -> Instruction:
        """
        Parse the header of an scf.for. We preserve the SSA results and the
        loop bounds tokens (lb/ub/step) for later analysis in the simulator.
        """
        outputs: List[str] = []
        m_res = self._result_re.match(line)
        if m_res and m_res.group("results"):
            outputs = [tok.strip() for tok in m_res.group("results").split(",")]

        attrs: Dict[str, object] = {
            "raw_header": line.strip(),
            "line_idx": line_idx,
        }

        # Extract lb/ub/step tokens for ValueResolver.trip_count_for().
        m_for = re.search(
            r"\bscf\.for\s+%[A-Za-z0-9_]+\s*=\s*([^ ]+)\s+to\s+([^ ]+)\s+step\s+([^ :]+)",
            line,
        )
        if m_for:
            attrs["for_lb"] = m_for.group(1).strip()
            attrs["for_ub"] = m_for.group(2).strip()
            attrs["for_step"] = m_for.group(3).strip()

        return Instruction(
            opcode="scf.for",
            outputs=outputs,
            inputs=[],
            attributes=attrs,
            body=[],
            location=f"{filename}:{line_idx + 1}",
        )

    def _parse_if_header(
        self,
        line: str,
        line_idx: int,
        filename: str,
    ) -> Instruction:
        """
        Parse the header of an scf.if. We record the condition SSA as input
        and keep the raw header for debugging / display.
        """
        outputs: List[str] = []
        m_res = self._result_re.match(line)
        if m_res and m_res.group("results"):
            outputs = [tok.strip() for tok in m_res.group("results").split(",")]

        # All SSA tokens on this line are treated as inputs; in practice the
        # first one after "scf.if" is the condition.
        inputs = self._ssa_re.findall(line)

        attrs: Dict[str, object] = {
            "raw_header": line.strip(),
            "line_idx": line_idx,
        }

        return Instruction(
            opcode="scf.if",
            outputs=outputs,
            inputs=inputs,
            attributes=attrs,
            body=[],
            location=f"{filename}:{line_idx + 1}",
        )

    def _parse_op_line(
        self,
        line: str,
        line_idx: int,
        filename: str,
    ) -> Optional[Instruction]:
        m_op = self._op_re.search(line)
        if not m_op:
            return None

        opcode = m_op.group(1)

        # Outputs: anything before '=' that looks like a %name.
        outputs: List[str] = []
        prefix = line[: m_op.start()]
        m_res = self._result_re.match(prefix)
        if m_res and m_res.group("results"):
            outputs = [tok.strip() for tok in m_res.group("results").split(",")]

        # Inputs: SSA operands found in parentheses after the opcode.
        inputs: List[str] = []
        # We only look at the part after the opcode; this avoids grabbing
        # loop induction variables from earlier in the line.
        suffix = line[m_op.end() :]

        # Try parenthesized format first: "llo.opcode"(%arg1, %arg2)
        paren_start = suffix.find("(")
        paren_end = suffix.find(")")
        if paren_start != -1 and paren_end != -1 and paren_end > paren_start:
            arg_slice = suffix[paren_start:paren_end]
            inputs = self._ssa_re.findall(arg_slice)
        else:
            # Try infix format: llo.opcode %arg1, %arg2
            # This is common for scalar arithmetic ops like llo.smul.u32, llo.smin.s32
            # Find all SSA values after the opcode, before any attribute block <{...}>
            attr_start = suffix.find("<{")
            colon_pos = suffix.find(":")

            # Take suffix up to attributes or type annotation
            end_pos = len(suffix)
            if attr_start != -1:
                end_pos = min(end_pos, attr_start)
            if colon_pos != -1:
                end_pos = min(end_pos, colon_pos)

            operand_region = suffix[:end_pos]
            inputs = self._ssa_re.findall(operand_region)

        attrs: Dict[str, object] = {"line_idx": line_idx}

        # Preserve a trimmed copy of the raw line for later debugging /
        # opcode-specific parsing (e.g. extracting vector tile shapes).
        attrs["raw_line"] = line.strip()

        # Best-effort capture of the trailing type / signature suffix,
        # e.g. "(vector<8x128xf32>) -> ()" or "() -> vector<8x128xf32>".
        m_type = self._type_suffix_re.search(line)
        if m_type:
            attrs["type_suffix"] = m_type.group(1).strip()

        # Scalar constant value, if present (for ValueResolver).
        if opcode == "constant":
            m_val = re.search(
                r"value\s*=\s*([-0-9]+)\s*:\s*i[0-9]+",
                line,
            )
            if m_val:
                try:
                    attrs["value"] = int(m_val.group(1))
                except ValueError:
                    pass

        # DMA enqueue: create a synthetic DMA token to establish a dependency
        # edge from the enqueue to the corresponding dma_done.
        if opcode == "enqueue_dma":
            # Create a synthetic DMA token to establish a dependency edge
            # from the enqueue to the corresponding dma_done.
            token = f"__dma_token_{self._next_dma_token_id}"
            self._next_dma_token_id += 1
            outputs.append(token)
            self._dma_token_queue.append(token)

        elif opcode == "dma_done":
            # Consume the next outstanding DMA token, if any.
            if self._dma_token_queue:
                token = self._dma_token_queue.popleft()
                inputs.append(token)

        return Instruction(
            opcode=opcode,
            outputs=outputs,
            inputs=inputs,
            attributes=attrs,
            body=[],
            location=f"{filename}:{line_idx + 1}",
        )
