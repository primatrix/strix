from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

import copy

from .domain import Instruction, PerformanceMetrics
from .hardware import HardwareSpec
from .value_resolver import ValueResolver

if TYPE_CHECKING:  # avoid import cycles at runtime
    from .simulator import Simulator


@dataclass
class BlockContext:
    """
    Context for block boundary detection and stack management.

    Encapsulates the event stack and provides operations for building
    the event tree during parsing.
    """
    event_stack: List["OpEvent"]

    def is_inside(self, event_type: Type["OpEvent"]) -> bool:
        """Check if currently inside a block of given type."""
        return any(isinstance(ev, event_type) for ev in self.event_stack)

    def current_block(self) -> Optional["OpEvent"]:
        """Get the innermost current block."""
        return self.event_stack[-1] if self.event_stack else None

    def add(self, ev: "OpEvent") -> None:
        """Add an event as a child of the current block."""
        if not self.event_stack:
            return
        self.current_block().children.append(ev)

    def enter(self, ev: "OpEvent") -> None:
        """
        Enter a new block: attach it to the current parent and push onto stack.
        """
        if not self.event_stack:
            return
        parent = self.current_block()
        ev.on_enter(parent)
        self.event_stack.append(ev)

    def close_if_needed(self, inst: Instruction) -> bool:
        """
        Close the current block if this instruction is its end marker.

        In LLO, each block has a one-to-one correspondence with its end
        marker (scf.for/scf.for.end, scf.if/scf.if.end, vmatprep.subr/vdwg),
        so each end instruction closes exactly one block.

        Returns True if a block was closed.
        """
        if len(self.event_stack) <= 1:
            return False
        top = self.event_stack[-1]
        if top.is_end_inst(inst, self):
            self.event_stack.pop()
            return True
        return False


class OpKind(str, Enum):
    """Semantic category for an OpEvent."""

    ROOT = "root"
    LOOP = "loop"
    IF = "if"
    LEAF = "leaf"
    STALL = "stall"
    BLOCK = "block"


class OpStream(str, Enum):
    """Logical execution stream for an OpEvent."""

    VPU = "VPU"
    DMA = "DMA"
    CONTROL = "Control"


@dataclass
class OpEvent:
    """
    One dynamic execution event in the simulated program.

    - name: opcode or block name.
    - kind: see OpKind.
    - flops/bytes: single-execution work quantities (no loop scaling).
    - duration_ns: single-execution time in nanoseconds (no loop scaling).
    - start_time_ns/end_time_ns: simulated timestamps in nanoseconds.
    - stream: see OpStream.
    - attributes: extra metadata (trip counts, source location, etc.).
    - children: nested events (for blocks like scf.for / scf.if).
    """

    name: str
    kind: OpKind
    flops: int = 0
    bytes: int = 0
    duration_ns: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0
    stream: OpStream = OpStream.VPU
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["OpEvent"] = field(default_factory=list)

    def compress_children(self, threshold: int = 10, pair_threshold: int = 5) -> None:
        """
        Merge consecutive similar operations to reduce trace clutter.

        This is a generic compression method that can be applied to any
        container event. It recognizes common patterns and creates
        appropriate BlockEvent types:
        1. Consecutive runs of the same opcode → RepeatedBlockEvent
        2. Alternating pairs like constant + vector_load → PatternBlockEvent

        Args:
            threshold: Minimum consecutive operations to merge into one
            pair_threshold: Minimum pairs to merge
        """
        if not self.children:
            return

        compressed: List[OpEvent] = []
        i = 0

        while i < len(self.children):
            current = self.children[i]

            # Pattern 1: Check for alternating constant + vector_load pairs
            if (i + 1 < len(self.children) and
                current.name == "constant" and
                self.children[i + 1].name == "vector_load"):

                # Count consecutive const+load pairs
                pair_count = 0
                j = i
                while (j + 1 < len(self.children) and
                       self.children[j].name == "constant" and
                       self.children[j + 1].name == "vector_load"):
                    pair_count += 1
                    j += 2

                if pair_count >= pair_threshold:
                    # Collect children for timing calculation
                    pattern_children = []
                    for k in range(i, j):
                        pattern_children.append(self.children[k])

                    # Create PatternBlockEvent
                    pattern_block = PatternBlockEvent(
                        pattern_name="const+load",
                        count=pair_count,
                        stream=OpStream.VPU,
                    )
                    pattern_block.children = pattern_children
                    compressed.append(pattern_block)
                    i = j
                    continue

            # Pattern 2: Check for consecutive same-opcode operations
            if current.kind == OpKind.LEAF:
                run_count = 1
                j = i + 1
                while (j < len(self.children) and
                       self.children[j].kind == OpKind.LEAF and
                       self.children[j].name == current.name):
                    run_count += 1
                    j += 1

                if run_count >= threshold:
                    # Collect children for timing calculation
                    repeated_children = []
                    for k in range(i, j):
                        repeated_children.append(self.children[k])

                    # Create RepeatedBlockEvent
                    repeated_block = RepeatedBlockEvent(
                        opcode=current.name,
                        count=run_count,
                        stream=current.stream,
                    )
                    repeated_block.children = repeated_children
                    compressed.append(repeated_block)
                    i = j
                    continue

            # No pattern matched, keep as-is
            compressed.append(current)
            i += 1

        self.children = compressed

    def get_metrics(self) -> PerformanceMetrics:
        """
        Aggregated performance metrics for this event subtree.

        For leaf events, this reflects this node's own flops/bytes/time.
        For container events (ROOT/LOOP/IF/BLOCK), this returns the sum
        of child metrics.
        """
        if not self.children:
            return PerformanceMetrics(
                flops=self.flops,
                bytes_accessed=self.bytes,
                estimated_time_ns=self.duration_ns,
                category=self.attributes.get("category", "Unknown"),
            )

        total_flops = 0
        total_bytes = 0
        total_time = 0
        for child in self.children:
            m = child.get_metrics()
            total_flops += m.flops
            total_bytes += m.bytes_accessed
            total_time += m.estimated_time_ns

        return PerformanceMetrics(
            flops=total_flops,
            bytes_accessed=total_bytes,
            estimated_time_ns=total_time,
            category="Aggregate",
        )

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        """
        Whether this instruction starts a block of this event type.

        Args:
            inst: The instruction to check.
            context: Optional context providing information about the current
                     execution stack (e.g., to avoid creating nested blocks
                     of the same type when not intended).
        """
        return False

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        """
        Whether this instruction ends a block of this event type.

        Args:
            inst: The instruction to check.
            context: Optional context providing information about the current
                     execution stack.
        """
        return False

    def on_enter(self, parent: "OpEvent") -> None:
        """
        Called when this event is pushed onto the stack. The default
        behaviour is to attach it as a child of the current parent.
        """
        parent.children.append(self)

    def prepare(
        self,
        resolver: Optional[ValueResolver] = None,
    ) -> None:
        """
        Structural preparation hook run before scheduling.

        The simulator first builds an OpEvent tree from a flat instruction
        stream, then calls `prepare()` once on the root. Container-style
        events (ROOT/LOOP/IF/BLOCK) override this to interpret control
        flow (e.g. loop expansion, if pruning, MXU macro grouping). The
        default behaviour is to recurse into children without changing
        metrics or timing.
        """
        for child in self.children:
            child.prepare(resolver)

    # ------------------------------------------------------------------ cost model hook

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        """
        Cost model entry point for this event.

        Leaf / BLOCK events are expected to override this and return
        their own `PerformanceMetrics`. Container-style events (ROOT /
        LOOP / IF) are never scheduled directly, so the default
        implementation just returns an empty, zero-cost metrics object.
        """
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=0,
            category=self.attributes.get("category", "Control"),
        )


# ---------------------------------------------------------------------------
# Helpers for higher-level cost modelling
# ---------------------------------------------------------------------------


def group_boundary_ssa(instructions: List[Instruction]) -> Tuple[List[str], List[str]]:
    """
    Compute a simple SSA boundary for a group of instructions:
      * inputs: SSA tokens read by the group but not defined within,
      * outputs: SSA tokens defined within the group.

    This is conservative (all defs are treated as outputs) but sufficient
    for establishing dependency edges and ready-times.
    """
    defs: set[str] = set()
    uses: set[str] = set()
    for inst in instructions:
        for tok in inst.outputs:
            defs.add(tok)
        for tok in inst.inputs:
            uses.add(tok)
    inputs = sorted(uses - defs)
    outputs = sorted(defs)
    return inputs, outputs


def _infer_vector_tile(
    instructions: List[Instruction],
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Best-effort inference of a representative vector tile shape and dtype
    from a group of instructions.

    We look for the first "vector<...>" occurrence in any Instruction's
    parsed type suffix and interpret it as:

        vector<d0xd1x...xDTYPE>

    returning ([d0, d1, ...], "DTYPE") when successful.
    """
    for inst in instructions:
        type_suffix = inst.attributes.get("type_suffix")
        if not isinstance(type_suffix, str):
            continue
        m_vec = re.search(r"vector<([^>]+)>", type_suffix)
        if not m_vec:
            continue
        inner = m_vec.group(1).strip()
        # Split like "8x128xf32" -> ["8", "128", "f32"].
        parts = [p.strip() for p in inner.split("x") if p.strip()]
        if len(parts) < 2:
            continue
        dtype = parts[-1]
        dim_tokens = parts[:-1]
        dims: List[int] = []
        ok = True
        for tok in dim_tokens:
            try:
                dims.append(int(tok))
            except ValueError:
                ok = False
                break
        if not ok or not dims:
            continue
        return dims, dtype

    return None, None


def _dtype_size(dtype: str) -> int:
    """Rough element size in bytes for a few common dtypes."""
    d = dtype.lower()
    if "bf16" in d:
        return 2
    if d in ("f16", "half"):
        return 2
    if d in ("f32", "float", "float32"):
        return 4
    if "int8" in d or d in ("i8", "s8", "u8"):
        return 1
    if "int16" in d or d in ("i16", "s16", "u16"):
        return 2
    if "int32" in d or d in ("i32", "s32", "u32"):
        return 4
    # Fallback to 4 bytes.
    return 4


def _dtype_peak_flops(spec: HardwareSpec, dtype: Optional[str]) -> float:
    """
    Heuristic mapping from dtype string to a peak FLOP/s estimate using
    the fields available in HardwareSpec.
    """
    if not dtype:
        return spec.peak_bf16_flops_per_s

    d = dtype.lower()
    # Treat bf16/f16 as the baseline peak.
    if "bf16" in d or d in ("f16", "half"):
        return spec.peak_bf16_flops_per_s
    # Assume f32 is somewhat slower than bf16; 0.5x is a crude default.
    if d in ("f32", "float", "float32"):
        return spec.peak_bf16_flops_per_s * 0.5
    # Map int8 to peak_int8_tops when available.
    if "int8" in d or d in ("i8", "s8", "u8"):
        return spec.peak_int8_tops * 1e12
    return spec.peak_bf16_flops_per_s


def _vector_flop_cost(
    inst: Instruction,
    spec: HardwareSpec,
    flops_per_element: float,
) -> PerformanceMetrics:
    num_elems = int(inst.attributes.get("num_elements", 0))
    flops = int(flops_per_element * num_elems)
    if flops <= 0:
        # Fallback to a tiny no-op to keep the simulator moving.
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=1,
            category="Compute",
        )
    time_units = flops / max(spec.flops_per_time_unit, 1.0)
    time_ns = spec.dispatch_overhead_ns + time_units * (spec.time_unit_s * 1e9)
    return PerformanceMetrics(
        flops=flops,
        bytes_accessed=0,
        estimated_time_ns=int(time_ns),
        category="Compute",
    )


# ---------------------------------------------------------------------------
# Block-style events used by the unified simulator stack.
# ---------------------------------------------------------------------------


class LoopBlockEvent(OpEvent):
    """
    Container event for a single static scf.for region.

    The simulator treats this as a block that *consumes* all instructions
    between "scf.for" and the matching "scf.for.end". When this block is
    popped from the stack, it runs a nested simulation over its recorded
    body instructions for a limited number of iterations, attaching the
    resulting child events to itself.
    """

    def __init__(self, header: Instruction):
        attrs: Dict[str, Any] = dict(header.attributes or {})
        super().__init__(
            name=header.opcode,
            kind=OpKind.LOOP,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.CONTROL,
            attributes=attrs,
            children=[],
        )
        self.header = header

    # ---- Stack protocol -------------------------------------------------

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return inst.opcode == "scf.for"

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return inst.opcode == "scf.for.end"

    def on_enter(self, parent: OpEvent) -> None:
        """Attach to parent as a child when the block opens."""
        parent.children.append(self)

    def prepare(
        self,
        resolver: Optional[ValueResolver] = None,
    ) -> None:
        """
        Prepare the loop body structure.

        Strategy:
        - If trip_count <= 100: expand all iterations via deepcopy
        - If trip_count > 100: keep single iteration, scale during scheduling
        """
        # Resolve static trip count when possible.
        trip_count = 1
        if resolver is not None:
            try:
                tc = resolver.trip_count_for(self.header)
                if isinstance(tc, int) and tc > 0:
                    trip_count = tc
            except Exception:
                pass

        self.attributes["trip_count"] = trip_count

        # Resolve and store loop bounds as integers instead of SSA tokens.
        # Remove raw_header to avoid clutter.
        if "raw_header" in self.attributes:
            del self.attributes["raw_header"]

        if resolver is not None:
            # Resolve for_lb, for_ub, for_step to their integer values.
            lb_tok = self.header.attributes.get("for_lb")
            ub_tok = self.header.attributes.get("for_ub")
            step_tok = self.header.attributes.get("for_step")

            if lb_tok is not None:
                lb_val = resolver.resolve_token(str(lb_tok))
                if lb_val is not None:
                    self.attributes["for_lb"] = lb_val

            if ub_tok is not None:
                ub_val = resolver.resolve_token(str(ub_tok))
                if ub_val is not None:
                    self.attributes["for_ub"] = ub_val

            if step_tok is not None:
                step_val = resolver.resolve_token(str(step_tok))
                if step_val is not None:
                    self.attributes["for_step"] = step_val

        # Prepare first iteration's children (structural processing).
        for child in self.children:
            child.prepare(resolver)

        # Decide whether to expand iterations
        # Note: Even with compression, 20 iterations × 27 nodes = 540 nodes,
        # and each aggregated node contains many children that need deepcopy.
        if trip_count <= 20:
            # Small loop: expand all iterations via deepcopy
            self.attributes["expanded"] = True
            if trip_count > 1:
                template = self.children[:]
                for _ in range(trip_count - 1):
                    for child in template:
                        cloned = copy.deepcopy(child)
                        self.children.append(cloned)
        else:
            # Large loop: keep single iteration, will be scaled during scheduling
            self.attributes["expanded"] = False


class RepeatedBlockEvent(OpEvent):
    """
    Block event representing a repeated sequence of the same operation.

    For example: 4096 consecutive vbitcast operations → RepeatedBlockEvent
    """

    def __init__(self, opcode: str, count: int, stream: OpStream):
        super().__init__(
            name=f"{opcode} ×{count}",
            kind=OpKind.BLOCK,
            stream=stream,
            attributes={"count": count, "opcode": opcode, "pattern": "repeated"},
        )
        self.opcode = opcode
        self.count = count

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        # Not created from instructions directly
        return False

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return False

    def on_enter(self, parent: OpEvent) -> None:
        parent.children.append(self)

    def prepare(self, resolver: Optional[ValueResolver] = None) -> None:
        # Children already prepared when created
        pass

    def exec(self, spec: HardwareSpec, resolver: ValueResolver) -> PerformanceMetrics:
        # Should never be called - timing comes from aggregating children
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=0,
            category="Control",
        )


class PatternBlockEvent(OpEvent):
    """
    Block event representing a repeated pattern of operations.

    For example: alternating constant + vector_load pairs
    """

    def __init__(self, pattern_name: str, count: int, stream: OpStream):
        super().__init__(
            name=f"{pattern_name} ×{count}",
            kind=OpKind.BLOCK,
            stream=stream,
            attributes={"count": count, "pattern": pattern_name},
        )
        self.pattern_name = pattern_name
        self.count = count

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return False

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return False

    def on_enter(self, parent: OpEvent) -> None:
        parent.children.append(self)

    def prepare(self, resolver: Optional[ValueResolver] = None) -> None:
        pass

    def exec(self, spec: HardwareSpec, resolver: ValueResolver) -> PerformanceMetrics:
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=0,
            category="Control",
        )


class IfBlockEvent(OpEvent):
    """
    Container event for a single-region scf.if.

    The simulator treats this as a block that consumes all instructions
    between "scf.if" and "scf.if.end". When the block is closed, it
    evaluates the condition using ValueResolver and, if taken, runs a
    nested simulation over the body instructions, attaching the resulting
    child events to itself.
    """

    def __init__(self, inst: Instruction):
        attrs: Dict[str, Any] = dict(inst.attributes or {})
        super().__init__(
            name=inst.opcode,
            kind=OpKind.IF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.CONTROL,
            attributes=attrs,
            children=[],
        )
        self.inst = inst

    # ---- Stack protocol -------------------------------------------------

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return inst.opcode == "scf.if"

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        return inst.opcode == "scf.if.end"

    def on_enter(self, parent: OpEvent) -> None:
        parent.children.append(self)

    def prepare(
        self,
        resolver: Optional[ValueResolver] = None,
    ) -> None:
        """
        Evaluate the scf.if condition (when statically known) and keep
        only the taken branch body. The surviving children are then
        structurally prepared in-place.
        """
        header = self.inst
        cond_var: Optional[str] = None
        cond_taken: Optional[bool] = None

        # The first SSA input of the header is the condition.
        if header.inputs:
            cond_var = header.inputs[0]
            if resolver is not None:
                val = resolver.resolve_token(cond_var)
                if val is not None:
                    cond_taken = bool(val)

        if cond_var is not None:
            self.attributes.setdefault("cond_var", cond_var)
        if cond_taken is not None:
            self.attributes.setdefault("cond_taken", cond_taken)

        # scf.if in this simplified form has a single region: if the
        # condition is taken (or unknown), we execute all children;
        # otherwise we drop them.
        effective_taken = cond_taken is None or cond_taken

        if not effective_taken:
            # Condition known-false: no children execute.
            self.children = []
            return

        # Condition taken or unknown: prepare children structurally.
        for child in self.children:
            child.prepare(resolver)


class AddOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class SubOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class MulOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class MaxOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class ExpOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        # Treat exp as 1 FLOP per element for now; calibrate later.
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class RecipOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return _vector_flop_cost(self.inst, spec, flops_per_element=1.0)


class VectorLoadOpEvent(OpEvent):
    """VMEM vector load."""

    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        size_bytes = int(self.inst.attributes.get("bytes", 0))
        if size_bytes <= 0:
            return PerformanceMetrics(
                flops=0,
                bytes_accessed=0,
                estimated_time_ns=1,
                category="Overhead",
            )
        time_units = size_bytes / max(spec.hbm_bytes_per_time_unit, 1.0)
        time_ns = spec.dispatch_overhead_ns + time_units * (spec.time_unit_s * 1e9)
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=size_bytes,
            estimated_time_ns=int(time_ns),
            category="Overhead",
        )


class VectorStoreOpEvent(OpEvent):
    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        size_bytes = int(self.inst.attributes.get("bytes", 0))
        if size_bytes <= 0:
            return PerformanceMetrics(
                flops=0,
                bytes_accessed=0,
                estimated_time_ns=1,
                category="Overhead",
            )
        time_units = size_bytes / max(spec.hbm_bytes_per_time_unit, 1.0)
        time_ns = spec.dispatch_overhead_ns + time_units * (spec.time_unit_s * 1e9)
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=size_bytes,
            estimated_time_ns=int(time_ns),
            category="Overhead",
        )


class EnqueueDMAOpEvent(OpEvent):
    """HBM <-> VMEM DMA enqueue (async, runs on DMA engine)."""

    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.DMA,  # Runs on DMA track
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        # Resolve size from last input (DMA size parameter)
        size_bytes = 0
        if resolver is not None:
            from_resolver = resolver.dma_size_for(self.inst)
            if from_resolver is not None:
                size_bytes = int(from_resolver)
        if not size_bytes:
            size_bytes = int(self.inst.attributes.get("size", 0))

        # Store size for later use
        self.attributes["dma_size"] = size_bytes

        # Calculate actual DMA transfer time (no VPU overhead)
        if size_bytes > 0:
            transfer_time_ns = size_bytes / max(spec.hbm_bytes_per_time_unit, 1.0)
        else:
            transfer_time_ns = 1.0  # Minimal time for 0-byte transfers

        return PerformanceMetrics(
            flops=0,
            bytes_accessed=size_bytes,
            estimated_time_ns=int(transfer_time_ns),
            category="Memory",
        )


class DmaDoneOpEvent(OpEvent):
    """DMA completion synchronization point."""

    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,  # Runs on VPU, may stall
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        # Actual timing is computed in _schedule_event based on DMA completion
        # This is just a placeholder
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=1,  # Best case: DMA already done
            category="Overhead",
        )
    
class BlockOpEvent(OpEvent):
    """Fallback for opcodes without an explicit model."""

    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.BLOCK,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=1,
            category="Control",
        )

class UnknownOpEvent(OpEvent):
    """Fallback for opcodes without an explicit model."""

    def __init__(self, inst: Instruction):
        super().__init__(
            name=inst.opcode,
            kind=OpKind.LEAF,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes=dict(inst.attributes or {}),
            children=[],
        )
        self.inst = inst

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        return PerformanceMetrics(
            flops=0,
            bytes_accessed=0,
            estimated_time_ns=0,
            category="Unknown",
        )


# ---------------------------------------------------------------------------
# Macro-level MXU matmul event (block-style OpEvent).
# ---------------------------------------------------------------------------


class MatmulOpEvent(OpEvent):
    """
    Macro-level MXU matmul event representing a whole sequence of LLO
    instructions:

      vmatprep.subr + vlatchi* +
      (vmatprep.mubr + vmatmul.mubr + vmatres + vadd.f32)* +
      vdwg

    For now we approximate FLOPs as (#vmatmul.mubr tiles) times
    DEFAULT_MATMUL_FLOPS_PER_CALL, and time as idealized compute time
    under peak BF16 throughput. This is intentionally coarse and can be
    refined as we model dtype, tile shapes and data movement in more
    detail.
    """

    def __init__(self, header: Instruction):
        super().__init__(
            name="mxu.matmul",
            kind=OpKind.BLOCK,
            flops=0,
            bytes=0,
            duration_ns=0,
            start_time_ns=0,
            end_time_ns=0,
            stream=OpStream.VPU,
            attributes={},
            children=[],
        )
        # Header instruction (typically the first vmatprep.subr).
        self.header = header
        # Flat list of micro-op Instructions that form this macro, filled
        # during `prepare()`.
        self.instructions: List[Instruction] = []

    @classmethod
    def is_start_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        """
        Heuristic: MXU macro begins at the first vmatprep.subr.

        Note: vmatprep.subr appears multiple times within a single matmul
        macro (e.g., 16 times for loading LHS weights). We use the context
        to avoid creating nested MatmulOpEvent blocks.
        """
        if inst.opcode != "vmatprep.subr":
            return False

        # If we're already inside a MatmulOpEvent, this is just another
        # vmatprep.subr within the same macro, not a new block start.
        if context and context.is_inside(MatmulOpEvent):
            return False

        return True

    @classmethod
    def is_end_inst(cls, inst: Instruction, context: Optional[BlockContext] = None) -> bool:
        """Heuristic: MXU macro ends at the next vdwg."""
        return inst.opcode == "vdwg"

    def prepare(
        self,
        resolver: Optional[ValueResolver] = None,
    ) -> None:
        """
        Collapse the MXU micro-ops in this block into a single macro
        event. We record the underlying Instruction objects in
        `self.instructions` and drop the children so that scheduling
        treats this as an atomic block.
        """
        insts: List[Instruction] = [self.header]
        for ev in self.children:
            inst = getattr(ev, "inst", None)
            if isinstance(inst, Instruction):
                insts.append(inst)
        self.instructions = insts

        # Drop micro-op children; the simulator will treat this as a single
        # schedulable macro event.
        self.children = []

    def exec(
        self,
        spec: HardwareSpec,
        resolver: Optional[ValueResolver],
    ) -> PerformanceMetrics:
        """
        Cost model for the MXU matmul macro based on the micro-op
        Instruction sequence captured during `prepare()`.

        On v6e with 256x256 MXU:
        - RHS (weights) is always 256x256
        - LHS (activations) is M x 256, where M is inferred from vector shape
          e.g., vector<8x128x2xbf16> -> M = 8*2 = 16 rows
        - FLOPs per vmatmul.mubr = 2 * M * 256 * 256

        Padding (indicated by vmatprep.mubr(%0) or vlatchi(%0)) doesn't
        change the compute cost, only the effective data read/written.
        """
        insts = self.instructions or []

        # Infer tile shape and dtype from vector type signatures.
        dims, dtype = _infer_vector_tile(insts)

        # Extract M dimension from vector shape.
        # For vector<8x128x2xbf16>: dims = [8, 128, 2]
        # M = dims[0] * dims[2] = 8 * 2 = 16 rows
        if dims and len(dims) >= 3:
            m_rows = dims[0] * dims[2]  # first_dim * pack_dim
        elif dims and len(dims) >= 1:
            m_rows = dims[0]  # fallback if no packing
        else:
            m_rows = 16  # default fallback

        # Count actual vmatmul.mubr operations.
        num_vmatmul_mubr = sum(1 for inst in insts if inst.opcode == "vmatmul.mubr")

        # Count padding vs non-padding vmatprep.mubr to estimate effective bytes.
        num_vmatprep_mubr_total = sum(
            1 for inst in insts if inst.opcode == "vmatprep.mubr"
        )
        num_vmatprep_mubr_padding = sum(
            1 for inst in insts
            if inst.opcode == "vmatprep.mubr" and "%0" in inst.inputs
        )
        num_vmatprep_mubr_actual = num_vmatprep_mubr_total - num_vmatprep_mubr_padding

        # Count vlatchi operations for RHS (weights).
        num_vlatchi_total = sum(1 for inst in insts if inst.opcode == "vlatchi")
        num_vlatchi_padding = sum(
            1 for inst in insts
            if inst.opcode == "vlatchi" and "%0" in inst.inputs
        )
        num_vlatchi_actual = num_vlatchi_total - num_vlatchi_padding

        dt_size = _dtype_size(dtype) if dtype else 2  # default to bf16 = 2 bytes

        # For v6e with vector<8x128x2xbf16>:
        # - LHS (activations): M x 256 bf16, where M = 8*2 = 16
        #   But vmatprep.mubr loads the actual data: M x 128 bf16 (padded to 256)
        # - RHS (weights): 256 x 256 bf16
        #   But vlatchi loads in chunks, actual data varies with padding
        # - Output: M x 256 f32

        # Infer actual data size from vector dimensions.
        if dims and len(dims) >= 2:
            k_cols = dims[1]  # e.g., 128 from vector<8x128x2>
        else:
            k_cols = 128  # default fallback

        # Bytes per operation (actual data, not padded).
        bytes_per_lhs_tile = m_rows * k_cols * dt_size  # e.g., 16 x 128 x 2 = 4KB
        bytes_per_rhs_tile = m_rows * k_cols * dt_size  # each vlatchi chunk
        bytes_per_output_tile = m_rows * spec.mxu_width * 4  # M x 256 x 4 (f32)

        # Only count actual (non-padding) data movement.
        bytes_activation = num_vmatprep_mubr_actual * bytes_per_lhs_tile
        bytes_weight = num_vlatchi_actual * bytes_per_rhs_tile
        bytes_output = num_vmatmul_mubr * bytes_per_output_tile

        total_bytes = bytes_activation + bytes_weight + bytes_output

        # Calculate matrix multiplication shapes.
        # Execution shape: what the MXU actually executes (with padding).
        # Total M dimension = m_rows * num_vmatmul_mubr
        exec_m = m_rows * num_vmatmul_mubr
        exec_k = spec.mxu_width  # Always padded to 256
        exec_n = spec.mxu_width  # Always padded to 256
        execution_shape = (exec_m, exec_k, exec_n)

        # Effective shape: actual useful computation (without padding).
        # M dimension: total LHS rows loaded by vmatprep.mubr
        # All vmatprep.mubr operations contribute to effective M (they load real data).
        effective_m = m_rows * num_vmatprep_mubr_total

        # K dimension: from vector shape (e.g., 128 from vector<8x128x2>)
        # This is the actual K before padding to MXU width.
        effective_k = k_cols

        # N dimension: inferred from non-padding vlatchi operations.
        # Each vlatchi loads a portion of RHS columns.
        # For square weight matrices, N ≈ K (both are k_cols).
        effective_n = k_cols

        effective_shape = (effective_m, effective_k, effective_n)

        # Calculate FLOPs and utilization.
        execution_flops = 2 * exec_m * exec_k * exec_n
        effective_flops = 2 * effective_m * effective_k * effective_n
        utilization = effective_flops / execution_flops if execution_flops > 0 else 0.0

        # Use execution FLOPs for cost model (what the hardware actually executes).
        flops = int(execution_flops)

        # Compute time only (data assumed to be in VMEM, not HBM-bound).
        # Matmul operations execute on MXU with data already resident in
        # vector memory, so we don't model HBM bandwidth constraints here.
        # Memory movement (DMA operations) is modeled separately.
        if flops > 0:
            peak_flops = _dtype_peak_flops(spec, dtype)
            compute_time_s = flops / max(peak_flops, 1.0)
            compute_time_ns = compute_time_s * 1e9
        else:
            compute_time_ns = 0.0

        base_time_ns = compute_time_ns
        if base_time_ns <= 0.0:
            base_time_ns = 1.0

        time_ns = spec.dispatch_overhead_ns + base_time_ns

        metrics = PerformanceMetrics(
            flops=flops,
            bytes_accessed=int(total_bytes),
            estimated_time_ns=int(time_ns),
            category="Compute",
        )

        # Attach high-level MXU attributes.
        self.attributes.update(
            {
                "category": metrics.category,
                "mxu_macro": True,
                "execution_shape": execution_shape,  # (M, K, N) MXU executes
                "effective_shape": effective_shape,  # (M, K, N) useful computation
                "utilization": round(utilization, 4),  # useful / total FLOPs
                "dtype": dtype if dtype else "bf16",
            }
        )

        # Persist metrics on the event for downstream analysis.
        self.flops = metrics.flops
        self.bytes = metrics.bytes_accessed
        self.duration_ns = metrics.estimated_time_ns
        self.attributes.setdefault("category", metrics.category)
        return metrics


# Block event registry used by the simulator's unified stack.
BLOCK_EVENT_TYPES: List[Type[OpEvent]] = [
    LoopBlockEvent,
    IfBlockEvent,
    MatmulOpEvent,
]


def make_event_for_instruction(inst: Instruction) -> OpEvent:
    """
    Factory mapping Instruction.opcode to an execution event subclass.
    """
    op = inst.opcode

    if op == "vadd.f32":
        return AddOpEvent(inst=inst)
    if op == "vsub.f32":
        return SubOpEvent(inst=inst)
    if op == "vmul.f32":
        return MulOpEvent(inst=inst)
    if op == "vmax.f32":
        return MaxOpEvent(inst=inst)
    if op == "vexp.f32":
        return ExpOpEvent(inst=inst)
    if op == "vrecip.f32.approx":
        return RecipOpEvent(inst=inst)
    if op == "vector_load":
        return VectorLoadOpEvent(inst=inst)
    if op == "vector_load_slane_stride":
        return VectorLoadOpEvent(inst=inst)
    if op == "vector_store":
        return VectorStoreOpEvent(inst=inst)
    if op == "enqueue_dma":
        return EnqueueDMAOpEvent(inst=inst)
    if op == "dma_done":
        return DmaDoneOpEvent(inst=inst)
    
    for BlockCls in BLOCK_EVENT_TYPES:
        if BlockCls.is_start_inst(inst):
            return BlockCls(inst)

    return UnknownOpEvent(inst=inst)
