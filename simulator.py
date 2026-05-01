from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .domain import Instruction, PerformanceMetrics
from .hardware import HardwareSpec
from .op_events import (
    BlockContext,
    OpEvent,
    OpKind,
    OpStream,
    group_boundary_ssa,
    make_event_for_instruction,
    MatmulOpEvent,
    UnknownOpEvent,
    RepeatedBlockEvent,
    PatternBlockEvent,
)
from .value_resolver import ValueResolver


@dataclass
class SimState:
    """
    Minimal mutable simulation state for a single Simulator.run().

    This keeps the dual clocks and SSA ready-times; the Simulator is
    responsible for advancing these when scheduling events.
    """

    vpu_clock: int = 0
    dma_clock: int = 0
    variable_ready_time: Dict[str, int] = field(default_factory=dict)
    # Track DMA operations: token -> (start_time, completion_time, bytes)
    dma_tokens: Dict[str, tuple[int, int, int]] = field(default_factory=dict)


class Simulator:
    """
    Pure simulator: expands control flow and emits a tree of OpEvent objects.

    Responsibilities:
      * interpret scf.for/scf.if by actually traversing their bodies,
      * use ValueResolver to resolve loop trip counts and DMA sizes,
      * for each dynamic instruction execution, call the cost model to get
        FLOPs/Bytes/time, and
      * maintain simple VPU/DMA clocks and SSA dependencies to assign
        start/end timestamps and stall events.

    It does NOT aggregate totals or build high-level reports; those are left
    to the analyzer.
    """

    def __init__(
        self,
        spec: HardwareSpec,
        arg_overrides: Optional[Dict[str, int]] = None,
    ):
        self.spec = spec

        # Seed scalar values for ValueResolver.
        initial_scalars: Dict[str, int] = {}
        if arg_overrides:
            for name, val in arg_overrides.items():
                key = name if name.startswith("%") else f"%{name}"
                initial_scalars[key] = int(val)
        self._initial_scalars = initial_scalars

        # Simulation state (initialised per run()).
        self.resolver: Optional[ValueResolver] = None
        self._state: Optional[SimState] = None

    # ------------------------------------------------------------------ public

    def run(self, root: Instruction) -> OpEvent:
        """
        Simulate the given Instruction tree and return a root OpEvent that
        contains all dynamic events as a subtree.
        """
        # Reset state.
        self.resolver = ValueResolver(self._initial_scalars)
        self._state = SimState()

        # The parser now returns a flat instruction stream in root.body.
        if root.is_container:
            inst_list: List[Instruction] = list(root.body)
        else:
            inst_list = [root]

        # Phase 1: build an OpEvent tree from the flat instruction stream.
        root_event = self._build_event_tree(inst_list)

        # Phase 3: schedule all executable events to assign timestamps and
        # insert STALL events where needed.
        self._schedule(root_event)

        return root_event

    def _build_event_tree(
        self,
        inst_list: List[Instruction],
    ) -> OpEvent:
        """
        Build an OpEvent tree over a flat instruction stream using a
        unified stack of container events.

        All block structures (scf.for / scf.if / MXU macros, etc.) are
        modeled via BlockEvent types registered in BLOCK_EVENT_TYPES.
        Containers are pushed onto a stack when their start instruction
        is seen; every non-structural instruction creates a leaf OpEvent
        that is attached to the current stack-top container via on_enter().
        When an end marker is encountered, the corresponding container is
        popped. No timing or cost modelling happens in this phase.
        """

        root = OpEvent(
            name="root",
            kind=OpKind.ROOT,
            stream=OpStream.CONTROL,
            attributes={},
        )

        # The stack always has at least the parent container on it.
        event_stack: List[OpEvent] = [root]

        # Build event tree while also feeding scalar instructions to the
        # ValueResolver for later trip-count/DMA-size evaluation.
        # Create context once with the event stack.
        context = BlockContext(event_stack=event_stack)

        for inst in inst_list:
            self.resolver.observe(inst)

            ev = make_event_for_instruction(inst)

            if ev.is_start_inst(inst, context):
                context.enter(ev)
            else:
                context.add(ev)

            # Close the current block if this instruction is its end marker.
            context.close_if_needed(inst)

        root.prepare(self.resolver)

        # Auto-compress repeated patterns in all containers
        self._auto_compress(root)

        return root

    def _auto_compress(self, root: OpEvent) -> None:
        """
        Automatically compress repeated patterns in all container events.

        This walks the tree and applies compression to any event with children,
        making compression universal without requiring explicit calls.
        """
        def compress_recursive(ev: OpEvent) -> None:
            # First compress children's children (bottom-up)
            for child in list(ev.children):
                compress_recursive(child)

            # Then compress this event's direct children
            if ev.children and len(ev.children) > 1:
                ev.compress_children(threshold=10, pair_threshold=5)

        compress_recursive(root)

    def _schedule(self, root: OpEvent) -> None:
        """
        Walk the prepared OpEvent tree, invoke per-op cost models and use
        `_schedule_event` to assign timestamps and insert STALL events.
        """
        assert self.resolver is not None
        assert self._state is not None
        resolver = self.resolver

        def _schedule_subtree(ev: OpEvent, parent: Optional[OpEvent]) -> None:
            # STALL events are produced by `_schedule_event`; they already
            # carry timing information and should not be re-scheduled.
            if ev.kind == OpKind.STALL:
                return

            # RepeatedBlockEvent and PatternBlockEvent: schedule children and aggregate
            if isinstance(ev, (RepeatedBlockEvent, PatternBlockEvent)):
                if ev.children:
                    for child in list(ev.children):
                        _schedule_subtree(child, ev)

                    # Aggregate timing from children
                    start = min(c.start_time_ns for c in ev.children) if ev.children else 0
                    end = max(c.end_time_ns for c in ev.children) if ev.children else 0
                    ev.start_time_ns = start
                    ev.end_time_ns = end
                    ev.duration_ns = max(end - start, 0)
                    ev.flops = sum(c.flops for c in ev.children)
                    ev.bytes = sum(c.bytes for c in ev.children)
                else:
                    # Empty block event
                    ev.start_time_ns = 0
                    ev.end_time_ns = 0
                    ev.duration_ns = 0
                    ev.flops = 0
                    ev.bytes = 0
                return

            # Any event that still has children after `prepare()` is treated
            # as一个容器：先调度 children，再把自己的时间/工作量聚合出来。
            if ev.children:
                for child in list(ev.children):
                    _schedule_subtree(child, ev)

                start = min(c.start_time_ns for c in ev.children) if ev.children else 0
                end = max(c.end_time_ns for c in ev.children) if ev.children else 0

                # For LOOP events, check if iterations were already expanded.
                # If expanded (trip_count <= 100): children already contain all iterations
                # If not expanded (trip_count > 100): scale metrics by trip_count
                if ev.kind == OpKind.LOOP:
                    expanded = ev.attributes.get("expanded", False)
                    trip_count = ev.attributes.get("trip_count", 1)

                    if not expanded:
                        # Large loop: scale single iteration by trip_count
                        ev.start_time_ns = start
                        ev.end_time_ns = start + (end - start) * trip_count
                        ev.duration_ns = (end - start) * trip_count
                        ev.flops = sum(c.flops for c in ev.children) * trip_count
                        ev.bytes = sum(c.bytes for c in ev.children) * trip_count
                    else:
                        # Small loop: already expanded, just aggregate
                        ev.start_time_ns = start
                        ev.end_time_ns = end
                        ev.duration_ns = max(end - start, 0)
                        ev.flops = sum(c.flops for c in ev.children)
                        ev.bytes = sum(c.bytes for c in ev.children)
                else:
                    # Regular container: just aggregate
                    ev.start_time_ns = start
                    ev.end_time_ns = end
                    ev.duration_ns = max(end - start, 0)
                    ev.flops = sum(c.flops for c in ev.children)
                    ev.bytes = sum(c.bytes for c in ev.children)

                ev.attributes.setdefault("category", "Control")

                # IMPORTANT: Advance clocks after container completes
                # Container events don't execute themselves, but they occupy time
                # The next sibling should not start before the container ends
                assert self._state is not None
                state = self._state

                # Advance VPU clock to the container's end time
                if ev.end_time_ns > state.vpu_clock:
                    state.vpu_clock = ev.end_time_ns

                # DO NOT advance DMA clock based on container end time!
                # DMA operations are async and should only be constrained by
                # the DMA engine's own schedule, not by VPU operations in the
                # same container. The DMA clock is advanced by enqueue_dma
                # operations directly.

                return

            # Leaf or block-style executable event.
            # Determine the underlying Instruction group that defines the
            # SSA boundary for this event.
            insts: List[Instruction] = []

            # Aggregated macros (e.g. MatmulOpEvent) expose an `instructions`
            # attribute; when present we treat that as the full group.
            inst_group = getattr(ev, "instructions", None)
            if isinstance(inst_group, list) and inst_group:
                insts = list(inst_group)
            else:
                inst = getattr(ev, "inst", None)
                if isinstance(inst, Instruction):
                    insts = [inst]

            # Invoke the per-event cost model.
            metrics = ev.exec(self.spec, resolver)  # type: ignore[arg-type]

            # Compute SSA boundary for dependency scheduling.
            if insts:
                inputs, outputs = group_boundary_ssa(insts)
            else:
                inputs, outputs = [], []

            stall_ev = self._schedule_event(ev, metrics, inputs, outputs)
            if stall_ev is not None and parent is not None:
                idx = parent.children.index(ev)
                parent.children.insert(idx, stall_ev)

        _schedule_subtree(root, None)

    def _schedule_event(
        self,
        ev: OpEvent,
        metrics: PerformanceMetrics,
        inputs: List[str],
        outputs: List[str],
    ) -> Optional[OpEvent]:
        """
        Given a costed event (ev + metrics) and its SSA boundary, assign
        start/end timestamps, emit any necessary Stall event, and advance
        the simulation clocks / ready-times.
        """
        assert self._state is not None
        state = self._state
        stall_event: Optional[OpEvent] = None

        # Import here to avoid circular dependency
        from .op_events import EnqueueDMAOpEvent, DmaDoneOpEvent

        # Special handling for async DMA operations
        if isinstance(ev, EnqueueDMAOpEvent):
            return self._schedule_enqueue_dma(ev, metrics, inputs, outputs)
        elif isinstance(ev, DmaDoneOpEvent):
            return self._schedule_dma_done(ev, metrics, inputs, outputs)

        # Determine execution stream and current clock from the metrics.
        is_dma = metrics.category == "Memory"
        stream = OpStream.DMA if is_dma else OpStream.VPU
        current_clock = state.dma_clock if is_dma else state.vpu_clock

        # SSA dependencies: event cannot start until all inputs are ready.
        max_dep = 0
        for tok in inputs:
            ready_at = state.variable_ready_time.get(tok, 0)
            if ready_at > max_dep:
                max_dep = ready_at
        actual_start = max(current_clock, max_dep)

        # Emit a Stall event if VPU waits on dependencies.
        if (not is_dma) and (actual_start > current_clock):
            stall_ns = actual_start - current_clock
            stall_event = OpEvent(
                name="STALL",
                kind=OpKind.STALL,
                flops=0,
                bytes=0,
                duration_ns=stall_ns,
                start_time_ns=current_clock,
                end_time_ns=actual_start,
                stream=OpStream.VPU,
                attributes={"reason": f"Waiting for {inputs}", "ssa_inputs": list(inputs), "ssa_outputs": []},
                children=[],
            )
            state.vpu_clock = actual_start

        # Fill in the main event's timing and metrics.
        end_time = actual_start + metrics.estimated_time_ns
        ev.stream = stream
        ev.flops = metrics.flops
        ev.bytes = metrics.bytes_accessed
        ev.duration_ns = metrics.estimated_time_ns
        ev.start_time_ns = actual_start
        ev.end_time_ns = end_time
        ev.attributes.setdefault("category", metrics.category)
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)

        # Advance clocks and mark outputs as ready.
        if is_dma:
            state.dma_clock = end_time
        else:
            state.vpu_clock = end_time

        for tok in outputs:
            state.variable_ready_time[tok] = end_time

        return stall_event

    def _schedule_enqueue_dma(
        self,
        ev: OpEvent,
        metrics: PerformanceMetrics,
        inputs: List[str],
        outputs: List[str],
    ) -> Optional[OpEvent]:
        """
        Schedule enqueue_dma: starts async DMA transfer.

        - DMA runs on DMA engine, doesn't block VPU
        - Track DMA token for later dma_done sync
        """
        assert self._state is not None
        state = self._state

        # DMA can start as soon as inputs are ready and DMA engine is free
        max_dep = 0
        for tok in inputs:
            ready_at = state.variable_ready_time.get(tok, 0)
            if ready_at > max_dep:
                max_dep = ready_at

        # DMA starts when both dependencies are ready and DMA engine is free
        dma_start = max(state.dma_clock, max_dep)
        dma_end = dma_start + metrics.estimated_time_ns

        # Schedule on DMA track
        ev.stream = OpStream.DMA
        ev.flops = 0
        ev.bytes = metrics.bytes_accessed
        ev.duration_ns = metrics.estimated_time_ns
        ev.start_time_ns = dma_start
        ev.end_time_ns = dma_end
        ev.attributes.setdefault("category", "Memory")
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)

        # Advance DMA clock
        state.dma_clock = dma_end

        # Track DMA token (output is __dma_token_N)
        if outputs:
            dma_token = outputs[0]
            dma_size = ev.attributes.get("dma_size", 0)
            state.dma_tokens[dma_token] = (dma_start, dma_end, dma_size)
            ev.attributes["dma_token"] = dma_token
            ev.attributes["dma_completion_time"] = dma_end
            # Mark token as ready immediately (it's a handle, not data)
            # But don't block VPU - token is just for tracking
            state.variable_ready_time[dma_token] = dma_start

        return None

    def _schedule_dma_done(
        self,
        ev: OpEvent,
        metrics: PerformanceMetrics,
        inputs: List[str],
        outputs: List[str],
    ) -> Optional[OpEvent]:
        """
        Schedule dma_done: wait for DMA completion.

        - Check if corresponding DMA has completed
        - If not, VPU stalls until DMA finishes
        - If yes, returns immediately
        """
        assert self._state is not None
        state = self._state
        stall_event: Optional[OpEvent] = None

        # Find DMA token (last input is __dma_token_N)
        dma_token = None
        if inputs:
            # Last input should be the DMA token
            for inp in reversed(inputs):
                if inp.startswith('__dma_token_'):
                    dma_token = inp
                    break

        # Wait for input dependencies (addresses, etc.)
        max_dep = 0
        for tok in inputs:
            if tok != dma_token:  # Don't wait for token itself
                ready_at = state.variable_ready_time.get(tok, 0)
                if ready_at > max_dep:
                    max_dep = ready_at
        actual_start = max(state.vpu_clock, max_dep)

        # Check if DMA has completed
        dma_completion_time = 0
        if dma_token and dma_token in state.dma_tokens:
            start_time, completion_time, size = state.dma_tokens[dma_token]
            dma_completion_time = completion_time
            ev.attributes["dma_token"] = dma_token
            ev.attributes["dma_start_time"] = start_time
            ev.attributes["dma_completion_time"] = completion_time
            ev.attributes["dma_size"] = size

        # If DMA not done yet, VPU must stall
        if dma_completion_time > actual_start:
            stall_ns = dma_completion_time - actual_start
            stall_event = OpEvent(
                name="DMA_STALL",
                kind=OpKind.STALL,
                flops=0,
                bytes=0,
                duration_ns=stall_ns,
                start_time_ns=actual_start,
                end_time_ns=dma_completion_time,
                stream=OpStream.VPU,
                attributes={"reason": f"Waiting for DMA {dma_token}", "ssa_inputs": list(inputs), "ssa_outputs": []},
                children=[],
            )
            actual_start = dma_completion_time
            state.vpu_clock = dma_completion_time

        # dma_done itself takes minimal time
        end_time = actual_start + 1
        ev.stream = OpStream.VPU
        ev.flops = 0
        ev.bytes = 0
        ev.duration_ns = 1
        ev.start_time_ns = actual_start
        ev.end_time_ns = end_time
        ev.attributes.setdefault("category", "Overhead")
        ev.attributes["ssa_inputs"] = list(inputs)
        ev.attributes["ssa_outputs"] = list(outputs)

        state.vpu_clock = end_time

        return stall_event
