from __future__ import annotations

from typing import Dict, List, Optional, Set

from .domain import Instruction


class ValueResolver:
    """
    Lightweight scalar value tracker used by the simulator.

    It maintains a map from SSA names (e.g. "%1198") to integer constants,
    and knows how to derive:
      * loop trip counts for scf.for,
      * DMA sizes for enqueue_dma (from the last scalar operand).

    The intent is that this class is easy to extend as we encounter more
    scalar ops that participate in loop bound / DMA setup computation.
    """

    def __init__(self, initial_scalars: Optional[Dict[str, int]] = None) -> None:
        # Map SSA name -> known integer value (when statically resolvable).
        self.scalars: Dict[str, int] = dict(initial_scalars or {})
        # Keep track of overridden values that should not be changed
        self._overrides: set = set(initial_scalars.keys()) if initial_scalars else set()

    # ------------------------------------------------------------------ helpers

    def resolve_token(self, tok: str) -> Optional[int]:
        """
        Resolve a token to an integer, if possible.

        Tokens starting with '%' are treated as SSA names and looked up
        in the scalar map. Other tokens are interpreted as integer literals.
        """
        tok = tok.strip()
        if not tok:
            return None
        if tok.startswith("%"):
            return self.scalars.get(tok)
        try:
            return int(tok)
        except ValueError:
            return None

    # ---------------------------------------------------------------- observe()

    def observe(self, inst: Instruction) -> None:
        """
        Update the scalar map based on one instruction.

        We only handle a narrow set of ops that are known to participate in
        loop bound / DMA size computation. This is intentionally conservative
        and can be extended over time.
        """
        op = inst.opcode

        # Integer constants.
        if op in ("constant", "arith.constant", "llo.constant"):
            val = inst.attributes.get("value")
            if val is None or not inst.outputs:
                return

            # Don't override user-specified values
            output_name = inst.outputs[0]
            if output_name in self._overrides:
                return

            try:
                self.scalars[output_name] = int(val)
            except (TypeError, ValueError):
                pass
            return

        # Basic integer arithmetic used in loop / DMA setup.
        # Support both with and without llo. prefix
        arithmetic_ops = {
            "sadd.s32", "llo.sadd.s32",
            "ssub.s32", "llo.ssub.s32",
            "smul.u32", "llo.smul.u32",
            "smin.s32", "llo.smin.s32",
            "smax.s32", "llo.smax.s32",
        }
        if op in arithmetic_ops:
            if len(inst.inputs) < 2 or not inst.outputs:
                return
            lhs_tok, rhs_tok = inst.inputs[0], inst.inputs[1]
            lhs = self.resolve_token(lhs_tok)
            rhs = self.resolve_token(rhs_tok)
            if lhs is None or rhs is None:
                return

            # Normalize opcode (remove llo. prefix if present)
            op_normalized = op.replace("llo.", "")

            if op_normalized == "sadd.s32":
                val = lhs + rhs
            elif op_normalized == "ssub.s32":
                val = lhs - rhs
            elif op_normalized == "smul.u32":
                val = lhs * rhs
            elif op_normalized == "smin.s32":
                val = min(lhs, rhs)
            elif op_normalized == "smax.s32":
                val = max(lhs, rhs)
            else:
                return

            self.scalars[inst.outputs[0]] = val
            return

        # llo.sld: scalar load from argument memref
        # Format: %result = llo.sld %arg + %offset : type
        # For static analysis, we can't resolve this without knowing the
        # memory content, so we rely on user-provided overrides.
        if op == "sld" or op == "llo.sld":
            # We can't resolve llo.sld statically, but we record the dependency
            # so we can report missing values later.
            # Users must provide override values for the result via --arg
            pass

        # Additional ops (sdiv/srem/etc.) can be added here if needed.

    # ---------------------------------------------------------- higher-level API

    def trip_count_for(self, loop_inst: Instruction) -> Optional[int]:
        """
        Compute a static trip count for an scf.for, when possible.

        The parser is expected to have stored the lb/ub/step SSA tokens in
        attributes:
            for_lb, for_ub, for_step
        """
        lb_tok = loop_inst.attributes.get("for_lb")
        ub_tok = loop_inst.attributes.get("for_ub")
        step_tok = loop_inst.attributes.get("for_step")
        if lb_tok is None or ub_tok is None or step_tok is None:
            return None

        lb = self.resolve_token(str(lb_tok))
        ub = self.resolve_token(str(ub_tok))
        step = self.resolve_token(str(step_tok))
        if lb is None or ub is None or step in (None, 0):
            return None

        trip = (ub - lb) // step
        if trip < 0:
            return 0
        return trip

    def dma_size_for(self, dma_inst: Instruction) -> Optional[int]:
        """
        Best-effort extraction of DMA size in bytes for an enqueue_dma.

        By convention in these LLO dumps, the *last* scalar operand of
        llo.enqueue_dma is the total size in bytes. When that operand is an
        SSA value and has been resolved to a constant, we return it.
        """
        if not dma_inst.inputs:
            return None
        len_tok = dma_inst.inputs[-1]
        return self.resolve_token(len_tok)

    def find_unresolved_dependencies(
        self,
        token: str,
        all_instructions: List[Instruction],
        visited: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Find all unresolved SSA dependencies for a given token.

        Returns a set of SSA names that need to be provided by the user.
        """
        if visited is None:
            visited = set()

        if token in visited:
            return set()
        visited.add(token)

        # If already resolved, no dependencies
        if self.resolve_token(token) is not None:
            return set()

        # Find the instruction that defines this token
        defining_inst = None
        for inst in all_instructions:
            if token in inst.outputs:
                defining_inst = inst
                break

        if defining_inst is None:
            # Token not defined by any instruction, it's an unresolved dependency
            return {token}

        # Recursively check all inputs
        unresolved = set()
        for inp in defining_inst.inputs:
            unresolved.update(
                self.find_unresolved_dependencies(inp, all_instructions, visited)
            )

        # If any input is unresolved, this token is also unresolved
        if unresolved:
            return unresolved

        return set()

    def check_all_dependencies_resolvable(
        self,
        all_instructions: List[Instruction]
    ) -> tuple[bool, Set[str]]:
        """
        Check if all llo.sld results that might be needed are resolvable.

        Returns:
            (all_resolvable, unresolved_sld_results)
        """
        unresolved = set()

        for inst in all_instructions:
            if inst.opcode in ("sld", "llo.sld") and inst.outputs:
                result = inst.outputs[0]
                # Check if this result can be resolved
                if self.resolve_token(result) is None:
                    # This llo.sld result is not provided by user
                    unresolved.add(result)

        return (len(unresolved) == 0, unresolved)

