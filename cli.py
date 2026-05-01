from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .analyzer import PerformanceAnalyzer
from .dataflow import extract_dataflow
from .dataflow_exporter import DataFlowDotExporter
from .domain import Instruction
from .exporters import ChromeTraceExporter, ConsoleExporter, JsonSummaryExporter
from .hardware import HardwareSpec
from .op_events import OpEvent
from .parser import LLOParser
from .simulator import Simulator
from .value_resolver import ValueResolver

# Package root directory (used to resolve scripts/).
_PACKAGE_DIR = Path(__file__).resolve().parent


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    """Add the LLO analyze arguments to *parser*."""
    parser.add_argument(
        "path",
        help="Path to a post-finalize-llo .txt/.mlir file.",
    )
    parser.add_argument(
        "--trace-output",
        "-t",
        default="trace.json",
        help="Where to write Chrome trace JSON (set to '' to disable).",
    )
    parser.add_argument(
        "--arg-override",
        "--arg",
        dest="arg_override",
        action="append",
        default=None,
        help=(
            "Override scalar SSA values for analysis, e.g. %%arg0=128 or %%1237=10. "
            "Can be specified multiple times. Required for DMA size resolution."
        ),
    )
    parser.add_argument(
        "--default-sld-value",
        type=int,
        default=None,
        help=(
            "Default value for all llo.sld results that are not explicitly specified. "
            "Useful when there are many llo.sld instructions. Common values: 0, 128."
        ),
    )
    parser.add_argument(
        "--exclude-instructions",
        nargs='+',
        default=None,
        help=(
            "Exclude instructions."
        ),
    )
    parser.add_argument(
        "--dump-tree",
        action="store_true",
        help=(
            "Print the parsed Instruction tree to stdout before running the "
            "simulator (for debugging the parser)."
        ),
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=None,
        help="Optional maximum depth when printing the instruction tree.",
    )
    parser.add_argument(
        "--dataflow-output",
        default=None,
        help=(
            "Where to write Graphviz DOT dataflow graph. "
            "Render with: dot -Tsvg output.dot -o output.svg"
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Static profiler and timeline simulator for Mosaic TPU LLO dumps "
            "(post-finalize-llo .txt/.mlir)."
        )
    )

    sub = ap.add_subparsers(dest="subcommand")

    # -- import subcommand --
    import_p = sub.add_parser(
        "import",
        help="Import a kernel module, benchmark on TPU, and download IR dumps.",
    )
    import_p.add_argument(
        "kernel",
        help="Kernel module path (e.g. kernels.chunk_kda_fwd).",
    )
    import_p.add_argument(
        "--shape",
        required=True,
        help="Comma-separated shape dimensions (e.g. 1,2048,4,128,128).",
    )
    import_p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for the kernel.",
    )
    import_p.add_argument(
        "--tpu-type",
        default="v7x",
        help="TPU type (default: v7x).",
    )
    import_p.add_argument(
        "--tpu-topology",
        default="2x2x1",
        help="TPU topology (default: 2x2x1).",
    )

    # -- analyze (legacy) subcommand: registered but also reachable without
    #    the subcommand word for backward compatibility --
    analyze_p = sub.add_parser(
        "analyze",
        help="Analyze a post-finalize LLO file (default when a file path is given).",
    )
    _add_analyze_args(analyze_p)

    return ap


def _print_instruction_tree(
    inst: Instruction,
    indent: int = 0,
    max_depth: Optional[int] = None,
) -> None:
    """Pretty-print the Instruction tree for manual inspection."""
    # Depth is counted in container nesting levels (module / scf.for / ...).
    current_depth = indent // 2
    if max_depth is not None and current_depth > max_depth:
        if current_depth == (max_depth + 1):
            # Only print an ellipsis once for this subtree.
            print(" " * indent + "...")
        return

    prefix = " " * indent

    # Basic IO string: "%0, %1 = opcode(%arg0, %arg1)"
    pieces: List[str] = []
    if inst.outputs:
        pieces.append(", ".join(inst.outputs))
        pieces.append("=")
    pieces.append(inst.opcode)
    if inst.inputs:
        pieces.append("(" + ", ".join(inst.inputs) + ")")
    io_str = " ".join(pieces)

    # A few key attributes to help debugging without spamming the log.
    attrs = dict(inst.attributes or {})
    attr_parts = []
    for k, v in list(attrs.items())[:4]:
        attr_parts.append(f"{k}={v!r}")
    attr_str = ", ".join(attr_parts)

    line = io_str
    if attr_str:
        line += f" [{attr_str}]"
    if inst.location:
        line += f"  @{inst.location}"

    print(prefix + line)

    for child in inst.body:
        _print_instruction_tree(child, indent=indent + 2, max_depth=max_depth)


def _print_opevent_tree(
    event: OpEvent,
    indent: int = 0,
    max_depth: Optional[int] = None,
) -> None:
    """Pretty-print the OpEvent tree for manual inspection."""
    # Depth is counted in container nesting levels.
    current_depth = indent // 2
    if max_depth is not None and current_depth > max_depth:
        if current_depth == (max_depth + 1):
            # Only print an ellipsis once for this subtree.
            print(" " * indent + "...")
        return

    prefix = " " * indent

    # Basic info: name, kind, stream
    pieces: List[str] = []
    pieces.append(event.name)
    pieces.append(f"[{event.kind.value}]")

    # Timing info
    time_parts = []
    if event.start_time_ns > 0 or event.end_time_ns > 0:
        time_parts.append(f"start={event.start_time_ns}ns")
        time_parts.append(f"end={event.end_time_ns}ns")
        time_parts.append(f"dur={event.duration_ns}ns")

    # Performance metrics
    perf_parts = []
    if event.flops > 0:
        perf_parts.append(f"flops={event.flops}")
    if event.bytes > 0:
        perf_parts.append(f"bytes={event.bytes}")

    # Build the line
    line = " ".join(pieces)
    if time_parts:
        line += f" ({', '.join(time_parts)})"
    if perf_parts:
        line += f" <{', '.join(perf_parts)}>"

    # A few key attributes
    attrs = dict(event.attributes or {})
    attr_parts = []
    for k, v in list(attrs.items())[:4]:
        attr_parts.append(f"{k}={v!r}")
    attr_str = ", ".join(attr_parts)
    if attr_str:
        line += f" [{attr_str}]"

    print(prefix + line)

    for child in event.children:
        _print_opevent_tree(child, indent=indent + 2, max_depth=max_depth)


def _run_import(args: argparse.Namespace) -> None:
    """Handle the ``import`` subcommand."""
    cmd: List[str] = [
        "bash",
        str(_PACKAGE_DIR / "scripts" / "run_benchmark.sh"),
        args.kernel,
        "--shape",
        args.shape,
        "--tpu-type",
        args.tpu_type,
        "--tpu-topology",
        args.tpu_topology,
    ]
    if args.chunk_size is not None:
        cmd.extend(["--chunk-size", str(args.chunk_size)])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` (legacy) subcommand."""
    spec = HardwareSpec()
    parser = LLOParser()
    # Parse any %argX=value style overrides into a dict passed to the simulator.
    arg_overrides: Dict[str, int] = {}
    if args.arg_override:
        for item in args.arg_override:
            if not item:
                continue
            if "=" not in item:
                continue
            name, val_str = item.split("=", 1)
            name = name.strip()
            val_str = val_str.strip()
            if not name:
                continue
            if not name.startswith("%"):
                name = "%" + name
            try:
                val = int(val_str)
            except ValueError:
                continue
            arg_overrides[name] = val

    root = parser.parse_file(args.path, exclude_instructions=args.exclude_instructions)

    # Apply default value for all llo.sld results if specified
    if args.default_sld_value is not None:
        for inst in root.body:
            if inst.opcode in ("sld", "llo.sld") and inst.outputs:
                result = inst.outputs[0]
                # Only set if not already specified by user
                if result not in arg_overrides:
                    arg_overrides[result] = args.default_sld_value

    # Pre-check: verify that all required runtime values are provided
    # Create a temporary resolver to check dependencies
    temp_resolver = ValueResolver(arg_overrides or None)
    for inst in root.body:
        temp_resolver.observe(inst)

    all_resolvable, unresolved_sld = temp_resolver.check_all_dependencies_resolvable(root.body)

    if not all_resolvable:
        print("\n" + "=" * 70)
        print("ERROR: Missing required runtime values")
        print("=" * 70)
        print("\nSome llo.sld instructions load values from function arguments.")
        print("These values must be provided via --arg for static analysis.\n")

        # Find source arguments for these llo.sld results
        sld_info = {}
        for inst in root.body:
            if inst.opcode in ("sld", "llo.sld") and inst.outputs:
                result = inst.outputs[0]
                if result in unresolved_sld and inst.inputs:
                    source = inst.inputs[0]  # %arg3, etc.
                    sld_info[result] = source

        print(f"Values that need to be specified ({len(unresolved_sld)}):")
        print("(These are loaded from function arguments via llo.sld)\n")

        sorted_unresolved = sorted(unresolved_sld)
        for result in sorted_unresolved[:20]:  # Show first 20
            source = sld_info.get(result, "unknown")
            print(f"  {result} (from {source})")

        if len(sorted_unresolved) > 20:
            print(f"  ... and {len(sorted_unresolved) - 20} more")

        print("\nOption 1 - Provide all values individually:")
        if sorted_unresolved:
            print(f"  python -m strix.cli {args.path} \\")
            for i, result in enumerate(sorted_unresolved[:3]):
                value = "128" if i == 0 else "0"
                print(f"    --arg \"{result}={value}\" \\")
            if len(sorted_unresolved) > 3:
                print(f"    ... (and {len(sorted_unresolved) - 3} more values)")

        print("\nOption 2 - Use default value for all (simpler):")
        print(f"  python -m strix.cli {args.path} \\")
        print(f"    --default-sld-value 128")

        print("\nNote: You need to provide actual values based on your workload.")
        print("These values affect loop bounds, DMA sizes, and timing estimates.")
        print("=" * 70)
        raise SystemExit(1)

    simulator = Simulator(
        spec,
        arg_overrides=arg_overrides or None,
    )
    root_event = simulator.run(root)

    if args.dump_tree:
        print("\n=== OpEvent Tree (after simulation) ===")
        _print_opevent_tree(root_event, indent=0, max_depth=args.tree_max_depth)
        print("=== End OpEvent Tree ===\n")

    analyzer = PerformanceAnalyzer(root_event)
    report = analyzer.analyze()

    # Console summary is always emitted.
    ConsoleExporter().export(root_event, report, None)

    trace_path = args.trace_output or ""
    if trace_path:
        ChromeTraceExporter().export(root_event, report, trace_path)

    if args.dataflow_output:
        df_graph = extract_dataflow(root_event)
        DataFlowDotExporter().export(df_graph, args.dataflow_output)


def _get_subcommands(ap: argparse.ArgumentParser) -> set[str]:
    """Extract registered subcommand names from the parser."""
    for action in ap._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    return set()


def preprocess_argv(
    argv: Optional[List[str]] = None,
    ap: Optional[argparse.ArgumentParser] = None,
) -> List[str]:
    """Apply backward-compat: prepend ``analyze`` when no subcommand given."""
    if ap is None:
        ap = build_arg_parser()
    subcommands = _get_subcommands(ap)
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] not in subcommands and not raw[0].startswith("-"):
        raw = ["analyze"] + raw
    return raw


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(preprocess_argv(argv, ap))

    if args.subcommand == "import":
        _run_import(args)
    elif args.subcommand == "analyze":
        _run_analyze(args)
    else:
        ap.print_help()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
