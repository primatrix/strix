# Strix

Strix is a small **static profiler** and **timeline simulator** for Mosaic TPU
LLO dumps (post-finalize-llo `.txt`/`.mlir`). It parses an LLO dump, runs a
simple dual-track (compute + DMA) simulator, prints a console summary, and can
emit a Chrome/Perfetto trace (`trace.json`).

## Requirements

- Python 3.9+ (no third-party dependencies)

## Quickstart

Run the CLI as a module from the directory *above* this repo (so Python can find
the `strix/` package):

```bash
python -m strix.cli path/to/post-finalize-llo.txt
```

This prints a short report to stdout and writes `trace.json` in the current
directory by default.

## Common Options

- Disable trace output:
  - `python -m strix.cli path/to/file.txt --trace-output ''`
- Override scalar SSA values (needed to resolve some DMA sizes / loop bounds):
  - `python -m strix.cli path/to/file.txt --arg %arg0=128 --arg %1237=10`
- Provide a default value for all unresolved `llo.sld` results:
  - `python -m strix.cli path/to/file.txt --default-sld-value 128`
- Exclude instructions by opcode:
  - `python -m strix.cli path/to/file.txt --exclude-instructions llo.nop llo.dbg`
- Dump the parsed tree (debugging):
  - `python -m strix.cli path/to/file.txt --dump-tree --tree-max-depth 8`

## Trace Viewing

The generated `trace.json` is compatible with Perfetto/Chrome trace viewers.
Open it in your preferred viewer to inspect the simulated Compute/DMA timelines
and stalls.
