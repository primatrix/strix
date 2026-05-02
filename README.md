# Strix

Strix is a **static profiler** and **timeline simulator** for TPU Mosaic/Pallas
LLO dumps (`post-finalize-llo` `.txt`/`.mlir`). It parses an LLO dump, runs a
dual-clock (VPU + DMA) simulator, analyzes stalls and bottlenecks, and prints a
console summary. It can also emit Chrome/Perfetto-compatible trace JSON and
Graphviz DOT dataflow graphs.

## Requirements

- Python 3.12+

Core analysis has **no third-party dependencies** (standard library only).

The optional `tpu` extra (for the `import` subcommand) needs JAX, libtpu,
protobuf, and `google-cloud-storage`:

```bash
pip install -e ".[tpu]"
```

## Quickstart

```bash
# Implicit analyze (backward-compatible)
python -m strix path/to/post-finalize-llo.txt

# Explicit analyze
python -m strix analyze path/to/post-finalize-llo.txt
```

Both `python -m strix` and `python -m strix.cli` are equivalent entry points.

This prints a performance summary to stdout and writes `trace.json` in the
current directory.

## Commands

Strix provides three subcommands:

| Command | Description |
|---|---|
| `analyze` | Parse and simulate an LLO dump. **Default** — if the first positional arg doesn't match any subcommand, `analyze` is prepended automatically. |
| `analyze-bundles` | Parse a `*-final_bundles.txt` file and map VLIW bundles to Pallas source lines. |
| `import` | Deploy a kernel to a GKE TPU pod, run a benchmark, and download IR dumps. |

---

### `analyze` — Parse and simulate an LLO dump

```
python -m strix analyze [OPTIONS] PATH
```

| Argument | Required | Description |
|---|---|---|
| `PATH` | yes | Path to a `post-finalize-llo` `.txt` or `.mlir` file |

| Option | Default | Description |
|---|---|---|
| `-t`, `--trace-output` | `trace.json` | Output path for Chrome trace JSON. Set to `''` to disable. |
| `--arg`, `--arg-override` | — | Override scalar SSA values, e.g. `--arg %arg0=128 --arg %1237=10`. Needed to resolve DMA sizes and loop bounds. |
| `--default-sld-value` | — | Default value for all unresolved `llo.sld` results, e.g. `--default-sld-value 128`. |
| `--exclude-instructions` | — | Exclude instructions by opcode, e.g. `--exclude-instructions llo.nop llo.dbg`. |
| `--dump-tree` | `False` | Print the parsed Instruction tree to stdout before simulating. |
| `--tree-max-depth` | — | Maximum depth when printing the Instruction/OpEvent tree. |
| `--dataflow-output` | — | Output path for Graphviz DOT dataflow graph. Render with: `dot -Tsvg output.dot -o output.svg` |

**Examples:**

```bash
# Basic analysis
python -m strix analyze dump.txt

# Override SSA args and disable trace
python -m strix analyze dump.txt --arg %arg0=128 --trace-output ''

# Generate dataflow graph
python -m strix analyze dump.txt --dataflow-output graph.dot
dot -Tsvg graph.dot -o graph.svg

# Debug: dump parsed tree with limited depth
python -m strix analyze dump.txt --dump-tree --tree-max-depth 5
```

The console output includes:
- **Overview**: total instructions, FLOPs, bytes, simulated time, VPU/DMA utilization
- **Bottleneck**: classification (Compute / Memory / Latency / Balanced) with stall ratio
- **Instruction mix**: breakdown by operation category

---

### `analyze-bundles` — Map VLIW bundles to source lines

```
python -m strix analyze-bundles [OPTIONS] PATH
```

| Argument | Required | Description |
|---|---|---|
| `PATH` | yes | Path to a `*-final_bundles.txt` file |

| Option | Description |
|---|---|
| `--json OUTPUT` | Write structured JSON to `OUTPUT` instead of console table. |
| `--line N` | Filter to source locations that include line `N`. |
| `--source-root DIR` | Local directory matching the TPU pod source paths, for inline code display. |

**Examples:**

```bash
# Console table
python -m strix analyze-bundles final_bundles.txt

# Filter to a specific source line
python -m strix analyze-bundles final_bundles.txt --line 42

# Show source code alongside bundle mappings
python -m strix analyze-bundles final_bundles.txt --source-root ./src

# Export as JSON
python -m strix analyze-bundles final_bundles.txt --json bundles.json
```

---

### `import` — Benchmark a kernel on TPU and download IR dumps

```
python -m strix import [OPTIONS] KERNEL
```

| Argument | Required | Description |
|---|---|---|
| `KERNEL` | yes | Kernel module path, e.g. `kernels.chunk_kda_fwd` |

| Option | Default | Description |
|---|---|---|
| `--shape` | (required) | Comma-separated shape dimensions, e.g. `1,2048,4,128,128` |
| `--chunk-size` | — | Chunk size for the kernel. |
| `--tpu-type` | `v7x` | TPU type (e.g. `v6e`, `v7x`). |
| `--tpu-topology` | `2x2x1` | TPU topology (e.g. `2x2x1`, `4x4x4`). |

Requires the `[tpu]` extras and authenticated GKE access.

**Example:**

```bash
python -m strix import kernels.chunk_kda_fwd \
    --shape 1,2048,4,128,128 \
    --tpu-type v7x \
    --tpu-topology 2x2x1
```

This deploys a K8s Job to the GKE cluster, runs the kernel with the given shape,
collects HLO/LLO/Mosaic IR dumps, and downloads them locally for further
analysis with `strix analyze`.

---

## Trace Viewing

The `--trace-output` file (default: `trace.json`) is compatible with
[Perfetto](https://ui.perfetto.dev) and Chrome `chrome://tracing`. The trace has
four tracks:

| TID | Track |
|---|---|
| 1 | Compute (VPU) |
| 2 | DMA |
| 3 | Stall |
| 4 | Control Flow |

Open the file in Perfetto to inspect simulated compute/DMA timelines, stall
regions, and loop nesting structure.

## Dataflow Graph

Use `--dataflow-output` to export a Graphviz DOT file showing the SSA dataflow
dependency graph. The graph includes:

- **Hardware clusters**: VPU and DMA operations grouped by execution unit
- **Loop subgraphs**: nested loop blocks shown as clusters
- **Time-bucket rank constraints**: same-T constraints displayed as rank annotations
- **Cross-stream edges**: dependencies between VPU and DMA highlighted

Render with Graphviz:

```bash
dot -Tsvg dataflow.dot -o dataflow.svg
```

## Architecture

```
LLO dump (.txt/.mlir)
        │
        ▼
   [Parser]          regex-based → Instruction tree
        │
        ▼
   [Simulator]       dual-clock (VPU/DMA), SSA resolution, loop expansion
        │
        ▼
   [Analyzer]        bottleneck classification, stall ratio, instruction mix
        │
        ├──► Console summary (always)
        ├──► Chrome trace JSON (--trace-output)
        └──► Graphviz DOT (--dataflow-output)
```

The hardware model defaults to v6e-class TPU specs (918 TFLOPS BF16, 32 GB HBM,
1600 GB/s bandwidth) and can be customized in `strix/hardware.py`.
