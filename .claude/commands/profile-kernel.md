Profile a Pallas kernel on remote TPU and download all IR dumps (HLO, LLO, Mosaic) to local. Supports two modes: full benchmark (deploy + run + collect) and compile-only (cross-compile for a target topology without running).

## Usage

```
/profile-kernel <config.yaml>
```

The argument `$ARGUMENTS` is the path to a YAML config file.

## Config File Format

```yaml
kernel: kernels.fused_moe          # Python module path (required)
shape: "1,2048,4,128,128"          # Comma-separated shape (required for benchmark mode)
chunk_size: 128                    # Optional
tpu_type: v7x                     # Default: v7x
tpu_topology: 2x2x1               # Default: 2x2x1
compile_only: false                # Default: false. If true, cross-compile only (no execution)
target_topology: 2x8x8             # Target topology for cross-compilation (used when compile_only: true)
```

## Instructions

1. Read the YAML config file at the path provided in `$ARGUMENTS`. If no path is provided, search for `*.profile.yaml` files in the repo and ask the user which one to use.

2. Parse the YAML and extract:
   - `kernel` (required) — Python module path
   - `shape` (required for benchmark mode) — comma-separated dimensions
   - `chunk_size` (optional)
   - `tpu_type` (default: `v7x`)
   - `tpu_topology` (default: `2x2x1`)
   - `compile_only` (default: `false`) — if true, use cross-compilation mode
   - `target_topology` (default: `2x8x8`) — target topology for cross-compilation

3. Validate required fields. If `compile_only` is false, `kernel` and `shape` must be present. If `compile_only` is true, `kernel` must be present.

4. **If `compile_only: true`** — run the cross-compilation script:
   ```bash
   python scripts/cross_compile.py \
     --topology <target_topology> \
     [--output-dir <output_dir>]
   ```

   This must run on a TPU VM. It will:
   - Create a virtual TPU mesh for the target topology using `jax.experimental.topologies`
   - Cross-compile the kernel with `compile_only=True` (returns kernel_fn + args without executing)
   - Lower and compile via libtpu to generate HLO/LLO/Mosaic IR
   - Dump IR files to `/tmp/cross_compile/<topology>/`

   **If `compile_only: false`** (default) — run the full benchmark:
   ```bash
   bash scripts/run_benchmark.sh <kernel> \
     --shape <shape> \
     [--chunk-size <chunk_size>] \
     [--tpu-type <tpu_type>] \
     [--tpu-topology <tpu_topology>]
   ```

   Run this from the repository root. This command will:
   - Create a K8s Job on the TPU cluster
   - Wait for it to complete
   - Download the results tarball from GCS
   - Extract to `benchmark_results/`

5. After completion, report to the user:
   - For **benchmark mode**: the job name, IR dump location (`benchmark_results/<job-name>/ir_dumps/{hlo,llo,mosaic}`), and suggest running `strix analyze` on the LLO files.
   - For **compile-only mode**: the target topology, IR dump location (`/tmp/cross_compile/<topology>/{hlo,llo,mosaic}`), compilation time, and suggest running `strix analyze` on the LLO files.

## Error Handling

- If the config file doesn't exist, report the error clearly.
- If `run_benchmark.sh` fails, show the last 50 lines of output and suggest checking `kubectl` connectivity or TPU node availability.
- If `cross_compile.py` fails, check that:
  - The script is running on a TPU VM (libtpu.so is required)
  - JAX >= 0.5.x is installed (needed for `jax.experimental.topologies`)
  - The target topology is valid (e.g. `2x8x8`, `2x2x1`)
