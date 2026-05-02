Profile a Pallas kernel on remote TPU and download all IR dumps (HLO, LLO, Mosaic) to local.

## Usage

```
/profile-kernel <config.yaml>
```

The argument `$ARGUMENTS` is the path to a YAML config file.

## Config File Format

```yaml
kernel: kernels.fused_moe          # Python module path (required)
shape: "1,2048,4,128,128"          # Comma-separated shape (required)
chunk_size: 128                    # Optional
tpu_type: v7x                     # Default: v7x
tpu_topology: 2x2x1               # Default: 2x2x1
```

## Instructions

1. Read the YAML config file at the path provided in `$ARGUMENTS`. If no path is provided, search for `*.profile.yaml` files in the repo and ask the user which one to use.

2. Parse the YAML and extract:
   - `kernel` (required) — Python module path
   - `shape` (required) — comma-separated dimensions
   - `chunk_size` (optional)
   - `tpu_type` (default: `v7x`)
   - `tpu_topology` (default: `2x2x1`)

3. Validate that `kernel` and `shape` are present. If missing, report the error and stop.

4. Build and run the benchmark command:
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
   - The job name
   - The location of IR dumps: `benchmark_results/<job-name>/ir_dumps/{hlo,llo,mosaic}`
   - Suggest running `strix analyze` on the LLO files if they want performance analysis

## Error Handling

- If the config file doesn't exist, report the error clearly.
- If `run_benchmark.sh` fails, show the last 50 lines of output and suggest checking `kubectl` connectivity or TPU node availability.
