# Design: `/profile-kernel` Claude Code Skill

**Date:** 2026-05-02
**Status:** Approved

## Problem

During daily Pallas kernel development, getting LLO/HLO/Mosaic IR dumps requires manually assembling `run_benchmark.sh` arguments. A Claude Code slash command would streamline this to a single invocation with a YAML config file.

## Design

### Skill Definition

A repository-level Claude Code command at `.claude/commands/profile-kernel.md`.

The skill:
1. Reads a YAML config file specified by the user
2. Calls `bash scripts/run_benchmark.sh` with the appropriate arguments
3. Reports the local path where IR dumps were extracted

### Config File Format

YAML files (e.g., `kernels/fused_moe.profile.yaml`):

```yaml
kernel: kernels.fused_moe          # Python module path (required)
shape: "1,2048,4,128,128"          # Comma-separated shape (required)
chunk_size: 128                    # Optional
tpu_type: v7x                     # Default: v7x
tpu_topology: 2x2x1               # Default: 2x2x1
```

### Execution Flow

```
User: /profile-kernel kernels/fused_moe.profile.yaml
  -> Skill reads YAML
  -> bash scripts/run_benchmark.sh kernels.fused_moe \
       --shape 1,2048,4,128,128 --chunk-size 128
  -> K8s Job created, runs on TPU, uploads to GCS
  -> run_benchmark.sh downloads tarball, extracts to benchmark_results/
  -> Skill reports: "IR dumps at benchmark_results/<job-name>/ir_dumps/{hlo,llo,mosaic}"
```

### What This Does NOT Do

- No new Python code
- No modifications to `run_benchmark.sh` or `cli.py`
- No automatic `strix analyze` (user runs manually)

### File Placement

```
.claude/commands/profile-kernel.md   # The skill definition
```
