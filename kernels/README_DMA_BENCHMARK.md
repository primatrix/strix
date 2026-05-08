# DMA Double-Buffer Load Benchmark

## Overview

`dma_double_buffer_load.py` is a standalone benchmark kernel that isolates and measures the DMA efficiency of loading expert weights from HBM to VMEM using double-buffering. This kernel extracts the weight loading pipeline pattern from the expert FFN kernel to profile pure DMA throughput without compute interference.

## Purpose

This benchmark helps answer:
- What is the raw DMA bandwidth for different weight tile shapes?
- How does double-buffering efficiency scale with different block sizes (bf, bd)?
- What is the DMA overhead for different numbers of loads (simulating expert iteration)?
- How do different hidden_size and intermediate_size configurations affect DMA performance?

## Usage

### Direct Execution

```bash
python kernels/dma_double_buffer_load.py [hidden_size] [intermediate_size] [num_loads]
```

Examples:
```bash
# Default Ling 2.6 shape (H=8192, I=2048, 64 loads)
python kernels/dma_double_buffer_load.py

# Custom shape
python kernels/dma_double_buffer_load.py 4096 2048 32

# Small test
python kernels/dma_double_buffer_load.py 1024 512 8
```

### Benchmark Runner Integration

The kernel follows the standard `kernel_fn` + `config` contract and can be used with `scripts/benchmark_runner.py`:

```bash
python scripts/benchmark_runner.py \
  --module strix.kernels.dma_double_buffer_load \
  --output-dir gs://your-bucket/dma-benchmark
```

### Programmatic Usage

```python
import jax.numpy as jnp
from strix.kernels.dma_double_buffer_load import dma_double_buffer_load

# Create weight matrix
w = jax.random.normal(key, (8192, 2048), dtype=jnp.bfloat16)

# Run benchmark with custom block sizes
result = dma_double_buffer_load(
    w,
    bf=2048,      # Intermediate dimension block size
    bd=1024,      # Hidden dimension block size
    num_loads=64  # Number of weight tiles to load
)

print(f"Checksum: {result}")
```

## Configuration

### Default Shape (config dict)
- `hidden_size`: 8192
- `intermediate_size`: 2048
- `num_loads`: 64
- `dtype`: bfloat16
- `weight_dtype`: bfloat16

### Block Size Parameters
- `bf`: Intermediate dimension block size (default: 2048)
- `bd`: Hidden dimension block size (default: 1024)

These control the tile size for each DMA transfer. Larger tiles mean fewer transfers but more VMEM usage.

## Implementation Details

### Double-Buffering Pattern

The kernel implements the same double-buffering strategy used in expert FFN:

1. **Prefetch**: Load first weight tile into buffer 0
2. **Loop**: For each subsequent load:
   - Wait for current buffer to be ready
   - Start loading next tile into the other buffer (ping-pong)
   - Consume current tile (read from VMEM, compute checksum)
   - Toggle buffer index (0 ↔ 1)

### DMA Pipeline

```
Buffer 0: [Load tile 0] -------- [Consume] [Load tile 2] -------- [Consume] ...
Buffer 1:                        [Load tile 1] -------- [Consume] [Load tile 3] ...
```

### Memory Layout

- **HBM Input**: Weight matrix `[hidden_size, intermediate_size]`
- **VMEM Scratch**: Double-buffer `[2, t_packing, bd_per_pack, bf]`
  - 2 buffers for ping-pong
  - t_packing = 2 for bfloat16 (packing factor)
  - bd_per_pack = bd / t_packing
- **Semaphores**: 2 DMA semaphores (one per buffer)

### Output

Returns a scalar float32 checksum computed by summing all loaded weight tiles. This forces VMEM reads and verifies that DMA actually happened.

## Shape Variations for Testing

Recommended test configurations:

| hidden_size | intermediate_size | num_loads | Description |
|-------------|-------------------|-----------|-------------|
| 8192 | 2048 | 64 | Ling 2.6 default (per-expert) |
| 8192 | 4096 | 64 | Larger intermediate dim |
| 4096 | 2048 | 32 | Smaller model |
| 16384 | 4096 | 128 | Larger model |
| 1024 | 512 | 8 | Quick test |

Block size variations:
- `bf=512, bd=512`: Small tiles, more transfers
- `bf=1024, bd=1024`: Medium tiles
- `bf=2048, bd=1024`: Default (matches expert FFN)
- `bf=2048, bd=2048`: Large tiles, fewer transfers

## Testing

Run structural tests (no TPU required):
```bash
PYTHONPATH=. python -m pytest tests/test_dma_double_buffer_load.py -v -k "not Importlib and not Execution and not ShapeVariations"
```

All tests (requires package installation):
```bash
PYTHONPATH=. python -m pytest tests/test_dma_double_buffer_load.py -v
```

## Performance Analysis

When profiling on TPU, look for:

1. **DMA Utilization**: What percentage of time is DMA active?
2. **Stall Cycles**: Are there gaps between DMA transfers?
3. **Bandwidth**: Actual bytes/sec vs theoretical peak
4. **Overlap**: How well does double-buffering hide DMA latency?

Compare against theoretical peak:
- TPU v7x HBM bandwidth: ~2.4 TB/s per chip
- Expected DMA bandwidth: ~1-2 TB/s (accounting for overhead)

## Related Files

- `kernels/expert_ffn.py`: Full expert FFN kernel (includes compute)
- `kernels/_fused_moe_impl.py`: Fused MoE kernel (upstream source)
- `tests/test_dma_double_buffer_load.py`: Test suite
- `scripts/benchmark_runner.py`: TPU benchmark harness
