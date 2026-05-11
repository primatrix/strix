# Double-Buffer Expert Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `kernels/double_buffer_expert.py` — a Pallas TPU kernel for one Ling 2.6 routed expert, following the §5.8 decode pipeline (B=1, persistent x, dual weight slots, low-priority W2 DMA, fp32 y_acc). Plumb `--num-tokens` through `benchmark_runner.py` and add a YAML config so the two recommended `(bt, bf)` benchmark pairs can be driven from CLI.

**Architecture:** Single standalone Pallas kernel with fixed shape `(d=8192, f=2048)` and two whitelisted `(num_tokens, bf)` pairs: `(256, 512)` and `(512, 256)`. Pipeline is Python-unrolled prologue + `for`-loop steady state + epilogue over `N_w = f/bf` weight tiles. Weights tile along f only; x and y_acc persist in VMEM for the full kernel invocation. Final write-back casts fp32 y_acc → bf16 via a dedicated VMEM staging buffer + async DMA.

**Tech Stack:** JAX, Pallas TPU (`jax.experimental.pallas.tpu`), bf16 weights/activations, fp32 accumulator, TPU v7x.

**Spec:** `docs/superpowers/specs/2026-05-11-double-buffer-expert-design.md`

---

## File Structure

- **Create:** `kernels/double_buffer_expert.py` — main kernel module (config, `kernel_fn`, `double_buffer_expert`, `_ref_expert_ffn`, `__main__`)
- **Create:** `tests/test_double_buffer_expert.py` — AST-only structural tests (no TPU)
- **Create:** `double_buffer_expert_ling2.6.yaml` — benchmark runner config
- **Modify:** `scripts/benchmark_runner.py` — add `--num-tokens` flag + plumbing in `parse_args` and `run_benchmark`
- **Modify:** `tests/test_benchmark_runner.py` — add tests for `--num-tokens` flag parsing + env var

---

## Task 1: Structural Tests for double_buffer_expert (TDD red phase)

**Files:**
- Create: `tests/test_double_buffer_expert.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_double_buffer_expert.py`:

```python
"""Structural tests for kernels/double_buffer_expert.py (no TPU required).

Mirrors tests/test_expert_ffn_kernel.py — AST-based checks that the module
exposes the correct public API per
docs/superpowers/specs/2026-05-11-double-buffer-expert-design.md.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_FILE = REPO_ROOT / "kernels" / "double_buffer_expert.py"


class TestFileExists:
    def test_kernel_file_exists(self):
        assert KERNEL_FILE.is_file(), "kernels/double_buffer_expert.py not found"


class TestModuleStructure:
    @pytest.fixture(autouse=True)
    def _parse_module(self):
        self.tree = ast.parse(KERNEL_FILE.read_text())

    def _func_names(self):
        return [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def _assign_names(self):
        names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
        return names

    def test_defines_kernel_fn(self):
        assert "kernel_fn" in self._func_names()

    def test_defines_double_buffer_expert(self):
        assert "double_buffer_expert" in self._func_names()

    def test_defines_ref_expert_ffn(self):
        assert "_ref_expert_ffn" in self._func_names()

    def test_defines_kernel_body(self):
        assert "_double_buffer_expert_kernel" in self._func_names()

    def test_defines_config(self):
        assert "config" in self._assign_names()


class TestConfig:
    @pytest.fixture(autouse=True)
    def _extract_config(self):
        source = KERNEL_FILE.read_text()
        tree = ast.parse(source)
        config_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "config":
                        config_node = node
                        break
        assert config_node is not None, "config assignment not found"
        module = ast.Module(body=[config_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, str(KERNEL_FILE), "exec")
        ns = {}
        exec(code, ns)
        self.config = ns["config"]

    def test_config_is_dict(self):
        assert isinstance(self.config, dict)

    def test_default_shape_has_ling_26_values(self):
        ds = self.config["default_shape"]
        assert ds["num_tokens"] == 256
        assert ds["hidden_size"] == 8192
        assert ds["intermediate_size"] == 2048

    def test_default_bf_is_512(self):
        assert self.config["bf"] == 512

    def test_dtype_is_bfloat16(self):
        assert self.config["dtype"] == "bfloat16"
        assert self.config["weight_dtype"] == "bfloat16"

    def test_act_fn_is_silu(self):
        assert self.config["act_fn"] == "silu"

    def test_tpu_type_is_v7x(self):
        assert self.config["tpu_type"] == "v7x"


class TestKernelFnSignature:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "double_buffer_expert", KERNEL_FILE,
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_kernel_fn_accepts_required_kwargs(self):
        sig = inspect.signature(self.module.kernel_fn)
        params = set(sig.parameters.keys())
        required = {
            "num_tokens", "hidden_size", "intermediate_size",
            "dtype", "weight_dtype", "act_fn", "bf",
        }
        missing = required - params
        assert not missing, f"kernel_fn missing kwargs: {sorted(missing)}"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
PYTHONPATH=. pytest tests/test_double_buffer_expert.py -v
```

Expected: Tests in `TestFileExists`, `TestModuleStructure`, `TestConfig`, `TestKernelFnSignature` all FAIL (module doesn't exist yet). Pytest should print collection errors for the import-based fixture and file-not-found assertions in other tests.

- [ ] **Step 1.3: Commit**

```bash
git add tests/test_double_buffer_expert.py
git commit -m "test: add structural tests for double_buffer_expert kernel (TDD red)"
```

---

## Task 2: Scaffold the Kernel Module (TDD green phase, body stubbed)

**Files:**
- Create: `kernels/double_buffer_expert.py`

- [ ] **Step 2.1: Create the module with config, ref impl, and stubbed kernel**

Create `kernels/double_buffer_expert.py`:

```python
"""Double-buffer expert FFN kernel — §5.8 decode pipeline for Ling 2.6.

Single routed expert, B=1, persistent x, dual weight slots, low-priority
W2 DMA, fp32 y_acc accumulator. Hard-coded for d=8192, f=2048, bf16.

Spec: docs/superpowers/specs/2026-05-11-double-buffer-expert-design.md
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._fused_moe_impl import activation_fn


_ALLOWED_CONFIGS = {(256, 512), (512, 256)}  # (num_tokens, bf) pairs from §5.8


config = {
    "default_shape": {
        "num_tokens": 256,
        "hidden_size": 8192,
        "intermediate_size": 2048,
    },
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "act_fn": "silu",
    "bf": 512,
    "tpu_type": "v7x",
    "tpu_topology": "2x2x1",
    "description": (
        "Double-buffer expert FFN — §5.8 B=1 decode "
        "(persistent x, dual W slots, low-pri W2)"
    ),
}


def _double_buffer_expert_kernel(*args, **kwargs):
    """Kernel body — implemented in Task 3."""
    raise NotImplementedError("Pallas kernel body implemented in Task 3")


def double_buffer_expert(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
    bf: int = 512,
) -> jax.Array:
    """Run the double-buffer expert FFN kernel.

    Shapes:
      tokens: (bt, d) bf16
      w1:     (d, f)  bf16
      w2:     (f, d)  bf16
      w3:     (d, f)  bf16
    Returns:
      output: (bt, d) bf16
    """
    bt, d = tokens.shape
    f_full = w1.shape[1]
    if d != 8192 or f_full != 2048:
        raise ValueError(
            f"Kernel hard-coded for d=8192, f=2048; got d={d}, f={f_full}"
        )
    if (bt, bf) not in _ALLOWED_CONFIGS:
        raise ValueError(
            f"Only (num_tokens, bf) in {sorted(_ALLOWED_CONFIGS)} supported; "
            f"got (num_tokens={bt}, bf={bf})"
        )
    if w3.shape != (d, f_full):
        raise ValueError(f"w3.shape={w3.shape} must be (d={d}, f={f_full})")
    if w2.shape != (f_full, d):
        raise ValueError(f"w2.shape={w2.shape} must be (f={f_full}, d={d})")

    # Body wired in Task 3.
    raise NotImplementedError("Pallas kernel wiring implemented in Task 3")


def _ref_expert_ffn(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    act_fn: str = "silu",
) -> jax.Array:
    """Pure-JAX fp32 reference: (silu(x @ W1) * (x @ W3)) @ W2."""
    gate = tokens.astype(jnp.float32) @ w1.astype(jnp.float32)
    up = tokens.astype(jnp.float32) @ w3.astype(jnp.float32)
    act = activation_fn(gate, up, act_fn)
    return (act @ w2.astype(jnp.float32)).astype(tokens.dtype)


def kernel_fn(
    num_tokens: int = 256,
    hidden_size: int = 8192,
    intermediate_size: int = 2048,
    dtype=jnp.bfloat16,
    weight_dtype=jnp.bfloat16,
    act_fn: str = "silu",
    bf: int = 512,
) -> Callable[[], jax.Array]:
    """Build random inputs and return a zero-arg closure calling the kernel.

    Contract matches scripts/benchmark_runner.py (see expert_ffn.py).
    """
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
    w1 = jax.random.normal(k2, (hidden_size, intermediate_size), dtype=weight_dtype)
    w2 = jax.random.normal(k3, (intermediate_size, hidden_size), dtype=weight_dtype)
    w3 = jax.random.normal(k4, (hidden_size, intermediate_size), dtype=weight_dtype)

    def run():
        return double_buffer_expert(
            tokens, w1, w2, w3, act_fn=act_fn, bf=bf,
        )

    return run
```

- [ ] **Step 2.2: Run structural tests — they should now pass**

```bash
PYTHONPATH=. pytest tests/test_double_buffer_expert.py -v
```

Expected: all tests PASS. (Note: `TestKernelFnSignature` imports the module, which succeeds because our import chain `_fused_moe_impl.activation_fn` already works — `expert_ffn.py` imports the same symbol.)

- [ ] **Step 2.3: Verify importability without TPU**

```bash
PYTHONPATH=. python -c "from kernels import double_buffer_expert; print(double_buffer_expert.config['bf'])"
```

Expected output: `512`

- [ ] **Step 2.4: Commit**

```bash
git add kernels/double_buffer_expert.py
git commit -m "feat: scaffold double_buffer_expert kernel module (stubbed body)"
```

---

## Task 3: Implement the Pallas Kernel Body

**Files:**
- Modify: `kernels/double_buffer_expert.py`

**Context:** This task implements the §5.8 pipeline. The kernel body takes HBM refs for inputs/output and VMEM scratch refs for buffers + semaphores. The Python-unrolled prologue + steady-state `for` + epilogue gives the compiler a complete view of DMA/MXU scheduling. `compute_tile` defers `wait_fetch_w2(slot)` until after the gate/up dots so W2's DMA overlaps with MXU (the §5.8 "W2 low priority" pattern).

- [ ] **Step 3.1: Replace the stubbed `_double_buffer_expert_kernel` with the full implementation**

Replace the stub function in `kernels/double_buffer_expert.py` with:

```python
def _double_buffer_expert_kernel(
    tokens_hbm,
    w1_hbm,
    w2_hbm,
    w3_hbm,
    output_hbm,
    # Scratch buffers (VMEM).
    b_w1_x2_vmem,
    b_w3_x2_vmem,
    b_w2_x2_vmem,
    b_x_vmem,
    b_y_acc_vmem,
    b_y_out_vmem,
    # Semaphores.
    weight_sems,
    x_sem,
    y_out_sem,
    *,
    act_fn: str,
    bf: int,
    intermediate_size: int,
):
    """Pallas kernel body — §5.8 B=1 decode pipeline."""
    n_w = intermediate_size // bf  # 4 (bf=512) or 8 (bf=256)

    # -- DMA helpers --
    def start_load_x():
        pltpu.make_async_copy(
            src_ref=tokens_hbm, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).start()

    def wait_load_x():
        pltpu.make_async_copy(
            src_ref=b_x_vmem, dst_ref=b_x_vmem, sem=x_sem.at[0],
        ).wait()

    def start_fetch_w1(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).start()

    def wait_fetch_w1(slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 0],
        ).wait()

    def start_fetch_w3(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[:, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).start()

    def wait_fetch_w3(slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 1],
        ).wait()

    def start_fetch_w2(slot, tile_idx):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).start()

    def wait_fetch_w2(slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=weight_sems.at[slot, 2],
        ).wait()

    # -- Compute function (W2 wait deferred inside) --
    def compute_tile(slot, is_first_tile):
        """Gate/up compute (W1, W3), then wait W2 (low-pri overlap), then accumulate."""
        x = b_x_vmem[...]
        w1 = b_w1_x2_vmem[slot]
        w3 = b_w3_x2_vmem[slot]
        gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)
        up = jnp.dot(x, w3, preferred_element_type=jnp.float32)
        act_up = activation_fn(gate, up, act_fn)

        wait_fetch_w2(slot)  # deferred — overlaps with gate/up MXU

        w2 = b_w2_x2_vmem[slot]
        partial = jnp.dot(act_up, w2, preferred_element_type=jnp.float32)
        if is_first_tile:
            b_y_acc_vmem[...] = partial
        else:
            b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial

    # -- Prologue (§5.8 stages L1..C2) --
    start_load_x()
    start_fetch_w1(0, 0)
    start_fetch_w3(0, 0)
    start_fetch_w2(0, 0)
    wait_load_x()
    wait_fetch_w1(0)
    wait_fetch_w3(0)

    if n_w >= 2:
        start_fetch_w1(1, 1)
        start_fetch_w3(1, 1)
        start_fetch_w2(1, 1)

    compute_tile(slot=0, is_first_tile=True)

    # -- Steady state: Python for-loop over tile in [1, n_w - 1) --
    for tile in range(1, n_w - 1):
        slot = tile % 2
        next_slot = 1 - slot
        start_fetch_w1(next_slot, tile + 1)
        start_fetch_w3(next_slot, tile + 1)
        start_fetch_w2(next_slot, tile + 1)
        wait_fetch_w1(slot)
        wait_fetch_w3(slot)
        compute_tile(slot, is_first_tile=False)

    # -- Epilogue: last tile, no more loads --
    if n_w >= 2:
        last_slot = (n_w - 1) % 2
        wait_fetch_w1(last_slot)
        wait_fetch_w3(last_slot)
        compute_tile(last_slot, is_first_tile=False)

    # -- Write-back: y_acc (fp32) → b_y_out_vmem (bf16) → output_hbm --
    b_y_out_vmem[...] = b_y_acc_vmem[...].astype(b_y_out_vmem.dtype)
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=output_hbm, sem=y_out_sem.at[0],
    ).start()
    pltpu.make_async_copy(
        src_ref=b_y_out_vmem, dst_ref=b_y_out_vmem, sem=y_out_sem.at[0],
    ).wait()
```

- [ ] **Step 3.2: Replace the `NotImplementedError` in `double_buffer_expert` with the pallas_call wiring**

Replace the `raise NotImplementedError(...)` line at the bottom of `double_buffer_expert` with:

```python
    dtype = tokens.dtype
    weight_dtype = w1.dtype

    scratch_shapes = (
        pltpu.VMEM((2, d, bf), weight_dtype),            # b_w1_x2_vmem
        pltpu.VMEM((2, d, bf), weight_dtype),            # b_w3_x2_vmem
        pltpu.VMEM((2, bf, d), weight_dtype),            # b_w2_x2_vmem
        pltpu.VMEM((bt, d), dtype),                      # b_x_vmem
        pltpu.VMEM((bt, d), jnp.float32),                # b_y_acc_vmem (fp32)
        pltpu.VMEM((bt, d), dtype),                      # b_y_out_vmem (bf16 stage)
        pltpu.SemaphoreType.DMA((2, 3)),                 # weight_sems[slot, channel]
        pltpu.SemaphoreType.DMA((1,)),                   # x_sem
        pltpu.SemaphoreType.DMA((1,)),                   # y_out_sem
    )

    scope_name = f"double-buffer-expert-bt{bt}-bf{bf}"
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _double_buffer_expert_kernel,
                act_fn=act_fn,
                bf=bf,
                intermediate_size=f_full,
            ),
            out_shape=jax.ShapeDtypeStruct((bt, d), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm, hbm, hbm, hbm],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=96 * 1024 * 1024,
            ),
            name=scope_name,
        )
    )

    @jax.jit
    def run_kernel(tokens, w1, w2, w3):
        return kernel(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
        )

    return run_kernel(tokens, w1, w2, w3)
```

- [ ] **Step 3.3: Verify import still succeeds and structural tests still pass**

```bash
PYTHONPATH=. pytest tests/test_double_buffer_expert.py -v
PYTHONPATH=. python -c "from kernels.double_buffer_expert import double_buffer_expert, _ref_expert_ffn; print('OK')"
```

Expected: all structural tests PASS; import prints `OK`.

- [ ] **Step 3.4: Compile-check (CPU-only verification that the Pallas tracing is valid)**

Pallas's CPU backend is not expected to execute but tracing/AOT compile should succeed. If running on a TPU machine, this will actually compile; on CPU it will raise `ValueError` about CPU backend, which is an acceptable "structurally valid" signal.

```bash
PYTHONPATH=. python -c "
import jax, jax.numpy as jnp
from kernels.double_buffer_expert import double_buffer_expert
tokens = jnp.zeros((256, 8192), dtype=jnp.bfloat16)
w1 = jnp.zeros((8192, 2048), dtype=jnp.bfloat16)
w2 = jnp.zeros((2048, 8192), dtype=jnp.bfloat16)
w3 = jnp.zeros((8192, 2048), dtype=jnp.bfloat16)
try:
    out = double_buffer_expert(tokens, w1, w2, w3, bf=512)
    print('TPU compile OK, out.shape=', out.shape)
except ValueError as e:
    if 'CPU backend' in str(e):
        print('PASS (tracing valid; CPU backend refused — expected off-TPU)')
    else:
        raise
except NotImplementedError as e:
    raise RuntimeError(f'Wiring still stubbed: {e}') from e
"
```

Expected: `TPU compile OK, out.shape= (256, 8192)` on TPU, or `PASS (tracing valid; CPU backend refused — expected off-TPU)` on CPU. If it raises `NotImplementedError`, the wiring step was missed.

- [ ] **Step 3.5: Commit**

```bash
git add kernels/double_buffer_expert.py
git commit -m "feat: implement double_buffer_expert Pallas kernel body (§5.8 pipeline)"
```

---

## Task 4: `__main__` Self-Check (correctness vs. reference)

**Files:**
- Modify: `kernels/double_buffer_expert.py` (append `__main__` block)

- [ ] **Step 4.1: Append the `__main__` block to `kernels/double_buffer_expert.py`**

Add at the end of the file:

```python
if __name__ == "__main__":
    bt = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    bf_arg = 512 if bt == 256 else 256

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (bt, 8192), dtype=jnp.bfloat16)
    w1 = jax.random.normal(k2, (8192, 2048), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3, (2048, 8192), dtype=jnp.bfloat16)
    w3 = jax.random.normal(k4, (8192, 2048), dtype=jnp.bfloat16)

    result = double_buffer_expert(tokens, w1, w2, w3, bf=bf_arg)
    ref = _ref_expert_ffn(tokens, w1, w2, w3)
    result_f32 = result.astype(jnp.float32)
    ref_f32 = ref.astype(jnp.float32)
    max_err = jnp.max(jnp.abs(result_f32 - ref_f32))
    rel_err = max_err / (jnp.max(jnp.abs(ref_f32)) + 1e-6)
    print(f"bt={bt}, bf={bf_arg}, max_abs_err={max_err:.4f}, rel_err={rel_err:.4f}")
    assert rel_err < 0.05, f"rel_err too high: {rel_err}"
    print("PASS")
```

- [ ] **Step 4.2: Run on TPU — variant A (bt=256, bf=512)**

```bash
PYTHONPATH=. python kernels/double_buffer_expert.py 256
```

Expected (on TPU): `bt=256, bf=512, max_abs_err=<small>, rel_err=<small>` followed by `PASS`. `rel_err` should be well under 0.05; typical values for bf16 matmul with fp32 accumulation are ~0.001–0.01.

If running off-TPU, this step is skipped. Note it and continue.

- [ ] **Step 4.3: Run on TPU — variant B (bt=512, bf=256)**

```bash
PYTHONPATH=. python kernels/double_buffer_expert.py 512
```

Expected: `bt=512, bf=256, ..., PASS`.

- [ ] **Step 4.4: Commit**

```bash
git add kernels/double_buffer_expert.py
git commit -m "feat: add double_buffer_expert __main__ self-check vs reference"
```

---

## Task 5: Benchmark Runner `--num-tokens` Plumbing

**Files:**
- Modify: `scripts/benchmark_runner.py`
- Modify: `tests/test_benchmark_runner.py`

### Task 5a: Add failing tests for `--num-tokens` parsing

- [ ] **Step 5a.1: Add tests to `tests/test_benchmark_runner.py`**

Find the `TestCliArgParsing` class and append these test methods (to mirror the existing `test_parses_chunk_size` pattern):

```python
    def test_parses_num_tokens(self):
        runner = _import_runner()
        args = runner.parse_args(
            ["--kernel", "k", "--shape", "1", "--num-tokens", "512"]
        )
        assert args.num_tokens == 512

    def test_num_tokens_default_is_none(self):
        runner = _import_runner()
        args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.num_tokens is None
```

And append to `TestEnvVarFallback`:

```python
    def test_num_tokens_from_env(self):
        runner = _import_runner()
        with patch.dict(os.environ, {"NUM_TOKENS": "512"}):
            args = runner.parse_args(["--kernel", "k", "--shape", "1"])
        assert args.num_tokens == 512
```

- [ ] **Step 5a.2: Run tests to verify they fail**

```bash
PYTHONPATH=. pytest tests/test_benchmark_runner.py -v -k "num_tokens"
```

Expected: all three new tests FAIL with `AttributeError: 'Namespace' object has no attribute 'num_tokens'` or similar.

- [ ] **Step 5a.3: Commit**

```bash
git add tests/test_benchmark_runner.py
git commit -m "test: add failing tests for benchmark_runner --num-tokens (TDD red)"
```

### Task 5b: Implement `--num-tokens` in `benchmark_runner.py`

- [ ] **Step 5b.1: Add the `--num-tokens` CLI argument**

In `scripts/benchmark_runner.py`, locate the argparse block in `parse_args()` — specifically the `--bd` definition (around line 96–101). Immediately after it, before the `--output-dir` definition, insert:

```python
    p.add_argument(
        "--num-tokens",
        type=int,
        default=_int_or_none(os.environ.get("NUM_TOKENS")),
        help="Override config default_shape.num_tokens (per-run override)",
    )
```

- [ ] **Step 5b.2: Plumb `num_tokens` into `run_benchmark`**

Modify the `run_benchmark` function signature and body (around lines 292–303). Change:

```python
def run_benchmark(kernel_fn, config, num_warmup, num_runs, chunk_size=None, ep_size=None, bf=None, bd=None):
    """Execute kernel benchmark and return list of timing values (seconds)."""
    kwargs = dict(config.get("default_shape", {}))
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    kwargs["ep_size"] = ep_size if ep_size is not None else config.get("ep_size", 4)
    if bf is not None:
        kwargs["bf"] = bf
    if bd is not None:
        kwargs["bd"] = bd
```

to:

```python
def run_benchmark(kernel_fn, config, num_warmup, num_runs, chunk_size=None, ep_size=None, bf=None, bd=None, num_tokens=None):
    """Execute kernel benchmark and return list of timing values (seconds)."""
    kwargs = dict(config.get("default_shape", {}))
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    kwargs["ep_size"] = ep_size if ep_size is not None else config.get("ep_size", 4)
    if bf is not None:
        kwargs["bf"] = bf
    if bd is not None:
        kwargs["bd"] = bd
    if num_tokens is not None:
        kwargs["num_tokens"] = num_tokens
```

- [ ] **Step 5b.3: Thread `args.num_tokens` from `main()` into `run_benchmark()`**

In `main()`, find the `run_benchmark` call (around line 668–674):

```python
    timings = run_benchmark(
        kernel_fn, config, args.num_warmup, args.num_runs,
        chunk_size=args.chunk_size,
        ep_size=args.ep_size,
        bf=args.bf,
        bd=args.bd,
    )
```

Change to:

```python
    timings = run_benchmark(
        kernel_fn, config, args.num_warmup, args.num_runs,
        chunk_size=args.chunk_size,
        ep_size=args.ep_size,
        bf=args.bf,
        bd=args.bd,
        num_tokens=args.num_tokens,
    )
```

- [ ] **Step 5b.4: Run tests to verify they pass**

```bash
PYTHONPATH=. pytest tests/test_benchmark_runner.py -v -k "num_tokens"
```

Expected: all three new `num_tokens` tests PASS.

- [ ] **Step 5b.5: Run the full runner test suite to check for regressions**

```bash
PYTHONPATH=. pytest tests/test_benchmark_runner.py -v
```

Expected: all existing tests continue to pass; the three new tests pass.

- [ ] **Step 5b.6: Commit**

```bash
git add scripts/benchmark_runner.py
git commit -m "feat(benchmark_runner): add --num-tokens flag + env var fallback"
```

---

## Task 6: Add Benchmark YAML Config

**Files:**
- Create: `double_buffer_expert_ling2.6.yaml`

- [ ] **Step 6.1: Create the YAML file at repo root**

Create `double_buffer_expert_ling2.6.yaml`:

```yaml
# Double-buffer expert (§5.8 pipeline) — Ling 2.6 1T decode profile
# ==================================================================
# Stage:  Decode (autoregressive, B=1 per routed expert)
# Model:  Ling-2.6-1T routed expert (d=8192, f=2048, bf16)
# Pipeline: docs/fused-moe-perf-analysis.md §5.8
#   persistent x, dual W slots, low-priority W2 DMA, fp32 y_acc
#
# Two recommended (num_tokens, bf) pairs from §5.8 "选择建议":
#
#   Variant A (bt=256, 8 MB tile):
#     python scripts/benchmark_runner.py \
#       --kernel kernels.double_buffer_expert \
#       --shape "256,8192,2048" --job-name dbe_bt256_bf512 \
#       --num-tokens 256 --bf 512 \
#       --output-dir /tmp/bench
#
#   Variant B (bt=512, 4 MB tile):
#     python scripts/benchmark_runner.py \
#       --kernel kernels.double_buffer_expert \
#       --shape "512,8192,2048" --job-name dbe_bt512_bf256 \
#       --num-tokens 512 --bf 256 \
#       --output-dir /tmp/bench
#
# Shape encoding (metadata): num_tokens,hidden_size,intermediate_size

kernel: kernels.double_buffer_expert
shape: "256,8192,2048"
tpu_type: v7x
tpu_topology: 2x2x1
```

- [ ] **Step 6.2: Sanity-check the YAML parses**

```bash
python -c "import yaml; print(yaml.safe_load(open('double_buffer_expert_ling2.6.yaml')))"
```

Expected: prints a dict with keys `kernel`, `shape`, `tpu_type`, `tpu_topology`.

- [ ] **Step 6.3: Commit**

```bash
git add double_buffer_expert_ling2.6.yaml
git commit -m "chore: add double_buffer_expert_ling2.6.yaml benchmark config"
```

---

## Final Verification

- [ ] **Step F.1: Run the full test suite (structural only; no TPU needed)**

```bash
PYTHONPATH=. pytest tests/test_double_buffer_expert.py tests/test_benchmark_runner.py -v
```

Expected: all tests PASS.

- [ ] **Step F.2: On TPU — run both benchmark variants end-to-end**

```bash
# Variant A
python scripts/benchmark_runner.py \
  --kernel kernels.double_buffer_expert \
  --shape "256,8192,2048" --job-name dbe_bt256_bf512 \
  --num-tokens 256 --bf 512 \
  --output-dir /tmp/bench_dbe_A

# Variant B
python scripts/benchmark_runner.py \
  --kernel kernels.double_buffer_expert \
  --shape "512,8192,2048" --job-name dbe_bt512_bf256 \
  --num-tokens 512 --bf 256 \
  --output-dir /tmp/bench_dbe_B
```

Expected: each run prints `[benchmark] Done!` and writes `metrics.jsonl` under `/tmp/bench_dbe_*/rank-0/benchmark/`.

- [ ] **Step F.3: Verify metrics files exist and contain timing**

```bash
cat /tmp/bench_dbe_A/rank-0/benchmark/metrics.jsonl | head -1
cat /tmp/bench_dbe_B/rank-0/benchmark/metrics.jsonl | head -1
```

Expected: one JSON line each with `kernel`, `shape`, `timings_ms`, `statistics.mean_ms`, etc.

- [ ] **Step F.4: Final commit (if any fixups needed)**

If the TPU runs surfaced bugs, fix them now, verify, and commit each fix separately.

```bash
git log --oneline --no-decorate -10
```

Expected: a clean sequence of commits from Tasks 1–6 plus any fix-ups.
