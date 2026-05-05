# Fused MoE Kernel — Systematic Theoretical Analysis

> Generalized parameterized framework for fused expert-parallel MoE performance analysis on TPU v7x.
> Covers decode and prefill regimes, arbitrary DP/EP/TP configurations.
> Builds on and generalizes the Ling 2.6 decode case study in `primatrix/wiki/`.

---

## Part 1: System Model

### 1.1 Hardware Parameters (TPU v7x)

| Symbol | Value | Description |
|--------|-------|-------------|
| $\Phi_{\text{MXU}}$ | 2,307 TFLOPS | Peak BF16 MXU throughput (dual MXU) |
| $\Phi_{\text{VPU}}$ | — | VPU throughput (scalar, not limiting for this kernel) |
| $\Theta_{\text{HBM}}$ | 3,690 GB/s | Peak HBM bandwidth |
| $\Theta_{\text{ICI}}$ | ~300 GB/s (per link) | Inter-chip interconnect bandwidth |
| $M_{\text{VMEM}}$ | 64 MiB | VMEM capacity per TensorCore |
| $R_{\text{VPR}}$ | ~32K | VPR register file capacity |
| v7x topology | 1 chip = 2 TensorCores | `tpuv7x=2` devices per chip |

### 1.2 Model & Data Parameters

| Symbol | Description | Ling 2.6 Decode |
|--------|-------------|-----------------|
| $B$ | Global batch size (tokens) | 256 |
| $E$ | Total routed experts | 256 |
| $k$ | Top-K routing | 8 |
| $H$ | Hidden dimension | 8192 |
| $I$ | Expert FFN intermediate dimension | 2048 |
| $I_{\text{se}}$ | Shared expert intermediate dimension | 2048 |
| $\tau$ | Data type size (BF16=2, FP8=1) | 2 |
| $\tau_{\text{acc}}$ | Accumulator type size (F32=4) | 4 |

### 1.3 Parallelism Parameters

The kernel uses a 2D mesh `("data", "tensor")`:

```
Mesh: devices.reshape(1, ep_size) → axis_names=("data", "tensor")
```

| Symbol | Formula | Description |
|--------|---------|-------------|
| $DP$ | — | Data-parallel replicas (along "data" axis) |
| $EP$ | $\text{ep\_size}$ | Expert-parallel shard count (along "tensor" axis) |
| $TP$ | — | Tensor-parallel within expert (not used in current kernel) |
| $N_{\text{dev}}$ | $DP \times EP$ | Total devices in mesh |

For the current kernel: $DP=1$, $EP=8$ (or $EP=4$ in some configs), mesh = `(1, EP)`.

**Critical insight**: DP and EP operate on orthogonal dimensions:
- **DP**: splits the *token batch* across replicas. Each DP replica sees $B/DP$ tokens, independent routing, no cross-replica communication for MoE.
- **EP**: splits the *experts* across devices. Tokens must be routed to the device holding their target expert via all-to-all.

### 1.4 Derived Per-Device Quantities

| Symbol | Formula | Ling 2.6 (EP=8) | Ling 2.6 (EP=4) |
|--------|---------|-----------------|-----------------|
| $T_{\text{local}}$ | $\lceil B / DP \rceil$ | 256 | 256 |
| $E_{\text{local}}$ | $E / EP$ | 32 | 64 |
| $Q_{\text{pairs}}$ | $T_{\text{local}} \times k$ | 2048 | 2048 |
| $\bar{n}_e$ | $Q_{\text{pairs}} / E_{\text{local}}$ | 64 | 32 |
| $\bar{n}_e^{\text{global}}$ | $(B \times k) / E$ | 8 | 8 |
| $P_{\text{local}}$ | $E_{\text{local}} / E$ | $1/EP$ | $1/EP$ |

**Key relationship**: $\bar{n}_e$ (per-device average tokens per local expert) = $B \cdot k \cdot EP / (DP \cdot E)$.

For decode with $B=256, k=8, E=256$: $\bar{n}_e = 256 \times 8 / 256 = 8$ (global average, independent of EP/DP). But $E_{\text{local}}$ scales as $E/EP$, so total expert processing work per device is $\propto 1/EP$ while per-expert token count stays at 8.

### 1.5 Block Config Parameters

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| `bt` | Outer token tile for routing/output | $\min(\text{bt\_cfg}, T_{\text{local}})$, divides $T_{\text{local}}$ |
| `bts` | Token staging tile inside expert_ffn | $\leq bt \times EP$ |
| `btc` | Compute tile M-dimension | $\leq bts$, divides $bts$ |
| `bf` | Intermediate size tile (I/O dim) | divides $I$ |
| `bfc` | Intermediate compute tile | $\leq bf$ |
| `bd1` | Hidden size tile for FFN1 K-dim | divides $H$ |
| `bd1c` | Hidden compute tile for FFN1 | $\leq bd1$ |
| `bd2` | Hidden size tile for FFN2 N-dim | divides $H$ |
| `bd2c` | Hidden compute tile for FFN2 | $\leq bd2$ |
| `bse` | Shared expert intermediate tile | divides $I_{\text{se}}$ |
| $p$ | `t_packing` = dtype packing factor | 2 (BF16), 4 (FP8) |

### 1.6 Stage-Scope Mapping

The user's proposed 4-stage decomposition maps to the kernel's execution as follows:

```
Stage 1: gate, topk, allreduce metadata, permute
         → wait_fetch_topk + t2e_routing computation + all_reduce_metadata

Stage 2: alltoall dispatch, expert compute, alltoall combine
         → start_a2a_scatter + expert_ffn loop (wait_scatter + ffn + start_gather)

Stage 3: compute with shared expert and MoE output
         → run_shared_expert_slice (interleaved with Stage 2) + acc_and_store_output

Stage 4: shared expert compute (data-independent from Stage 1-2)
         → shared expert FFN (gate+up+activation+down), but interleaved with Stage 2
           in the current implementation
```

**Data dependency DAG**:

```
  Tokens (HBM)
      │
      ├──→ [S1] Gate + TopK ──→ allreduce metadata ──→ permute/routing table
      │                                                        │
      │                                                        ▼
      │                                                   [S2] A2A Scatter
      │                                                        │
      │                                                        ▼
      │                                              [S2] Expert FFN × E_local
      │                                                        │
      │                                                        ▼
      │                                              [S2] A2A Combine
      │                                                        │
      │                                                        ▼
      │                                              [S3] Output Accumulation
      │
      └──→ [S4] Shared Expert FFN ──────────────────────────→ [S3] (add SE output)
                                                                   │
                                                                   ▼
                                                              Output (HBM)

S4 is data-independent of S1-S2. S3 depends on both S2 and S4.
```

---

## Part 2: Stage-by-Stage Theoretical Formulas

### 2.1 Stage 1: Gate, TopK, AllReduce Metadata, Permute

#### 2.1.1 Gate Logits (if not pre-computed)

If gate logits need to be computed on-device (typically done in a separate upstream op):

$$\text{FLOPs}_{\text{gate}} = T_{\text{local}} \times H \times E \quad \text{MAYBE skipping if pre-computed}$$

In practice, gate logits are computed upstream; the fused MoE kernel receives `topk_ids` and `topk_weights` already computed.

#### 2.1.2 TopK Data Fetch

**HBM Read**:

$$HBM_{\text{read}}^{S1} = bt \times k \times (\tau_{\text{f32}} + \tau_{\text{s32}}) = bt \times k \times (4 + 4) = 8 \cdot bt \cdot k \ \text{bytes}$$

Per bt tile: topk_weights (F32) + topk_ids (S32), each of shape `[bt, k]`.

#### 2.1.3 Routing Table Construction

Compute `t2e_routing` and `expert_sizes`:

```
For each token t in [0, bt):
  For each expert rank r in [0, k):
    e_id = topk_ids[t, r]
    t2e_routing[t, r] = e_id
    expert_sizes[0, e_id] += 1
```

**VPU FLOPs**: $\approx bt \times k \times E$ comparison operations (broadcasted equality check), plus $bt \times k$ reductions. But this is negligible (~$10^5$ ops) compared to MXU operations.

**LLO observation**: `vrot.slane`, `vbcast.lane`, `vpop`, `vselect` chain — scalar/vector operations, no MXU used.

#### 2.1.4 AllReduce Metadata

Allgather per-device `expert_sizes` (shape `[1, E]` S32) across all EP devices.

**Communication volume per device**:

$$V_{\text{comm}}^{S1} = E \times \tau_{\text{s32}} = E \times 4 \ \text{bytes}$$

Using recursive doubling (power-of-two EP): $\lceil \log_2 EP \rceil$ rounds.

For $E=256, EP=8$: $V_{\text{comm}} = 1 \text{ KB}$, 3 rounds.

**Latency model**:

$$T_{S1}^{\text{comm}} \approx \lceil \log_2 EP \rceil \times (t_{\text{latency}} + V_{\text{comm}} / \Theta_{\text{ICI}}) + t_{\text{barrier}}$$

Where $t_{\text{latency}} \approx 2\text{-}5\ \mu\text{s}$ (ICI link setup), $t_{\text{barrier}} \approx 5\text{-}10\ \mu\text{s}$.

For current config: $T_{S1}^{\text{comm}} \approx 10\text{-}20\ \mu\text{s}$.

#### 2.1.5 Permute

After receiving global expert sizes, compute `expert_starts` via prefix-sum — pure VMEM/SMEM scalar operation, negligible time.

#### 2.1.6 Stage 1 Summary

| Metric | Formula | Value (Ling 2.6, EP=8) |
|--------|---------|------------------------|
| HBM Read | $8 \cdot bt \cdot k$ | 4 KB |
| HBM Write | 0 | 0 |
| VPU FLOPs | $\sim bt \cdot k \cdot E$ | ~131K |
| ICI Comm | $E \cdot 4 \cdot \lceil \log_2 EP \rceil$ (total) | ~3 KB |
| Bottleneck | Communication latency | ~10-20 μs |

### 2.2 Stage 2: All-to-All Dispatch + Expert Compute + All-to-All Combine

#### 2.2.1 All-to-All Dispatch (Scatter)

Each token is replicated to its top-k expert devices. Per-device scatter volume:

$$V_{\text{scatter}} = T_{\text{local}} \times k \times H \times \tau$$

The fraction of local vs remote traffic:

$$f_{\text{local}} = \frac{E_{\text{local}}}{E} = \frac{1}{EP}$$

$$f_{\text{remote}} = 1 - \frac{1}{EP}$$

**Per-device ICI outbound**:

$$V_{\text{ICI-out}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau \times (1 - \frac{1}{EP})$$

**Per-device ICI inbound** (symmetric):

$$V_{\text{ICI-in}}^{S2\text{-disp}} = V_{\text{ICI-out}}^{S2\text{-disp}}$$

**HBM traffic (local DMA)**:

$$V_{\text{DMA-local}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau \times \frac{1}{EP}$$

**HBM Write (scratch buffers)**:

$$V_{\text{HBM-write}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau$$

Tokens copied to `a2a_s_hbm` scratch buffers, organized into `expert_buffer_count` slots.

**Latency model**:

$$T_{S2}^{\text{scatter}} \approx \max\left( \frac{V_{\text{DMA-local}}}{\Theta_{\text{HBM}}}, \frac{V_{\text{ICI-out}}}{\Theta_{\text{ICI}}} \right) + N_{\text{tokens}} \times t_{\text{DMA-setup}}$$

For Ling 2.6 decode (EP=8): $V_{\text{ICI-out}} = 256 \times 8 \times 8192 \times 2 \times 7/8 = 28.7 \text{ MB}$, $T_{S2}^{\text{scatter}} \approx 40\text{-}60\ \mu\text{s}$.

**For EP=1 (no all-to-all)**: $V_{\text{ICI-out}} = 0$, scatter degenerates to local DMA only.

#### 2.2.2 Expert FFN Compute

For each local expert $e$, with $n_e$ tokens (dynamic, routing-dependent):

**Weight loading (per expert)**:

Gate (W1): $H \times I \times \tau$ bytes
Up (W3): $H \times I \times \tau$ bytes
Down (W2): $I \times H \times \tau$ bytes

$$W_{\text{per-expert}} = (2H \cdot I + I \cdot H) \times \tau = 3 \cdot H \cdot I \cdot \tau$$

**Tiling factor** (how many times weights are loaded):

In the weight-streaming (Scheme A) approach, weights are loaded per `(bf, bd1/bd2)` tile, with `btc` as the inner M-dimension.

$$n_{\text{bd1}} = H / bd1,\quad n_{\text{bd2}} = H / bd2,\quad n_{\text{bf}} = I / bf$$

Each per-expert FFN processes $n_e$ tokens through:

**FFN1 (Gate + Up)**: For each `(bf_id, bd1_id)` tile, load W1/W3 tile and token tile, then two MXU matmuls.

$$\text{FLOPs}_{\text{FFN1}} = 2 \cdot n_{\text{bf}} \cdot n_{\text{bd1}} \cdot p \cdot (2 \cdot btc \cdot (bd1/p) \cdot bf)$$

$$\text{FLOPs}_{\text{FFN1}} = 4 \cdot n_{\text{bf}} \cdot n_{\text{bd1}} \cdot btc \cdot \frac{bd1}{p} \cdot bf$$

With $btc = bd1/p = bf$ (common case for block configs) and $n_{\text{bf}} \cdot bf = I$, $n_{\text{bd1}} \cdot bd1 = H$:

$$\boxed{\text{FLOPs}_{\text{FFN1}} = 4 \cdot btc \cdot H \cdot I}$$

**FFN2 (Down)**: For each `(bf_id, bd2_id)` tile, load W2 tile and activation, then one MXU matmul.

$$\boxed{\text{FLOPs}_{\text{FFN2}} = 2 \cdot btc \cdot H \cdot I}$$

For $btc=64$ (the full tile, including padding): $\text{FLOPs}_{\text{FFN1}} = 4 \times 64 \times 8192 \times 2048 = 4.29 \text{ GFLOPS}$, $\text{FLOPs}_{\text{FFN2}} = 2.15 \text{ GFLOPS}$.

**Per-expert useful FLOPs** (accounting for padding):

$$\boxed{\text{FLOPs}_{\text{useful}} = \text{FLOPs}_{\text{executed}} \times \frac{\bar{n}_e}{btc}}$$

For decode with $\bar{n}_e = 8$: useful FLOPs are only $8/64 = 12.5\%$ of executed.

**Total per-device MXU FLOPs**:

$$\boxed{\text{FLOPs}_{\text{total}} = E_{\text{local}} \times 6 \cdot btc \cdot H \cdot I}$$

#### 2.2.3 HBM Traffic Per Expert (Detailed)

Per expert FFN with tile-based weight loading:

$$HBM_{\text{weights}} = 3 \cdot H \cdot I \cdot \tau$$

Token staging (HBM → VMEM):

$$HBM_{\text{tokens}} = n_e \cdot H \cdot \tau$$

Result writeback:

$$HBM_{\text{results}} = n_e \cdot H \cdot \tau$$

**Total per-expert HBM read**:

$$\boxed{HBM_{\text{read}}^{\text{expert}} = 3 \cdot H \cdot I \cdot \tau + n_e \cdot H \cdot \tau}$$

**Total per-device HBM read (all local experts)**:

$$\boxed{HBM_{\text{read}}^{\text{all}} = E_{\text{local}} \times 3 \cdot H \cdot I \cdot \tau + \sum_{e} n_e \cdot H \cdot \tau}$$

Since $\sum_e n_e = Q_{\text{pairs}} = T_{\text{local}} \times k$:

$$\boxed{HBM_{\text{read}}^{\text{all}} = E_{\text{local}} \times 3 \cdot H \cdot I \cdot \tau + T_{\text{local}} \times k \times H \times \tau}$$

**Dominant term**: For $E=256, EP=8, H=8192, I=2048$:
- Weight term: $32 \times 3 \times 8192 \times 2048 \times 2 = 3.0 \text{ GB}$
- Token term: $256 \times 8 \times 8192 \times 2 = 32 \text{ MB}$
- Weights dominate by ~100:1

#### 2.2.4 Arithmetic Intensity

Per-expert arithmetic intensity (with padding):

$$\boxed{AI_{\text{executed}} = \frac{6 \cdot btc \cdot H \cdot I}{3 \cdot H \cdot I \cdot \tau + n_e \cdot H \cdot \tau} \approx \frac{2 \cdot btc}{\tau}}$$

For $\tau=2$ (BF16): $AI_{\text{executed}} \approx btc = 64 \text{ FLOPs/byte}$. Ridge point = $2307/3.69 = 625$.

Per-expert useful AI:

$$\boxed{AI_{\text{useful}} = AI_{\text{executed}} \times \frac{\bar{n}_e}{btc} \approx \frac{2 \cdot \bar{n}_e}{\tau}}$$

For $\bar{n}_e = 8, \tau = 2$: $AI_{\text{useful}} = 8.0 \text{ FLOPs/byte}$.

**Critical insight**: $AI_{\text{useful}}$ depends only on $\bar{n}_e$ and $\tau$, not on block config or hidden dimensions. This means:

- **Decode** ($\bar{n}_e \approx 8$): $AI \approx 8 \ll 625$ → **extremely memory-bound**
- **Prefill** ($\bar{n}_e \approx 512$): $AI \approx 512 \approx 625$ → **near ridge point, potentially compute-bound**

#### 2.2.5 All-to-All Combine (Gather)

Reverse of scatter: each expert's FFN output is sent back to the token's original device.

Per-device gather volume (symmetric to scatter):

$$V_{\text{gather}} = T_{\text{local}} \times k \times H \times \tau$$

$$V_{\text{ICI-in}}^{S2\text{-gather}} = V_{\text{ICI-out}}^{S2\text{-disp}}$$

#### 2.2.6 Stage 2 Total Time Estimate

$$\boxed{T_{S2} \approx \max\left( \frac{HBM_{\text{read}}^{\text{all}}}{\Theta_{\text{HBM}}}, \frac{\text{FLOPs}_{\text{total}}}{\Phi_{\text{MXU}}} \right) + T_{\text{scatter}} + T_{\text{gather}}}$$

For Ling 2.6 decode (EP=8): $T_{S2}^{\text{HBM}} \approx 3.0 \text{ GB} / 3690 \text{ GB/s} = 0.81 \text{ ms}$, $T_{S2}^{\text{compute}} \approx 206 \text{ GFLOPS} / 2307 \text{ TFLOPS} = 0.09 \text{ ms}$. Stage 2 is **HBM-bound** with theoretical lower bound ~0.8 ms.

### 2.3 Stage 3: Shared Expert + MoE Output Combination

#### 2.3.1 Output Accumulation

For each token $t$, aggregate top-k expert outputs weighted by topk_weights, plus shared expert output:

$$y_t = \sum_{r=1}^{k} w_{t,r} \cdot \text{expert\_output}_{t,r} + y_t^{\text{se}}$$

**HBM Read**: Load top-k expert outputs from `a2a_g_hbm`.

$$HBM_{\text{read}}^{S3} = T_{\text{local}} \times k \times H \times \tau$$

**VPU FLOPs**: $T_{\text{local}} \times k \times H$ multiply-adds.

$$FLOPs_{S3} = 2 \cdot T_{\text{local}} \cdot k \cdot H$$

For Ling 2.6 decode: 8 MB read, ~4 MFLOPS — negligible (< 10 μs).

#### 2.3.2 Shared Expert Contribution

Shared expert output (F32 accumulator in VMEM) is added to the MoE weighted sum:

```python
y_t_final = bf16(y_t_f32 + se_output_f32[t])
```

This is a VMEM-resident operation, no additional HBM traffic.

#### 2.3.3 Output Writeback

$$HBM_{\text{write}}^{S3} = T_{\text{local}} \times H \times \tau$$

For Ling 2.6: 1 MB.

### 2.4 Stage 4: Shared Expert FFN (Standalone Analysis)

#### 2.4.1 Shared Expert Architecture

The shared expert is a separate FFN (Gate + Up + Activation + Down) applied to ALL tokens, independent of routing:

$$y_{\text{se}} = W2_{\text{se}} \cdot \text{SiLU}(W1_{\text{se}} \cdot x) \odot (W3_{\text{se}} \cdot x)$$

Weights: $W1_{\text{se}}, W3_{\text{se}} \in \mathbb{R}^{H \times I_{\text{se}}}$, $W2_{\text{se}} \in \mathbb{R}^{I_{\text{se}} \times H}$.

#### 2.4.2 SE HBM Traffic (Full)

$$HBM_{\text{read}}^{S4\text{-weights}} = 3 \cdot H \cdot I_{\text{se}} \cdot \tau$$

$$HBM_{\text{read}}^{S4\text{-tokens}} = T_{\text{local}} \cdot H \cdot \tau$$

$$HBM_{\text{write}}^{S4} = 0 \quad \text{(output stays in VMEM as F32 accumulator)}$$

#### 2.4.3 SE FLOPs

$$\boxed{\text{FLOPs}_{S4} = 6 \cdot T_{\text{local}} \cdot H \cdot I_{\text{se}}}$$

For Ling 2.6 decode: $6 \times 256 \times 8192 \times 2048 = 25.8 \text{ GFLOPS}$.

#### 2.4.4 SE Weight-to-Token Ratio

The key tradeoff for SE scheduling:

$$\frac{HBM_{\text{weights}}^{S4}}{HBM_{\text{tokens}}^{S4}} = \frac{3 \cdot H \cdot I_{\text{se}} \cdot \tau}{T_{\text{local}} \cdot H \cdot \tau} = \frac{3 \cdot I_{\text{se}}}{T_{\text{local}}}$$

For Ling 2.6 decode ($I_{\text{se}}=2048, T_{\text{local}}=256$): ratio = 24:1 — weight loading dominates.

For prefill ($T_{\text{local}} = 2048$): ratio = 3:1 — more balanced.

#### 2.4.5 SE Arithmetic Intensity

$$AI_{\text{SE}} = \frac{6 \cdot T_{\text{local}} \cdot H \cdot I_{\text{se}}}{3 \cdot H \cdot I_{\text{se}} \cdot \tau + T_{\text{local}} \cdot H \cdot \tau} = \frac{6 \cdot T_{\text{local}} \cdot I_{\text{se}}}{3 \cdot I_{\text{se}} \cdot \tau + T_{\text{local}} \cdot \tau}$$

For $T_{\text{local}} = 256$: $AI_{\text{SE}} \approx 98 \text{ FLOPs/byte}$ (memory-bound, but less severe than routed experts).
For $T_{\text{local}} = 2048$: $AI_{\text{SE}} \approx 384 \text{ FLOPs/byte}$ (approaching ridge point).

---

## Part 3: LLO Overhead & VMEM Analysis

### 3.1 Decomposing the Theory-Measurement Gap

Existing benchmark: theoretical lower bound ~1.8 ms (HBM-limited for EP=4, bt=64), measured ~149-150 ms — a **~83x gap**.

| Overhead Category | Estimated Contribution | Mechanism |
|-------------------|----------------------|-----------|
| DMA setup overhead | ~40-60% | Per-tile DMA descriptor construction, semaphore ops (~1,536 DMA requests for 64 experts) |
| Synchronization barriers | ~10-15% | `sync_barrier()`, `wait_scatter_recv()`, `wait_gather_recv_all()` per iteration |
| Loop control (SALU) | ~15-25% | `lax.fori_loop` over 64 experts, unroll=false, each iteration has control-flow overhead |
| Small DMA inefficiency | ~5-10% | ~256 KB token tiles, ~KB-scale metadata transfers — DMA engine under-utilized |
| VMEM spill/fill | ~2-5% | VSTORE:SPILL observed in ~14% of bundles |
| ICI latency | ~2-5% | Cross-chip all-to-all latency for scatter/gather |

**Effective HBM bandwidth**: $6,249 \text{ MB} / 149 \text{ ms} \approx 42 \text{ GB/s}$ — only 1.1% of peak 3,690 GB/s.

### 3.2 K-ary Expert Processing Overhead Model

Rather than treating overhead as a multiplier on theory, we model it additively:

$$\boxed{T_{\text{measured}} = T_{\text{HBM}} + T_{\text{compute}} + E_{\text{local}} \times t_{\text{overhead-per-expert}} + T_{\text{comm}} + T_{\text{fixed}}}$$

Where:
- $T_{\text{HBM}} = HBM_{\text{read}}^{\text{all}} / \Theta_{\text{HBM}}^{\text{effective}}$ — effective BW much lower than peak due to many small DMAs
- $t_{\text{overhead-per-expert}}$ = DMA setup + semaphore wait + loop control per expert iteration
- $T_{\text{fixed}}$ = barrier + output acc + other fixed costs

From LLO analysis: $t_{\text{overhead-per-expert}} \approx 2.3 \text{ ms}$ (derived from 149 ms total / 64 experts), dominated by SALU control flow.

### 3.3 VMEM Occupancy Formula

#### 3.3.1 Buffer Inventory

| Buffer | Shape | Size Formula | Ling 2.6 (bt=64, bf=1024, bd1=bd2=2048) |
|--------|-------|-------------|------------------------------------------|
| Token input (×2) | `[bt, H]` BF16 | $2 \times bt \times H \times \tau$ | 2 MB |
| W1 double buffer | `[p, bd1/p, bf]` BF16 ×2 | $2 \times 2 \times bd1 \times bf \times \tau$ | 8 MB |
| W3 double buffer | same as W1 | $2 \times 2 \times bd1 \times bf \times \tau$ | 8 MB |
| W2 double buffer | `[p, bf, bd2/p]` BF16 ×2 | $2 \times 2 \times bf \times bd2 \times \tau$ | 8 MB |
| Gate accumulator | `[btc, bf]` F32 | $btc \times bf \times \tau_{\text{acc}}$ | 0.25 MB |
| Up accumulator | `[btc, bf]` F32 | $btc \times bf \times \tau_{\text{acc}}$ | 0.25 MB |
| FFN2 accumulator | `[btc, bd2c]` F32 ×3 (triple buffer) | $3 \times btc \times bd2c \times \tau_{\text{acc}}$ | 1.5 MB |
| SE W1 buffer (×2) | `[p, bd1/p, bse]` BF16 ×2 | $2 \times 2 \times bd1 \times bse \times \tau$ | 2 MB |
| SE W2 buffer (×2) | `[p, bse, bd2/p]` BF16 ×2 | $2 \times 2 \times bse \times bd2 \times \tau$ | 2 MB |
| SE W3 buffer (×2) | same as SE W1 | $2 \times 2 \times bd1 \times bse \times \tau$ | 2 MB |
| SE accumulator | `[bt, H]` F32 | $bt \times H \times \tau_{\text{acc}}$ | 2 MB |
| Output buffer (×2) | `[bt, H]` BF16 | $2 \times bt \times H \times \tau$ | 2 MB |
| A2A scatter (×2) | `[E_buf, bt*EP, p, H/p]` BF16 | varies | ~16-32 MB |

#### 3.3.2 Peak VMEM Constraint

$$\boxed{M_{\text{VMEM}} = \sum_{\text{buffers}} \text{size} \leq 64 \text{ MiB}}$$

For Ling 2.6 with EP=4 (large bt=64): ~56 MB → 87% utilization.

For Ling 2.6 with EP=8 (bt may differ): lower utilization since $E_{\text{local}}$ halved.

**Key tradeoff**: Larger `bt` increases VMEM pressure (token buffers grow as $bt \times H$) but reduces $n_{bt}$ (fewer outer loop iterations). The constraint is:

$$\boxed{bt_{\text{max}}} \approx \frac{64 \text{ MiB} - W_{\text{buffers}} - A_{\text{buffers}}}{2 \times H \times (\tau + \tau_{\text{acc}})}$$

For $H=8192$: $bt_{\text{max}} \approx 512$ (theoretical, before accounting for weight buffers).

### 3.4 DMA Efficiency Model

The effective HBM bandwidth for the kernel is a function of DMA request size:

$$\Theta_{\text{HBM}}^{\text{effective}}(s) = \Theta_{\text{HBM}}^{\text{peak}} \times \eta(s)$$

Where $\eta(s)$ is the DMA efficiency for transfer size $s$:

| Transfer Size | Typical $\eta$ | Example |
|--------------|----------------|---------|
| 4 MB (weight tile, bd1=2048, bf=1024) | ~80-90% | W1/W3 tile load |
| 256 KB (token tile, bt=64) | ~40-60% | Token staging |
| 4 KB (metadata) | ~5-10% | Routing table |

This explains much of the theory-measurement gap: the kernel issues many small DMAs where the HBM engine cannot reach peak throughput.

---

## Part 4: Decision Points & Tradeoff Analysis

### 4.1 Shared Expert Overlap with Stage 1

#### 4.1.1 Current Scheme: SE Interleaved with Stage 2

SE is currently sliced and interleaved with routed expert FFN (se_before + se_after per expert iteration). SE computation overlaps with expert FFN DMA — both compete for HBM bandwidth.

#### 4.1.2 Proposed Alternative: SE Overlap with Stage 1

**Key insight**: SE only needs the input tokens, not routing results. Stage 1 does not use the tokens' HBM bandwidth (only ~4 KB of metadata).

**Feasibility**: During Stage 1 (gate computation + allreduce metadata), the tokens are idle in HBM. We could:

1. Prefetch SE weights and tokens into VMEM during Stage 1
2. Execute SE FFN1 (gate+up) while waiting for allreduce barrier
3. Execute SE FFN2 (down) during scatter setup

**VMEM constraint**: SE buffers (W1/W3/W2 buffers + SE accumulator) compete with Stage 1/2 buffers. Need to check peak VMEM:

$$M_{\text{VMEM}}^{\text{SE-during-S1}} = M_{\text{SE-buffers}} + M_{\text{S1-buffers}} \leq 64 \text{ MiB}$$

SE buffers: ~8 MB (W1/W3/W2 double buffers + accumulator).
S1 buffers: token input (2 MB) + weight buffers for upcoming expert FFN (8+8+8=24 MB).
Total: ~34 MB — feasible.

**Benefit**: SE computation (~26 μs theoretical) becomes "free" on the critical path, since it overlaps with Stage 1's communication latency (~15-20 μs).

**Risk**: Additional HBM traffic during scatter phase increases contention:
- SE weights (96 MB) loaded during Stage 1
- This may delay scatter DMA fetching tokens from HBM

**Quantitative assessment**: SE weight loading (96 MB) takes ~26 μs at peak HBM BW. Stage 1 takes ~15-20 μs. Partial overlap possible: SE FFN1 (weights + compute) ≈ 13 μs of the 15-20 μs Stage 1 window. Remaining ~13 μs spills into Stage 2.

#### 4.1.3 Recommendation

Overlap is modestly beneficial (~10-15 μs saved) but increases complexity. The current interleaving scheme is reasonable because:
1. SE is already off the critical path (hidden behind expert FFN)
2. SE weight loading competes with expert weight loading anyway
3. Stage 1 is too short (15 μs) to hide the full SE computation (26 μs)

### 4.2 A2A Combine + Next Dispatch DMA Contention

#### 4.2.1 Current Scheme: Pipelined Scatter + Gather

In the batch scatter path:
- All scatter DMAs issued in one pass (token-loop)
- Per expert: wait scatter recv → ffn → start gather
- Gather DMAs issued per-expert, potentially overlapping with subsequent experts' scatter

In the pipelined path (when `expert_buffer_count < local_num_experts`):
- Per expert: start scatter(N+1) during expert N's FFN
- This creates back-to-back DMA: gather(N) DMAs + scatter(N+1) DMAs

#### 4.2.2 DMA Contention Model

DMA engine has limited queue depth ($D_{\text{queue}}$) and HBM bandwidth is shared. When back-to-back DMAs are issued:

$$T_{\text{DMA-total}} = \sum_i T_{\text{DMA}}(s_i) + \sum_{i,j} \delta(s_i, s_j)$$

Where $\delta(s_i, s_j)$ is the switching overhead between DMA streams $i$ and $j$.

**Quantitative estimate**: With 64 experts, each issuing ~8 gather DMAs (per bf×bd2 tile) and the next expert issuing ~8 scatter DMAs, the interleaving creates ~512 DMA stream switches. Each switch costs ~50-100 ns (HBM controller re-arbitration). Total overhead: ~25-50 μs.

This is relatively small compared to the total kernel time (150 ms) but non-negligible for the theoretical bound (1.8 ms).

#### 4.2.3 Mitigation: Batching

Batching all gathers after all FFN completions (deferred scheme, see 4.3) eliminates this contention entirely by separating the DMA phases.

### 4.3 Deferred Combine + Shared Expert Overlap

#### 4.3.1 Current Scheme: Interleaved Gather

```python
for e in experts:
    expert_ffn(e)
    start_a2a_gather(e)  # DMA issued immediately
```

#### 4.3.2 Proposed: Deferred Gather with SE Fill

```python
for e in experts:
    expert_ffn(e)
    # DON'T start gather yet — defer

# All FFN done, now:
for e in experts:
    start_a2a_gather(e)  # batch all gather DMAs

# Meanwhile: run remaining SE slices
```

#### 4.3.3 Analysis

**Advantage**:
- Clean separation of compute (FFN) and communication (gather) phases
- SE computation fills the gap between FFN completion and gather completion
- No DMA contention between gather and next scatter

**Disadvantage**:
- Gather no longer overlaps with expert FFN compute
- Critical path: $T_{\text{FFN-all}} + \max(T_{\text{SE-remaining}}, T_{\text{gather-all}})$
- vs current: $T_{\text{FFN-all}} + T_{\text{gather-tail}}$ (most gather hidden)

**Quantitative comparison**:

Current scheme: $T_{\text{crit}} \approx T_{\text{FFN-all}} + T_{\text{gather-tail}} \approx 64 \times 28 \mu\text{s} + 5 \mu\text{s} = 1797 \mu\text{s}$

Deferred scheme: $T_{\text{crit}} \approx T_{\text{FFN-all}} + \max(T_{\text{SE-rem}}, T_{\text{gather-all}})$

Where $T_{\text{SE-rem}} \approx 26 \mu\text{s}$ (SE is mostly done by expert 4 in current scheme, could be deferred), $T_{\text{gather-all}} \approx 40 \mu\text{s}$.

$T_{\text{crit}} \approx 1792 + \max(26, 40) = 1832 \mu\text{s}$ — **marginally worse** (+35 μs).

But the deferred scheme has better HBM utilization (no DMA contention), potentially reducing per-expert DMA overhead by 1-2%. The net effect is approximately neutral for the current config.

**When deferred is better**: When SE is larger relative to expert FFN (large $I_{\text{se}} / I$), or when gather DMA contention is severe (small `expert_buffer_count`).

### 4.4 Shared Expert TP Splitting

#### 4.4.1 Current Scheme: SE Replicated

Each EP device holds a full copy of SE weights ($3 \times H \times I_{\text{se}} \times \tau$ bytes) and computes SE independently on its local tokens. No cross-device communication for SE.

SE weight memory per device: $3 \times 8192 \times 2048 \times 2 = 96 \text{ MB}$ (HBM, loaded once per bt tile).

#### 4.4.2 TP Splitting Option

Split SE weights across TP devices along the intermediate dimension:

- Each TP shard holds $I_{\text{se}} / TP$ of W1/W3 columns and W2 rows
- FFN1: partial output, needs allreduce to get full activation
- FFN2: each shard computes partial output, needs allreduce to get full result

**Communication cost per TP split**:

$$V_{\text{comm}}^{\text{TP}} = 2 \times T_{\text{local}} \times I_{\text{se}} \times \tau_{\text{f32}} \quad \text{(FFN1 output + FFN2 output)}$$

For Ling 2.6 decode: $2 \times 256 \times 2048 \times 4 = 4 \text{ MB}$.

**Compute saving per device**: Each device computes $1/TP$ of SE FLOPs. But since SE is already < 1% of total FLOPs (26 GFLOPS vs 412 GFLOPS), this saving is negligible.

**HBM saving per device**: Each device loads $3 \times H \times I_{\text{se}} / TP \times \tau$ bytes. With TP=2: 48 MB vs 96 MB.

**Latency analysis**:

No TP: $T_{\text{SE}} \approx 96 \text{ MB} / \Theta_{\text{HBM}} = 26 \mu\text{s}$ (HBM-bound)

TP=2: $T_{\text{SE}} \approx 48 / 3690 + 4 / 300 = 13 + 13 = 26 \mu\text{s}$ (HBM + ICI)
Plus allreduce latency: ~10-20 μs → total ~36-46 μs.

**Conclusion**: TP splitting SE is **not beneficial** for decode because:
1. Communication latency dominates the saved HBM time
2. SE is already a small fraction of total runtime
3. SE is not on the critical path (interleaved with expert FFN)

**When TP might help**: For very large $I_{\text{se}}$ (e.g., 8192+) where SE weight HBM traffic becomes significant. Or in prefill where $T_{\text{local}}$ is large and allreduce bandwidth utilization is efficient.

### 4.5 o_proj + DP Attention Recomposition & Overlap

#### 4.5.1 Current Architecture

After the fused MoE kernel, the output $y \in \mathbb{R}^{B \times H}$ is distributed across EP devices:
- Each device holds the output for tokens that were routed to its local experts
- Token distribution depends on routing decisions

Downstream attention requires:
- Each DP replica needs its complete token sequence
- If DP>1, tokens need to be re-distributed (allgather or reshape)

#### 4.5.2 Token Distribution After MoE

After the kernel: tokens are on the device that hosts the expert, NOT necessarily their original device. For the next layer:
- Gate/routing re-computes from scratch (independent per layer)
- Input tokens need to be on the correct DP+EP shard

Current sglang-jax pattern: **replicate tokens across all EP devices** (allgather after MoE), so each device has the full output. Then DP sharding for attention can proceed.

#### 4.5.3 Overlap Strategy

The output writeback (`start_send_bo`) already uses DMA to stream output from VMEM to HBM. This can be **overlapped** with the beginning of the next layer's gate projection:

```
Timeline:
  [Expert FFN] [Acc Output] [o_proj DMA] [Next Layer Gate]
                                │              │
                                └── OVERLAP ──┘
```

But this requires the next layer to start before o_proj DMA completes — feasible if the gate projection processes tokens in tiles.

**Quantitative model**:

$$T_{\text{o-proj}} = \frac{T_{\text{local}} \times H \times \tau}{\Theta_{\text{HBM}}} \quad \text{(DMA write)}$$

For Ling 2.6 decode: $256 \times 8192 \times 2 / 3.69 \times 10^{12} \approx 1.1 \mu\text{s}$ — negligible.

The **dominant cost** is actually the allgather for cross-EP token replication (if needed):

$$V_{\text{allgather}} = T_{\text{local}} \times H \times \tau \times (EP - 1)$$

For EP=8: $256 \times 8192 \times 2 \times 7 / 8 = 3.5 \text{ MB}$ per device outbound.

#### 4.5.4 Recommendation

1. **Fuse o_proj with output accumulation**: Instead of writing BF16 output then reading it back for attention, keep the F32 accumulator in VMEM, apply o_proj matmul directly, then write o_proj output.
2. **Overlap o_proj DMA with next gate**: Start next layer's gate matmul as soon as the first token tile of o_proj output is in HBM.
3. **Avoid unnecessary allgather**: If the attention is also EP-sharded (e.g., DeepSpeed Ulysses-style sequence parallelism), token redistribution may not be needed.

### 4.6 Prefill vs Decode Optimal Expert Design

#### 4.6.1 Regime Classification

The key discriminator is $\bar{n}_e$ (average tokens per expert):

| Regime | $\bar{n}_e$ | $AI_{\text{useful}}$ | Bottleneck | Example |
|--------|-------------|---------------------|------------|---------|
| **Decode** | 1-16 | 1-16 | Extreme HBM-bound | $B=256, E=256, k=8 \rightarrow \bar{n}_e=8$ |
| **Small Prefill** | 32-128 | 32-128 | HBM-bound | $B=1024, E=256, k=8 \rightarrow \bar{n}_e=32$ |
| **Large Prefill** | 256-2048 | 256-2048 | Near ridge / compute-bound | $B=8192, E=256, k=8 \rightarrow \bar{n}_e=256$ |

The ridge point for BF16 is $AI_{\text{ridge}} = 625$. At $\bar{n}_e \approx 300$, $AI_{\text{useful}} \approx 300$, still somewhat memory-bound. Full compute-bound requires $\bar{n}_e > 625$.

#### 4.6.2 Decode Optimal Design ($\bar{n}_e \ll btc$)

**Problem**: $btc=64$ but $\bar{n}_e \approx 8$ → 87.5% of MXU FLOPs are wasted on padding.

**Optimization strategies** (in priority order):

1. **Reduce `btc`**: Set $btc = \min(\bar{n}_e, \text{aligned})$. But this increases $n_{\text{btc}}$ (more sub-tiles), which increases DMA overhead proportionally. There's a sweet spot.

2. **FP8 quantization**: Halves weight HBM traffic → nearly halves HBM time. This is the **highest-impact** change for decode.

3. **Expert merging / batching**: Group multiple small experts into shared GEMM calls to amortize DMA setup. E.g., batch 4 experts with similar token counts into a single matmul dispatch.

4. **Token-stationary (Scheme B)**: When $n_{bt} > 1$ (due to load imbalance), switch from sweeping all experts per bt tile to processing each expert once with its full token batch. This eliminates redundant weight loading at the cost of token replication ($k \times$).

5. **Reduce EP size**: Larger $E_{\text{local}}$ per device means more compute per communication. But weight memory grows $\propto EP$, may exceed HBM capacity.

**Optimal decode block config selection**:

$$\min_{bt, btc, \ldots} \quad T_{\text{total}} = n_{bt} \times \left[ \frac{E_{\text{local}} \times 3 \cdot H \cdot I \cdot \tau}{\Theta_{\text{HBM}}^{\text{eff}}} + E_{\text{local}} \times \frac{6 \cdot btc \cdot H \cdot I}{\Phi_{\text{MXU}}} \right]$$

Subject to:
- $bt \leq T_{\text{local}}$, $bt \mid T_{\text{local}}$
- $btc \leq bt$, $btc \mid bt$
- $VMEM \leq 64 \text{ MiB}$
- $btc \geq 128$ (MXU tile granularity constraint)

For decode, the HBM term dominates → choose **largest possible `bt`** to minimize $n_{bt}$, avoiding redundant weight loads.

#### 4.6.3 Prefill Optimal Design ($\bar{n}_e \gg btc$)

**Problem**: $\bar{n}_e \approx 512\text{-}2048$, $btc$ likely smaller → compute-bound or near-roofline. Weight loading cost is amortized over many tokens.

**Optimization strategies**:

1. **Maximize `btc`**: Set $btc = \min(\bar{n}_e, \text{VMEM-constrained limit})$ to maximize MXU utilization. Larger $btc$ → fewer sub-tiles → less DMA overhead.

2. **Token-stationary (Scheme B preferred)**: Keep tokens in VMEM, stream weights through. Only load each expert's weights once. Since $\bar{n}_e$ is large, the token replication cost ($k \times H \times \tau$) becomes material — choose $bt$ to balance token VMEM vs weight buffers.

3. **Double-buffer tokens**: When $bt$ is large (prefill), token double-buffering across bf/bd dimensions is critical to hide DMA latency.

4. **Overlap communication with compute**: Large token counts mean large all-to-all volumes. Consider starting scatter for the next bt tile while computing the current tile (if $n_{bt} > 1$).

**Prefill token-stationary VMEM constraint**:

$$\boxed{bt_{\text{max}} = \frac{64 \text{ MiB} - W_{\text{buffers}}}{H \times (\tau + 2 \times \tau_{\text{acc}} \times bf / H)}}$$

With $H=8192, \tau=2, \tau_{\text{acc}}=4, bf=1024, W_{\text{buffers}}=24 \text{ MB}$:
$bt_{\text{max}} \approx 2048$ tokens — sufficient for most prefill batches.

#### 4.6.4 Transition Point: When to Switch Strategy

The optimal strategy changes when $\bar{n}_e$ crosses the weight-loading / token-compute tradeoff:

$$\bar{n}_e^* \approx \frac{3 \cdot I \cdot \tau}{H \cdot \tau} = \frac{3 \cdot I}{H}$$

For Ling 2.6: $\bar{n}_e^* = 3 \times 2048 / 8192 = 0.75$. Since $\bar{n}_e \geq 1$ always, prefill-like strategies are *always* preferred for Ling 2.6. But the practical constraint is VMEM — $bt$ can't exceed ~2048.

The *effective* criterion is:

$$\boxed{\text{Strategy} = \begin{cases}
\text{Weight-streaming (Scheme A)} & \text{if } T_{\text{local}} \leq bt_{\text{max}} \text{ AND } n_{bt}=1 \\
\text{Token-stationary (Scheme B)} & \text{if } \bar{n}_e \text{ is large AND weight reload cost } > \text{ token replication cost} \\
\text{Hybrid} & \text{otherwise — batch small experts, stream large ones}
\end{cases}}$$

For the current Ling 2.6 config (decode, $T_{\text{local}}=256$, $bt=64$): Scheme A with $n_{bt}=1$ is optimal — weights loaded exactly once, simple control flow.

#### 4.6.5 Quantitative Comparison: Decode vs Prefill

| Metric | Decode (B=256) | Small Prefill (B=2048) | Large Prefill (B=8192) |
|--------|---------------|----------------------|----------------------|
| $T_{\text{local}}$ (EP=8) | 256 | 2048 | 8192 |
| $\bar{n}_e$ | 8 | 64 | 256 |
| $AI_{\text{useful}}$ | 8.4 | 67.1 | 268.4 |
| $AI_{\text{executed}}$ (btc=64) | 67.1 | 67.1 | 67.1 |
| HBM Read (weights) | 3.0 GB | 3.0 GB | 3.0 GB |
| HBM Read (tokens) | 32 MB | 256 MB | 1 GB |
| MXU FLOPs (exec) | 206 GFLOPS | 206 GFLOPS | 206 GFLOPS |
| MXU FLOPs (useful) | 26 GFLOPS | 206 GFLOPS | 206 GFLOPS |
| $T_{\text{HBM}}$ | 0.81 ms | 0.88 ms | 1.08 ms |
| $T_{\text{compute}}$ | 0.09 ms | 0.09 ms | 0.09 ms |
| Bottleneck | HBM-BW (9:1) | HBM-BW (10:1) | HBM-BW (12:1) |
| Optimal $btc$ | 64 (any) | 128 | 256+ |
| Quantization gain (FP8) | ~2× faster | ~2× faster | ~1.5× faster |

> **Key insight**: Even at large batch sizes, Ling 2.6 decode/prefill is HBM-bound because weight loading dominates (3 GB of weights vs at most ~1 GB of tokens). The AI is bounded by $\bar{n}_e$, which must exceed ~300 to reach the ridge point. The kernel is structural HBM-bound for this model architecture.

---

## References

- Existing Ling 2.6 analysis: `primatrix/wiki/fused_moe_ling2.6_theoretical_perf_analysis.md`
- TPU perf model steps: `primatrix/wiki/steps_fused_moe_per_expert.json`, `steps_fused_moe_bt_tile.json`
- Kernel implementation: `kernels/_fused_moe_impl.py`
- Benchmark results: `benchmark_results/benchmark_result.json`
- LLO dumps: `benchmark_results/ir_dumps/llo/`
