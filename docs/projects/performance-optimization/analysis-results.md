# 理论分析结果

# ALModel Parallelism Strategy Analysis

## Model

| Property | Value |
| --- | --- |
| Parameters | 17.43B (expert: 16.11B, non-expert: 1.33B) |
| Layers | 21 total (16 GLA + 5 MLA, 20 MoE + 1 dense MLP) |
| Experts | 256 x FFN 512, top\_k=8 |
| Attention | 16h x 128d, EMB\_DIM=2048 |

## Training

| Property | Value |
| --- | --- |
| Global batch size | 5120 |
| Sequence length | 4096 |
| Forward FLOPs/sample | 13.33 TFLOPs |
| Backward FLOPs/sample | 26.63 TFLOPs |
| Training FLOPs/sample | 39.96 TFLOPs |
| Training FLOPs/step | 204.57 PFLOPs |
| Remat overhead (full) | +33.4% |

## Hardware

| Property | Value |
| --- | --- |
| Devices | 128 x 1154 TFLOPs/device = 147648 TFLOPs total |
| HBM | 96 GiB/device |
| ICI bandwidth | 1200 GB/s bidirectional (nominal) |
| Measured BW (GB/s) | AR={2: 23.1, 4: 46.3, 8: 80.1}, AG={2: 34.3, 4: 89.9, 8: 186.3}, RS={2: 46.0, 4: 92.5, 8: 185.3}, A2A={2: 42.7, 4: 66.7, 8: 79.5} |
| PP ppermute | 600 GB/s (overlapped) |
| FSDP overlap | 85% |
| XLA reserve | 20% |
| Total memory (no parallelism) | W=64.9GB + O=129.9GB + G=64.9GB = 259.8GB |

## Configs with MFU >= 20% (sorted by step time)

```python
  REMAT_LABELS = {
      "save_all":                         "save_all",
      "minimal_with_context":             "min+ctx",
      "minimal":                          "minimal",
      "save_dot_with_context_except_mlp": "dot-mlp+ctx",
      "save_dot_except_mlpwi":            "dot-mlpwi",
      "save_dot_except_mlp":              "dot-mlp",
      "save_qkv_proj":                    "qkv_proj",
      "save_out_proj":                    "out_proj",
      "full":                             "full",
  }
```

| # | TP | DP | PP | EP | FSDP | CP | Remat | PDB | MB | GA | W(GB) | O(GB) | G(GB) | FBuf | Act(GB) | Rsv | Tot(GB) | Trn(PF) | Rmt% | Comp | TP(s) | EP(s) | FSDP+ | DP(s) | CP(s) | PP+(s) | Bub | Opt | Step(s) | MFU% | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 4 | 1 | 1 | 32 | 1 | out\_proj | 10 | 10 | 4 | 2.0 | 4.1 | 2.0 | 8.1 | 54.5 | 21.2 | 92.0 | 204.6 | 32 | 1.82 | 0.00 | 0.00 | 2.51 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 4.40 | 31.5 | FSDP |
| 2 | 1 | 1 | 1 | 1 | 128 | 1 | qkv\_proj | 10 | 10 | 4 | 0.5 | 1.0 | 0.5 | 8.3 | 61.8 | 21.6 | 93.8 | 204.6 | 29 | 1.78 | 0.00 | 0.00 | 2.64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.43 | 31.3 | FSDP |
| 3 | 1 | 2 | 1 | 1 | 64 | 1 | out\_proj | 10 | 10 | 4 | 1.0 | 2.0 | 1.0 | 8.3 | 54.5 | 20.1 | 86.9 | 204.6 | 32 | 1.82 | 0.00 | 0.00 | 2.58 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 4.44 | 31.2 | FSDP |
| 4 | 1 | 1 | 1 | 1 | 64 | 2 | out\_proj | 10 | 20 | 4 | 1.0 | 2.0 | 1.0 | 8.3 | 54.5 | 20.1 | 86.9 | 204.6 | 32 | 1.82 | 0.00 | 0.00 | 2.58 | 0.00 | 0.32 | 0.00 | 0.00 | 0.00 | 4.72 | 29.4 | FSDP |
| 5 | 1 | 2 | 1 | 1 | 32 | 2 | out\_proj | 10 | 20 | 4 | 2.0 | 4.1 | 2.0 | 8.1 | 54.5 | 21.2 | 92.0 | 204.6 | 32 | 1.82 | 0.00 | 0.00 | 2.51 | 0.09 | 0.32 | 0.00 | 0.00 | 0.00 | 4.74 | 29.2 | FSDP |
| 6 | 1 | 1 | 1 | 1 | 32 | 4 | out\_proj | 10 | 40 | 4 | 2.0 | 4.1 | 2.0 | 8.1 | 54.5 | 21.2 | 92.0 | 204.6 | 32 | 1.82 | 0.00 | 0.00 | 2.51 | 0.00 | 0.41 | 0.00 | 0.00 | 0.00 | 4.75 | 29.2 | FSDP |
| 7 | 1 | 8 | 1 | 1 | 16 | 1 | qkv\_proj | 8 | 8 | 5 | 4.1 | 8.1 | 4.1 | 7.9 | 49.4 | 22.1 | 95.6 | 204.6 | 29 | 1.78 | 0.00 | 0.00 | 3.39 | 0.09 | 0.00 | 0.00 | 0.00 | 0.01 | 5.27 | 26.3 | FSDP |
| 8 | 1 | 4 | 1 | 1 | 32 | 1 | dot-mlp+ctx | 8 | 8 | 5 | 2.0 | 4.1 | 2.0 | 8.1 | 54.7 | 21.3 | 92.2 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.62 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 5.40 | 25.7 | FSDP |
| 9 | 1 | 1 | 1 | 1 | 128 | 1 | dot-mlp+ctx | 8 | 8 | 5 | 0.5 | 1.0 | 0.5 | 8.3 | 54.7 | 19.5 | 84.6 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.45 | 25.4 | FSDP |
| 10 | 1 | 2 | 1 | 1 | 64 | 1 | dot-mlp+ctx | 8 | 8 | 5 | 1.0 | 2.0 | 1.0 | 8.3 | 54.7 | 20.1 | 87.1 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.71 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 5.46 | 25.4 | FSDP |
| 11 | 1 | 4 | 1 | 1 | 16 | 2 | qkv\_proj | 8 | 16 | 5 | 4.1 | 8.1 | 4.1 | 7.9 | 49.4 | 22.1 | 95.6 | 204.6 | 29 | 1.78 | 0.00 | 0.00 | 3.39 | 0.13 | 0.32 | 0.00 | 0.00 | 0.01 | 5.64 | 24.6 | FSDP |
| 12 | 1 | 1 | 1 | 1 | 16 | 8 | qkv\_proj | 8 | 64 | 5 | 4.1 | 8.1 | 4.1 | 7.9 | 49.4 | 22.1 | 95.6 | 204.6 | 29 | 1.78 | 0.00 | 0.00 | 3.39 | 0.00 | 0.47 | 0.00 | 0.00 | 0.01 | 5.66 | 24.5 | FSDP |
| 13 | 1 | 1 | 1 | 1 | 64 | 2 | dot-mlp+ctx | 8 | 16 | 5 | 1.0 | 2.0 | 1.0 | 8.3 | 54.7 | 20.1 | 87.1 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.71 | 0.00 | 0.32 | 0.00 | 0.00 | 0.00 | 5.73 | 24.2 | FSDP |
| 14 | 1 | 2 | 1 | 1 | 32 | 2 | dot-mlp+ctx | 8 | 16 | 5 | 2.0 | 4.1 | 2.0 | 8.1 | 54.7 | 21.3 | 92.2 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.62 | 0.09 | 0.32 | 0.00 | 0.00 | 0.00 | 5.74 | 24.1 | FSDP |
| 15 | 1 | 1 | 1 | 1 | 32 | 4 | dot-mlp+ctx | 8 | 32 | 5 | 2.0 | 4.1 | 2.0 | 8.1 | 54.7 | 21.3 | 92.2 | 204.6 | 23 | 1.71 | 0.00 | 0.00 | 3.62 | 0.00 | 0.41 | 0.00 | 0.00 | 0.00 | 5.75 | 24.1 | FSDP |
| 16 | 1 | 2 | 1 | 1 | 16 | 4 | qkv\_proj | 8 | 32 | 5 | 4.1 | 8.1 | 4.1 | 7.9 | 49.4 | 22.1 | 95.6 | 204.6 | 29 | 1.78 | 0.00 | 0.00 | 3.39 | 0.18 | 0.41 | 0.00 | 0.00 | 0.01 | 5.77 | 24.0 | FSDP |
| 17 | 1 | 1 | 1 | 8 | 16 | 1 | minimal | 5 | 5 | 8 | 0.8 | 1.6 | 0.8 | 3.0 | 66.2 | 21.7 | 94.0 | 204.6 | 11 | 1.54 | 0.00 | 4.40 | 0.23 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.17 | 22.5 | EP a2a |
| 18 | 1 | 16 | 1 | 8 | 1 | 1 | min+ctx | 1 | 1 | 40 | 12.4 | 24.9 | 12.4 | 0.0 | 13.6 | 19.0 | 82.3 | 204.6 | 7 | 1.48 | 0.00 | 4.40 | 0.00 | 0.29 | 0.00 | 0.00 | 0.00 | 0.02 | 6.20 | 22.3 | EP a2a |
| 19 | 1 | 1 | 1 | 8 | 16 | 1 | dot-mlp+ctx | 8 | 8 | 5 | 0.8 | 1.6 | 0.8 | 3.0 | 54.7 | 18.2 | 79.0 | 204.6 | 23 | 1.71 | 0.00 | 4.40 | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.25 | 22.2 | EP a2a |
| 20 | 1 | 1 | 1 | 8 | 16 | 1 | dot-mlp | 10 | 10 | 4 | 0.8 | 1.6 | 0.8 | 3.0 | 65.1 | 21.3 | 92.5 | 204.6 | 27 | 1.76 | 0.00 | 4.40 | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.28 | 22.1 | EP a2a |

### Column Legend

| Column | Description |
| --- | --- |
| TP / DP / PP / EP / FSDP / CP | Tensor / Data / Pipeline / Expert / Fully-Sharded-Data / Context parallelism degree |
| Remat | Rematerialization (activation checkpointing) policy |
| PDB | per\_device\_batch\_size, the MaxText config value for batch size per device |
| MB | Micro batch size per device |
| GA | Gradient accumulation steps |
| W(GB) | Model weights per device (GB) |
| O(GB) | Optimizer states per device (GB), including Adam mu + nu |
| G(GB) | Gradients per device (GB) |
| FBuf | FSDP all-gather prefetch buffer (GB), peak = 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Act(GB) | Activation memory per device (GB), depends on micro batch and remat policy |
| Rsv | XLA overhead reserve (GB), 20% of modeled memory for comm buffers & HLO temps |
| Tot(GB) | Total HBM usage per device (GB) = W + O + G + FBuf + Act + Rsv |
| Trn(PF) | Training FLOPs per step (PetaFLOPs), useful fwd+bwd without remat overhead |
| Rmt% | Remat overhead percentage, extra compute from reactivation recomputation |
| Comp | Compute time (s), actual FLOPs / (num\_devices x peak TFLOPs/device) |
| TP(s) | TP all-reduce communication time (s), non-overlappable |
| EP(s) | EP all-to-all communication time (s), non-overlappable |
| FSDP+ | FSDP exposed communication time (s), partially overlapped with compute (efficiency=85%) |
| DP(s) | DP gradient all-reduce time (s), non-overlappable |
| CP(s) | CP KV all-gather communication time (s), non-overlappable |
| PP+(s) | PP ppermute communication time (s), fully overlapped by XLA scheduler |
| Bub(s) | PP bubble idle time (s), formula: (PP-1)/(num\_repeats x GA + PP-1) |
| Opt | Optimizer step time (s), memory-bound AdamW (28B/param, HBM\_BW=3690GB/s) |
| Step(s) | Total step time (s) = Comp + all comm + Bub + Opt |
| MFU% | Model FLOPs Utilization (%), useful TFLOPs / (step\_time x num\_devices x peak TFLOPs) |
| Bottleneck | The dominant time component limiting throughput |

### #1: TP=1 DP=4 PP=1 EP=1 FSDP=32 CP=1 remat=save\_out\_proj

#### Memory per device

| Component | Size (GB) | Detail |
| --- | --- | --- |
| Weights | 2.03 | expert: 1.88 + non-expert: 0.15 |
| Optimizer | 4.06 |  |
| Gradients | 2.03 |  |
| FSDP buffer | 8.14 | all-gather peak: 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Activations | 54.53 | micro\_batch=10 x inflight=1, remat=save\_out\_proj |
| XLA reserve | 21.24 | 30% of modeled |
| **Total** | **92.02** | **/ 96 GiB HBM (96% used)** |

**Batch**: per\_device\_batch=10, micro\_batch=10 (PDB×TP×CP), GA=4, batch\_par=128, effective\_batch/dev=40

**FLOPs**: useful=204.6 PFLOPs, actual=269.1 PFLOPs, remat=+32%

#### Communication

| Type | Volume (GB) | Time (s) | Note |
| --- | --- | --- | --- |
| TP all-reduce | 0.0 | 0.00 | non-overlappable |
| EP all-to-all | 0.0 | 0.00 | non-overlappable |
| FSDP gather+RS | 754.9 | 4.06 total, 2.51 exposed | overlap=85% |
| DP all-reduce | 3.0 | 0.07 | non-overlappable |
| CP all-gather (KV) | 0.0 | 0.00 | non-overlappable |
| PP ppermute (ICI) | 0.0 | 0.00 | fully overlapped |
| Optimizer (AdamW) | 14.2 | 0.00 | memory-bound, 28B/param, HBM=3690GB/s |

#### Performance

* **Step time**: 4.402s (compute=1.823 + comm=2.576 + optimizer=0.004)

* **MFU**: 31.5%

* **Bottleneck**: FSDP

### #2: TP=1 DP=1 PP=1 EP=1 FSDP=128 CP=1 remat=save\_qkv\_proj

#### Memory per device

| Component | Size (GB) | Detail |
| --- | --- | --- |
| Weights | 0.51 | expert: 0.47 + non-expert: 0.04 |
| Optimizer | 1.01 |  |
| Gradients | 0.51 |  |
| FSDP buffer | 8.33 | all-gather peak: 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Activations | 61.80 | micro\_batch=10 x inflight=1, remat=save\_qkv\_proj |
| XLA reserve | 21.65 | 30% of modeled |
| **Total** | **93.81** | **/ 96 GiB HBM (98% used)** |

**Batch**: per\_device\_batch=10, micro\_batch=10 (PDB×TP×CP), GA=4, batch\_par=128, effective\_batch/dev=40

**FLOPs**: useful=204.6 PFLOPs, actual=263.4 PFLOPs, remat=+29%

#### Communication

| Type | Volume (GB) | Time (s) | Note |
| --- | --- | --- | --- |
| TP all-reduce | 0.0 | 0.00 | non-overlappable |
| EP all-to-all | 0.0 | 0.00 | non-overlappable |
| FSDP gather+RS | 773.2 | 4.16 total, 2.64 exposed | overlap=85% |
| DP all-reduce | 0.0 | 0.00 | non-overlappable |
| CP all-gather (KV) | 0.0 | 0.00 | non-overlappable |
| PP ppermute (ICI) | 0.0 | 0.00 | fully overlapped |
| Optimizer (AdamW) | 3.6 | 0.00 | memory-bound, 28B/param, HBM=3690GB/s |

#### Performance

* **Step time**: 4.426s (compute=1.784 + comm=2.641 + optimizer=0.001)

* **MFU**: 31.3%

* **Bottleneck**: FSDP

### #3: TP=1 DP=2 PP=1 EP=1 FSDP=64 CP=1 remat=save\_out\_proj

#### Memory per device

| Component | Size (GB) | Detail |
| --- | --- | --- |
| Weights | 1.01 | expert: 0.94 + non-expert: 0.08 |
| Optimizer | 2.03 |  |
| Gradients | 1.01 |  |
| FSDP buffer | 8.27 | all-gather peak: 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Activations | 54.53 | micro\_batch=10 x inflight=1, remat=save\_out\_proj |
| XLA reserve | 20.06 | 30% of modeled |
| **Total** | **86.91** | **/ 96 GiB HBM (91% used)** |

**Batch**: per\_device\_batch=10, micro\_batch=10 (PDB×TP×CP), GA=4, batch\_par=128, effective\_batch/dev=40

**FLOPs**: useful=204.6 PFLOPs, actual=269.1 PFLOPs, remat=+32%

#### Communication

| Type | Volume (GB) | Time (s) | Note |
| --- | --- | --- | --- |
| TP all-reduce | 0.0 | 0.00 | non-overlappable |
| EP all-to-all | 0.0 | 0.00 | non-overlappable |
| FSDP gather+RS | 767.1 | 4.12 total, 2.58 exposed | overlap=85% |
| DP all-reduce | 1.0 | 0.04 | non-overlappable |
| CP all-gather (KV) | 0.0 | 0.00 | non-overlappable |
| PP ppermute (ICI) | 0.0 | 0.00 | fully overlapped |
| Optimizer (AdamW) | 7.1 | 0.00 | memory-bound, 28B/param, HBM=3690GB/s |

#### Performance

* **Step time**: 4.444s (compute=1.823 + comm=2.620 + optimizer=0.002)

* **MFU**: 31.2%

* **Bottleneck**: FSDP

### #4: TP=1 DP=1 PP=1 EP=1 FSDP=64 CP=2 remat=save\_out\_proj

#### Memory per device

| Component | Size (GB) | Detail |
| --- | --- | --- |
| Weights | 1.01 | expert: 0.94 + non-expert: 0.08 |
| Optimizer | 2.03 |  |
| Gradients | 1.01 |  |
| FSDP buffer | 8.27 | all-gather peak: 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Activations | 54.53 | micro\_batch=20 x inflight=1, remat=save\_out\_proj |
| XLA reserve | 20.06 | 30% of modeled |
| **Total** | **86.91** | **/ 96 GiB HBM (91% used)** |

**Batch**: per\_device\_batch=10, micro\_batch=20 (PDB×TP×CP), GA=4, batch\_par=128, effective\_batch/dev=80

**FLOPs**: useful=204.6 PFLOPs, actual=269.1 PFLOPs, remat=+32%

#### Communication

| Type | Volume (GB) | Time (s) | Note |
| --- | --- | --- | --- |
| TP all-reduce | 0.0 | 0.00 | non-overlappable |
| EP all-to-all | 0.0 | 0.00 | non-overlappable |
| FSDP gather+RS | 767.1 | 4.12 total, 2.58 exposed | overlap=85% |
| DP all-reduce | 0.0 | 0.00 | non-overlappable |
| CP all-gather (KV) | 12.5 | 0.32 | non-overlappable |
| PP ppermute (ICI) | 0.0 | 0.00 | fully overlapped |
| Optimizer (AdamW) | 7.1 | 0.00 | memory-bound, 28B/param, HBM=3690GB/s |

#### Performance

* **Step time**: 4.718s (compute=1.823 + comm=2.894 + optimizer=0.002)

* **MFU**: 29.4%

* **Bottleneck**: FSDP

### #5: TP=1 DP=2 PP=1 EP=1 FSDP=32 CP=2 remat=save\_out\_proj

#### Memory per device

| Component | Size (GB) | Detail |
| --- | --- | --- |
| Weights | 2.03 | expert: 1.88 + non-expert: 0.15 |
| Optimizer | 4.06 |  |
| Gradients | 2.03 |  |
| FSDP buffer | 8.14 | all-gather peak: 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Activations | 54.53 | micro\_batch=20 x inflight=1, remat=save\_out\_proj |
| XLA reserve | 21.24 | 30% of modeled |
| **Total** | **92.02** | **/ 96 GiB HBM (96% used)** |

**Batch**: per\_device\_batch=10, micro\_batch=20 (PDB×TP×CP), GA=4, batch\_par=128, effective\_batch/dev=80

**FLOPs**: useful=204.6 PFLOPs, actual=269.1 PFLOPs, remat=+32%

#### Communication

| Type | Volume (GB) | Time (s) | Note |
| --- | --- | --- | --- |
| TP all-reduce | 0.0 | 0.00 | non-overlappable |
| EP all-to-all | 0.0 | 0.00 | non-overlappable |
| FSDP gather+RS | 754.9 | 4.06 total, 2.51 exposed | overlap=85% |
| DP all-reduce | 2.0 | 0.09 | non-overlappable |
| CP all-gather (KV) | 12.5 | 0.32 | non-overlappable |
| PP ppermute (ICI) | 0.0 | 0.00 | fully overlapped |
| Optimizer (AdamW) | 14.2 | 0.00 | memory-bound, 28B/param, HBM=3690GB/s |

#### Performance

* **Step time**: 4.743s (compute=1.823 + comm=2.916 + optimizer=0.004)

* **MFU**: 29.2%

* **Bottleneck**: FSDP

## Worst 10 configurations (slowest)

10 of 10 shown

| # | TP | DP | PP | EP | FSDP | CP | Remat | PDB | MB | GA | W(GB) | O(GB) | G(GB) | FBuf | Act(GB) | Rsv | Tot(GB) | Trn(PF) | Rmt% | Comp | TP(s) | EP(s) | FSDP+ | DP(s) | CP(s) | PP+(s) | Bub | Opt | Step(s) | MFU% | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2 | 1 | 2 | 4 | 8 | save\_all | 1 | 8 | 40 | 8.7 | 17.5 | 8.7 | 4.0 | 19.3 | 17.5 | 75.8 | 204.6 | 0 | 1.39 | 0.00 | 4.68 | 33.47 | 0.38 | 0.47 | 0.00 | 0.00 | 0.02 | 40.41 | 3.4 | FSDP |
| 2 | 1 | 4 | 1 | 2 | 4 | 4 | save\_all | 1 | 4 | 40 | 8.7 | 17.5 | 8.7 | 4.0 | 20.2 | 17.8 | 77.0 | 204.6 | 0 | 1.39 | 0.00 | 4.68 | 33.47 | 0.28 | 0.41 | 0.00 | 0.00 | 0.02 | 40.25 | 3.4 | FSDP |
| 3 | 1 | 8 | 1 | 2 | 4 | 2 | save\_all | 1 | 2 | 40 | 8.7 | 17.5 | 8.7 | 4.0 | 22.1 | 18.3 | 79.4 | 204.6 | 0 | 1.39 | 0.00 | 4.68 | 33.47 | 0.19 | 0.32 | 0.00 | 0.00 | 0.02 | 40.07 | 3.5 | FSDP |
| 4 | 1 | 16 | 1 | 2 | 4 | 1 | save\_all | 1 | 1 | 40 | 8.7 | 17.5 | 8.7 | 4.0 | 25.8 | 19.5 | 84.3 | 204.6 | 0 | 1.39 | 0.00 | 4.68 | 33.47 | 0.20 | 0.00 | 0.00 | 0.00 | 0.02 | 39.76 | 3.5 | FSDP |
| 5 | 1 | 8 | 1 | 1 | 16 | 1 | save\_all | 1 | 1 | 40 | 4.1 | 8.1 | 4.1 | 7.9 | 25.8 | 15.0 | 64.9 | 204.6 | 0 | 1.39 | 0.00 | 0.00 | 38.11 | 0.09 | 0.00 | 0.00 | 0.00 | 0.01 | 39.59 | 3.5 | FSDP |
| 6 | 1 | 2 | 1 | 1 | 8 | 8 | save\_all | 1 | 8 | 40 | 8.1 | 16.2 | 8.1 | 7.3 | 19.3 | 17.7 | 76.8 | 204.6 | 0 | 1.39 | 0.00 | 0.00 | 35.49 | 0.35 | 0.47 | 0.00 | 0.00 | 0.02 | 37.71 | 3.7 | FSDP |
| 7 | 1 | 4 | 1 | 1 | 8 | 4 | save\_all | 1 | 4 | 40 | 8.1 | 16.2 | 8.1 | 7.3 | 20.2 | 18.0 | 78.1 | 204.6 | 0 | 1.39 | 0.00 | 0.00 | 35.49 | 0.26 | 0.41 | 0.00 | 0.00 | 0.02 | 37.56 | 3.7 | FSDP |
| 8 | 1 | 2 | 1 | 4 | 2 | 8 | save\_all | 1 | 8 | 40 | 10.0 | 19.9 | 10.0 | 1.9 | 19.3 | 18.3 | 79.4 | 204.6 | 0 | 1.39 | 0.00 | 4.50 | 30.74 | 0.43 | 0.47 | 0.00 | 0.00 | 0.02 | 37.55 | 3.7 | FSDP |
| 9 | 1 | 8 | 1 | 1 | 8 | 2 | save\_all | 1 | 2 | 40 | 8.1 | 16.2 | 8.1 | 7.3 | 22.1 | 18.6 | 80.5 | 204.6 | 0 | 1.39 | 0.00 | 0.00 | 35.49 | 0.18 | 0.32 | 0.00 | 0.00 | 0.02 | 37.38 | 3.7 | FSDP |
| 10 | 1 | 4 | 1 | 4 | 2 | 4 | save\_all | 1 | 4 | 40 | 10.0 | 19.9 | 10.0 | 1.9 | 20.2 | 18.6 | 80.7 | 204.6 | 0 | 1.39 | 0.00 | 4.50 | 30.74 | 0.32 | 0.41 | 0.00 | 0.00 | 0.02 | 37.38 | 3.7 | FSDP |

### Column Legend

| Column | Description |
| --- | --- |
| TP / DP / PP / EP / FSDP / CP | Tensor / Data / Pipeline / Expert / Fully-Sharded-Data / Context parallelism degree |
| Remat | Rematerialization (activation checkpointing) policy |
| PDB | per\_device\_batch\_size, the MaxText config value for batch size per device |
| MB | Actual physical per-device batch = PDB × TP × CP (batch not sharded by tensor/context axes) |
| GA | Gradient accumulation steps |
| W(GB) | Model weights per device (GB) |
| O(GB) | Optimizer states per device (GB), including Adam mu + nu |
| G(GB) | Gradients per device (GB) |
| FBuf | FSDP all-gather prefetch buffer (GB), peak = 2 layers x full\_layer\_weight x (FSDP-1)/FSDP |
| Act(GB) | Activation memory per device (GB), depends on micro batch and remat policy |
| Rsv | XLA overhead reserve (GB), 30% of modeled memory for comm buffers & HLO temps |
| Tot(GB) | Total HBM usage per device (GB) = W + O + G + FBuf + Act + Rsv |
| Trn(PF) | Training FLOPs per step (PetaFLOPs), useful fwd+bwd without remat overhead |
| Rmt% | Remat overhead percentage, extra compute from reactivation recomputation |
| Comp | Compute time (s), actual FLOPs / (num\_devices x peak TFLOPs/device) |
| TP(s) | TP all-reduce communication time (s), non-overlappable |
| EP(s) | EP all-to-all communication time (s), non-overlappable |
| FSDP+ | FSDP exposed communication time (s), partially overlapped with compute (efficiency=85%) |
| DP(s) | DP gradient all-reduce time (s), non-overlappable |
| CP(s) | CP KV all-gather communication time (s), non-overlappable |
| PP+(s) | PP ppermute communication time (s), fully overlapped by XLA scheduler |
| Bub(s) | PP bubble idle time (s), formula: (PP-1)/(num\_repeats x GA + PP-1) |
| Opt | Optimizer step time (s), memory-bound AdamW (28B/param, HBM\_BW=3690GB/s) |
| Step(s) | Total step time (s) = Comp + all comm + Bub + Opt |
| MFU% | Model FLOPs Utilization (%), useful TFLOPs / (step\_time x num\_devices x peak TFLOPs) |
| Bottleneck | The dominant time component limiting throughput |

## Summary

* **Best config**: TP=1 DP=4 PP=1 EP=1 FSDP=32 CP=1 remat=save\_out\_proj

* **Best step time**: 4.402s, MFU: 31.5%

* **MFU range**: 3.4% - 31.5%

* **Step time range**: 4.402s - 40.406s
