# RFC: Multi-Token Prediction (MTP) for MaxText

## Overview

Add Multi-Token Prediction (MTP) auxiliary training objective to MaxText, following the architecture described in [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437). MTP enables the model to predict multiple future tokens simultaneously through auxiliary prediction layers that share the main model's embedding and decoder layer architecture. The implementation supports segment-aware rolling for multi-document packing, configurable loss normalization, AQT quantization, MoE integration (load-balance loss collection and expert bias updates from MTP layers), and numerical alignment with Megatron-LM.

## Motivation & Goals

### Current Problems

1. **Single-token prediction limitation**: Standard language model training only optimizes for next-token prediction (t+1). This underutilizes the training signal available from each sequence, as the model could learn richer representations by also predicting further-ahead tokens (t+2, t+3, ...).

1. **Speculative decoding readiness**: MTP heads trained alongside the main model can serve as draft heads for speculative decoding at inference time, potentially accelerating generation without requiring a separate draft model.

1. **Megatron-LM migration**: Teams migrating from Megatron-LM to MaxText need MTP behavior that numerically aligns with Megatron's implementation, including per-layer loss normalization, final layernorm placement, and double-normalization avoidance.

### Goals

- **Configurable MTP layers**: Support 1 to N auxiliary prediction layers via `mtp_num_layers` config
- **Segment-aware rolling**: Correctly handle multi-document packing (`reset_attention_mask=True`) by preventing cross-document information leakage during token shifting
- **Megatron-LM alignment**: Produce numerically equivalent forward/backward passes when configured with matching options (`mtp_final_layernorm`, `mtp_per_layer_loss_norm`)
- **Zero impact when disabled**: With `mtp_num_layers=0` (default), no code paths are activated and no parameters are added

## Design & Technical Details

### Architecture Overview

```text
Main Decoder:
  input_tokens → Embedding → Decoder Layers → hidden_state → Logits (predict t+1)
                                                    │
                                                    ▼
MTP Block (side-car, training only):
  for k = 1 to mtp_num_layers:
    ┌──────────────────────────────────────────────────────┐
    │  Roll input_ids, target_ids, mask, position left by 1│
    │  (segment-aware: zero out at document boundaries)    │
    ├──────────────────────────────────────────────────────┤
    │  e_target = Embedding(rolled_input_ids)              │
    │  h_k = TransformerLayer(                             │
    │          W_p(concat(RMSNorm(h_{k-1}),               │
    │                     RMSNorm(e_target)))              │
    │        )                                             │
    │  [optional] h_k = RMSNorm(h_k)   # final_layernorm  │
    │  logits_k = OutputProjection(h_k)                    │
    │  loss_k = CrossEntropy(logits_k, rolled_target_ids)  │
    └──────────────────────────────────────────────────────┘
    h_{k-1} ← h_k   (chain hidden states across layers)

Loss Combination:
  total_loss = main_loss + mtp_loss_scaling_factor * avg_mtp_loss
```

MTP operates as a "side-car" module: it sows losses into intermediate outputs but does not alter the primary model's logits or hidden states.

### MultiTokenPredictionLayer

Each MTP layer implements:

```text
h_next = TransformerLayer(W_p(concat(RMSNorm(h_prev), RMSNorm(e_target))))
```

**Components:**

| Component | Type | Purpose |
| ------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------- |
| `embedding_norm` | RMSNorm | Normalize target token embedding |
| `hidden_state_norm` | RMSNorm | Normalize previous hidden state |
| `projection_layer` | DenseGeneral(2\*emb_dim → emb_dim) | Project concatenated features back to model dimension |
| `transformer_layer` | DecoderLayer (Linen) wrapped via ToNNX | Full transformer layer (attention + FFN), supports AQT quantization via `quant` parameter |
| `final_layernorm` | RMSNorm (optional) | Per-layer output normalization for Megatron alignment |

**Property-based naming**: Attributes are stored with 1-indexed names (e.g., `mtp_1_embedding_norm`) via Python property setters, matching the paper's convention and ensuring clean checkpoint keys.

### MultiTokenPredictionBlock

Orchestrates the sequential execution of MTP layers. Key responsibilities:

1. **Rolling**: Shift `input_ids`, `target_ids`, `target_mask`, `position_ids` left by one position per iteration, using segment-aware rolling when `decoder_segment_ids` is available
1. **Forward pass**: Chain hidden states through MTP layers
1. **Loss collection**: Store per-layer losses and weights via `self.sow()` with custom NNX Variable types (`mtp_losses`, `mtp_acceptance`) for extraction by the training loop. Using `sow()` instead of direct Variable attribute assignment avoids checkpoint template issues.
1. **Eval metrics**: Store predictions for acceptance rate calculation

### Token Rolling Mechanism

MTP requires shifting the prediction target forward by k positions for the k-th layer. This is achieved by iteratively left-rolling the sequence arrays.

#### Simple Rolling (`roll_and_mask`)

```python
def roll_and_mask(x, shift=-1):
    return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)
```

Left-shifts the sequence by one position and zeros out the last position (no valid target).

#### Segment-Aware Rolling (`roll_and_mask_by_segment`)

In multi-document packing, a sequence contains multiple documents:

```text
token:       [A1] [A2] [A3] [B1] [B2] [B3]
segment_ids: [1]  [1]  [1]  [2]  [2]  [2]
              ↑── doc 1 ──↑  ↑── doc 2 ──↑
```

Simple rolling would cause position 2 (last token of doc 1) to receive `B1` (first token of doc 2) as its target — cross-document leakage. Segment-aware rolling detects document boundaries and zeros out those positions:

```text
Simple roll:         [A2] [A3] [B1] [B2] [B3] [0]
                                ↑ leaked from doc 2

Segment-aware roll:  [A2] [A3] [0]  [B2] [B3] [0]
                                ↑ zeroed at boundary
```

**Boundary detection logic:**

```python
seg_current = segment_ids
seg_next = roll(segment_ids, -1)  # what segment is at position i+1
is_boundary = (seg_current != seg_next) | (seg_current == 0)
result = jnp.where(is_boundary, 0, rolled)
```

#### Three Segment Variables in Training

| Variable | Mutates? | Purpose |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `decoder_segment_ids` | No | Passed to MTP transformer layers for attention masking; stored for eval |
| `rolled_segment_ids` | Left-shifted each iteration | Tracks where document boundaries are in the currently-shifted data, used by `roll_and_mask_by_segment` |
| `self.segment_ids` | Stored once (original value) | Passed to `calculate_mtp_acceptance_rate` for segment-aware alignment of main model predictions |

`rolled_segment_ids` itself is shifted using simple `roll_and_mask` (not segment-aware), because the segment IDs are boundary definitions — there is no semantic leakage risk, and simple shifting correctly tracks boundary movement.

### Loss Computation

#### Per-Layer Loss Collection

Each MTP layer computes masked cross-entropy:

```python
mtp_xent, _ = cross_entropy_with_logits(
    mtp_logits, one_hot(rolled_target_ids, vocab_size), 0.0
)
mtp_xent_masked = mtp_xent * rolled_target_mask
mtp_losses_list.append(jnp.sum(mtp_xent_masked))  # sum of losses
mtp_weights_list.append(jnp.sum(rolled_target_mask))  # count of valid tokens
```

#### Normalization Modes

**Global normalization** (default, `mtp_per_layer_loss_norm=False`):

```text
avg_mtp_loss = sum(all_layer_losses) / sum(all_layer_weights)
```

All tokens across all layers contribute equally. Layers with more valid tokens have more influence.

**Per-layer normalization** (`mtp_per_layer_loss_norm=True`, Megatron-LM style):

```text
per_layer_avg = [loss_k / weight_k for k in layers]
avg_mtp_loss = mean(per_layer_avg)
```

Each layer contributes equally regardless of valid token count. This matters because later MTP layers have fewer valid tokens (more positions zeroed out from rolling).

#### Final Loss Combination

```python
total_loss = main_loss + mtp_loss_scaling_factor * avg_mtp_loss
```

With gradient accumulation, MTP loss is scaled by `total_weights` to match the main loss convention (raw sum, divided later).

### Output Projection and Double Normalization

When `mtp_final_layernorm=False` (default):

```text
MTP hidden → apply_output_head() → decoder_norm → projection → logits
```

When `mtp_final_layernorm=True` (Megatron alignment):

```text
MTP hidden → final_layernorm → apply_output_projection() → projection → logits
                                (no decoder_norm)
```

`apply_output_projection()` is a dedicated method that skips the shared `decoder_norm`, preventing double normalization when the MTP layer already has its own final layernorm. This matches Megatron-LM's architecture where `TransformerBlock` applies `final_layernorm` before the output projection.

Similarly, the main model's hidden state is passed through `apply_decoder_norm` before entering the MTP block when `mtp_final_layernorm=True`, matching Megatron's flow where MTP receives post-norm hidden states.

### Eval: Acceptance Rate

`calculate_mtp_acceptance_rate` measures how well the MTP head's predictions agree with the main model's predictions:

1. Get main model predictions: `argmax(logits)`
1. Roll main predictions forward by `mtp_eval_target_module` steps (segment-aware)
1. Compare with MTP layer predictions: `agreement = (mtp_preds == rolled_main_preds) * valid_mask`
1. Return `sum(agreement) / sum(valid_mask) * 100` (percentage)

Only computed when `mtp_eval_target_module > 0`, and only for the specified layer.

### Intermediate Output Transport

MTP uses custom NNX Variable types as a side-channel to pass data out of the forward pass without affecting the model's primary output:

```python
class mtp_losses(nnx.Variable):    # → 'mtp_losses' Linen collection
class mtp_acceptance(nnx.Variable): # → 'mtp_acceptance' Linen collection
```

These are stored via `self.sow()` (not direct attribute assignment) to ensure they are always materialized tracers and avoid checkpoint template issues. They are automatically converted to Linen mutable collections by the `ToLinen` wrapper and are transient (not part of checkpoints) — recomputed on every forward pass.

```python
# Inside MultiTokenPredictionBlock.__call__():
self.sow(mtp_losses, "losses", jnp.stack(mtp_losses_list))
self.sow(mtp_losses, "weights", jnp.stack(mtp_weights_list))
```

| Variable | Collection | Contents |
| ------------- | ---------------- | ------------------------------------------- |
| `losses` | `mtp_losses` | Per-layer loss sums `[num_layers]` |
| `weights` | `mtp_losses` | Per-layer valid token counts `[num_layers]` |
| `mtp_preds` | `mtp_acceptance` | Predictions from target eval layer |
| `mtp_mask` | `mtp_acceptance` | Valid token mask for eval |
| `segment_ids` | `mtp_acceptance` | Original segment IDs for eval rolling |

### Model Integration

#### Initialization (models.py)

```python
if self.config.mtp_num_layers > 0:
    layer_types = self.decoder.get_decoder_layers()
    mtp_layer = layer_types[-1]  # reuse last decoder layer blueprint
    self.mtp_block = multi_token_prediction_block_as_linen(
        config=self.config,
        mesh=self.mesh,
        transformer_layer_module=mtp_layer,
        decoder=self.decoder,
        rngs=self.make_rng("mtp_block"),
        quant=self.quant,
    )
```

MTP reuses the same `DecoderLayer` architecture as the main model for architectural consistency. The Linen `DecoderLayer` is wrapped via `nnx_wrappers.ToNNX` + `lazy_init()` inside each `MultiTokenPredictionLayer`, enabling it to work within the NNX-based MTP module hierarchy. The `quant` parameter is threaded through to support AQT quantization. The `decoder` reference provides access to `shared_embedding`, `apply_output_head`, and `apply_output_projection`.

#### Forward Pass (models.py)

MTP block is called after the main decoder forward pass. It receives:

- `main_hidden_state`: Final hidden state from the decoder (optionally post-norm)
- `shared_embedding`: For computing target token embeddings and output projection
- Original `input_ids`, `target_ids`, `target_mask`, `position_ids`, `decoder_segment_ids`

#### Training Loop (train.py)

```python
if config.mtp_num_layers > 0 and is_train:
    mtp_loss, raw_mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss  # scaled by mtp_loss_scaling_factor
```

`raw_mtp_loss` (unscaled) is logged separately for monitoring.

**MoE + MTP integration**: When the model uses MoE (`num_experts > 1`) with MTP, the training loop also:

1. Collects MoE load-balance losses (`moe_lb_loss`) from MTP transformer layers and adds them to the total loss
1. Collects MoE expert counts (`mtp_expert_counts`) from MTP layers for routed bias updates
1. After `apply_gradients`, applies expert bias updates to MTP MoE gate parameters

This matches Megatron-LM's recursive module traversal which updates all routers including those inside MTP layers.

#### Gradient Accumulation (gradient_accumulation.py)

When gradient accumulation is enabled (`gradient_accumulation_steps > 1`):

- `mtp_loss` and `raw_mtp_loss` are accumulated across microbatches and normalized by `gradient_accumulation_steps` for consistent reporting with GA=1
- `moe_expert_counts` and `mtp_expert_counts` are averaged (not summed) across microbatches to prevent the effective update rate from being amplified by the number of GA steps

## Megatron-LM Alignment Fixes

### MTP MoE Expert Bias Updates

**Problem**: Megatron's `_update_router_expert_bias` recursively traverses all modules including MTP layers. MaxText left MTP MoE gate biases frozen at zero, causing ~0.5% systematic MTP loss drift.

**Fix**: Collect `mtp_expert_counts` from MTP transformer layers and apply `expert_counts_to_bias_update` to MTP MoE gate bias parameters after `apply_gradients`, mirroring the backbone MoE bias update logic.

## Configuration

| Field | Type | Default | Description |
| ------------------------- | ---------- | ------- | -------------------------------------------------------------------------- |
| `mtp_num_layers` | int >= 0 | 0 | Number of auxiliary MTP layers. 0 disables MTP entirely. |
| `mtp_loss_scaling_factor` | float >= 0 | 0.1 | Scaling factor (lambda) for the MTP auxiliary loss |
| `mtp_eval_target_module` | int >= 0 | 0 | Which MTP layer (1-indexed) to use for acceptance rate eval. 0 disables. |
| `mtp_final_layernorm` | bool | False | Apply RMSNorm after each MTP transformer layer output (Megatron alignment) |
| `mtp_per_layer_loss_norm` | bool | False | Normalize loss per-layer independently before averaging (Megatron style) |

### Example Configuration

```yaml
# Enable MTP with 1 auxiliary layer (DeepSeek-V3 style)
mtp_num_layers: 1
mtp_loss_scaling_factor: 0.1
mtp_eval_target_module: 1

# For Megatron-LM alignment
mtp_final_layernorm: true
mtp_per_layer_loss_norm: true
```

## File Inventory & Change Description

### New Files

| File | Lines | Description |
| ---------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/maxtext/layers/multi_token_prediction.py` | ~523 | Core implementation: `roll_and_mask`, `roll_and_mask_by_segment`, `MultiTokenPredictionLayer` (with ToNNX wrapping and quant support), `MultiTokenPredictionBlock` (with `sow()` output), `calculate_mtp_loss`, `calculate_mtp_acceptance_rate`, `multi_token_prediction_block_as_linen` |
| `tests/unit/multi_token_prediction_test.py` | ~670+ | Unit tests: layer forward pass, block sow functionality, loss aggregation, roll_and_mask variants, segment-aware rolling, loss normalization modes, output projection routing, GA metrics |
| `tests/unit/test_mtp_megatron_alignment.py` | ~1290 | Megatron-MaxText numerical alignment tests (L1-L7): sub-module, forward pass, backward pass, rolling, block loss, full pipeline, optimizer updates |

### Modified Files

| File | Description |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/maxtext/models/models.py` | MTP block instantiation (with `quant` passthrough) and forward pass integration in `TransformerPure` |
| `src/maxtext/trainers/pre_train/train.py` | MTP loss calculation; MoE lb_loss collection from MTP layers; MTP expert_counts collection and bias updates; acceptance rate eval logging |
| `src/maxtext/layers/decoders.py` | Add `apply_output_projection()` method for norm-free logit projection; add `apply_decoder_norm()` method |
| `src/maxtext/utils/gradient_accumulation.py` | Add `raw_mtp_loss` accumulation; normalize `mtp_loss`/`raw_mtp_loss` by GA steps; average `moe_expert_counts`/`mtp_expert_counts` across microbatches |
| `src/maxtext/configs/types.py` | Add `MTP` config class with all MTP-related fields |
| `src/maxtext/configs/base.yml` | Add MTP config defaults |

## Backward Compatibility

### Fully Compatible

- **Default off**: `mtp_num_layers=0` (default) activates no MTP code paths. No extra parameters, no extra computation, no model output changes.
- **No checkpoint impact**: MTP intermediate outputs (`mtp_losses`, `mtp_acceptance`) are transient NNX Variables, not part of checkpoint state. Existing checkpoints load without modification.
- **Additive changes only**: `apply_output_projection()` is a new method that does not alter existing `apply_output_head()` behavior.

### Notes

- **MTP parameters are extra**: When `mtp_num_layers > 0`, the model gains additional trainable parameters (one transformer layer + projection + norms per MTP layer). These are not present in non-MTP checkpoints and must be initialized from scratch or converted from Megatron MTP weights.
- **Training-only overhead**: MTP layers are only active during training. Eval only computes acceptance rate metrics (no extra transformer forward passes beyond the target module's stored predictions).

## Test Plan

### Test Matrix

#### 1. Unit Tests (`multi_token_prediction_test.py`)

- **MultiTokenPredictionLayerTest**: Single layer forward pass (with DecoderLayer + ToNNX), output shape/dtype, NaN/Inf checks, 1-indexed naming
- **MultiTokenPredictionBlockTest**: Multi-layer block, sow functionality (losses/weights correctly stored via `sow()`), loss aggregation logic
- **TestRollAndMask**: `roll_and_mask` basic behavior, `roll_and_mask_by_segment` document boundary handling, `segment_ids=None` fallback
- **TestCalculateMtpLoss**: Global normalization, per-layer normalization (Megatron style), empty/None inputs, array format (NNX Variable) vs tuple format (Linen sow)
- **TestOutputProjectionMode**: `mtp_final_layernorm=False` routes to `apply_output_head`, `mtp_final_layernorm=True` routes to `apply_output_projection` (no double-norm)
- **TestGradientAccumulationMtpMetrics**: `raw_mtp_loss` normalization by GA steps, `mtp_expert_counts` averaging across microbatches, `None` expert counts safely skipped

#### 2. Megatron Alignment Tests (`test_mtp_megatron_alignment.py`)

Requires `megatron-core` + `torch`; skipped via `pytest.importorskip` otherwise.

- L1: Sub-module alignment (RMSNorm, Linear projection, SwiGLU MLP)
- L2: MTP layer forward pass (zeroed transformer for architecture isolation)
- L3: MTP layer parameter gradient alignment
- L4: `roll_tensor` vs `roll_and_mask` consistency
- L5: MTP block cross-entropy loss alignment
- L6: Full pipeline (decoder_norm → MTP → output_projection) forward + gradients
- L7: Adam optimizer single-step alignment
- Tolerance: `atol_fwd=1e-5`, `atol_grad=1e-4`

### Running Tests

```bash
# Unit tests (no external dependencies)
pytest tests/unit/multi_token_prediction_test.py -v

# Megatron alignment tests (requires megatron-core + torch)
pytest tests/unit/test_mtp_megatron_alignment.py -v

# Full suite
pytest tests/unit/multi_token_prediction_test.py \
       tests/unit/test_mtp_megatron_alignment.py -v
```
