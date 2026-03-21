---
name: "mhc-integrator"
description: "Use when the user asks to integrate mHC (manifold-constrained hyper-connections) into any LLM architecture or training framework. Triggers for: 'add mHC to [model]', 'enable multi-stream residual in [framework]', 'implement residual-stream routing', 'add hyper-connections to training', 'mHC for llama-factory', 'mHC for transformers', 'mHC for [any model]'. This skill provides architecture-agnostic mHC integration with support for HuggingFace transformers, llama-factory, native torch implementations, and custom decoders. Do not use for basic FlashAttention optimization or MoE routing - those need separate skills."
---

# mHC Integrator

Universal mHC integration skill for adding manifold-constrained hyper-connections to any LLM architecture or training framework.

## Core Concept

mHC modifies residual connections with learnable stream routing:
```
x_{l+1} = H_l^{res} x_l + H_l^{post} · F(H_l^{pre} x_l, W_l)
```

Key constraints:
- `H_res`: doubly stochastic (row/col sums = 1, non-negative) via Sinkhorn-Knopp or Newton-Schulz
- `H_pre`, `H_post`: non-negative (softmax)

## Quick Start

1. Identify the model's residual pattern (pre-norm, post-norm, RMSNorm locations)
2. Choose integration path based on framework:
   - **HuggingFace models** → read `references/huggingface.md`
   - **llama-factory** → read `references/llama-factory.md`
   - **Native torch / custom** → read `references/native-torch.md`
3. Use `StaticMHCAdapter` from `assets/templates/portable_static_mhc.py`
4. Add config keys: `mhc_enabled`, `mhc_num_streams`, `mhc_projection`, `mhc_sinkhorn_iters`, `mhc_sinkhorn_tau`
5. Expand streams at model entry, wrap each residual branch, reduce after final norm

## Integration Patterns

### Pattern A: Expand → Wrap Branches → Reduce

```
input
  ↓
expand_streams (repeat or identity_safe)
  ↓
block 1: wrap attention branch + wrap MLP branch
block 2: ...
block N: ...
  ↓
reduce_streams (sum)
  ↓
output
```

### Pattern B: Minimal (num_streams=1)

When `num_streams=1`, mHC degenerates to standard residual. Use this for baseline comparison.

## Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `mhc_enabled` | `False` | Enable/disable mHC |
| `mhc_num_streams` | `4` | Number of residual streams |
| `mhc_projection` | `"sinkhorn"` | `"sinkhorn"` or `"orthostochastic"` |
| `mhc_sinkhorn_iters` | `10` | Sinkhorn iterations |
| `mhc_sinkhorn_tau` | `0.05` | Sinkhorn temperature |
| `mhc_residual_identity_mix` | `False` | Mix H_res with identity |
| `mhc_residual_alpha` | `0.01` | Identity mix coefficient |

## Initialization Strategies

### from_scratch (new training)
- `expand_streams_repeat` at entry
- Diagonal-dominant H_res init (0 on diag, -8 off diag)
- H_pre selects one primary stream per layer
- Zero-initialize H_post

### checkpoint_retrofit (existing weights)
- `expand_streams_identity_safe` at entry
- Stronger diagonal bias (-12 off diag)
- H_pre/H_post default to stream 0
- Enable `mhc_residual_identity_mix` for sensitive checkpoints

## Framework-Specific Guidance

### HuggingFace (Llama, Qwen, Mistral, Gemma, etc.)
- Read `references/huggingface.md`
- Wrap `self_attn` and `mlp` branches in each `DecoderLayer`
- Preserve `use_cache`, `past_key_value`, `output_attentions`
- Handle tuple returns from attention

### llama-factory
- Read `references/llama-factory.md`
- Typically modify the `LLMFactory` or model wrapper
- May need to handle `LoraConfig` integration
- Checkpoint loading requires `identity_safe` expansion

### Native torch / custom implementations
- Read `references/native-torch.md`
- Most flexible - adapt to any residual pattern
- Ensure expand happens once at entry, reduce once at exit
- Test with `num_streams=1` for regression

## Validation Checklist

- [ ] H_res row/col sums ≈ 1, all entries ≥ 0
- [ ] H_pre, H_post non-negative
- [ ] `num_streams=1` → standard residual behavior
- [ ] Logits shape unchanged from baseline
- [ ] Gradients flow to H_res_logits, H_pre_logits, H_post_logits
- [ ] Cache (use_cache=True) works correctly
- [ ] Training loss converges similarly to baseline initially

## Common Issues

**Gradient vanishes**: Check H_res initialization, try identity_mix with small alpha

**Loss explodes**: Reduce learning rate, check H_post initialization

**Memory increase**: num_streams=N multiplies residual memory by N

**Cache breaks**: Ensure past_key_value handling in attention wrapper

## Non-Examples (Use Other Skills)

- "Optimize attention kernel" → performance/vectorization skill
- "Add MoE routing" → MoE-specific skill
- "Implement mHC from scratch for research" → mHC-algorithm skill
