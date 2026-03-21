# HuggingFace Integration Guide

Integration for Llama, Qwen, Mistral, Gemma, and other `transformers` decoder models.

## Target Files

Typically `modeling_llama.py`, `modeling_qwen2.py`, `modeling_mistral.py`, etc.

## Residual Pattern

Most HF decoders use pre-norm:
```
hidden_states → RMSNorm → attention → residual → RMSNorm → mlp → residual → next layer
```

Each decoder layer has two residual merges.

## Integration Steps

### 1. Add Config Fields

In `config.py` or `configuration_*.py`:
```python
mhc_enabled: bool = False
mhc_num_streams: int = 4
mhc_projection: str = "sinkhorn"  # or "orthostochastic"
mhc_sinkhorn_iters: int = 10
mhc_sinkhorn_tau: float = 0.05
mhc_residual_identity_mix: bool = False
mhc_residual_alpha: float = 0.01
```

### 2. Modify Model Class

In the main model class (`LlamaModel`, `Qwen2Model`, etc.):

```python
from .portable_static_mhc import (
    StaticMHCAdapter,
    expand_streams_repeat,
    reduce_streams_sum,
)

def __init__(self, config):
    super().__init__(config)
    # After embedding
    if config.mhc_enabled and config.mhc_num_streams > 1:
        self.expand_streams = partial(
            expand_streams_repeat,
            num_streams=config.mhc_num_streams
        )
        self.reduce_streams = partial(
            reduce_streams_sum,
            num_streams=config.mhc_num_streams
        )
        self.mhc_adapters = nn.ModuleList([
            StaticMHCAdapter(
                num_streams=config.mhc_num_streams,
                dim=config.hidden_size,
                layer_index=i,
                init_mode="from_scratch" if training else "checkpoint_retrofit",
                mhc_projection=config.mhc_projection,
                mhc_sinkhorn_iters=config.mhc_sinkhorn_iters,
                mhc_sinkhorn_tau=config.mhc_sinkhorn_tau,
                mhc_residual_identity_mix=config.mhc_residual_identity_mix,
                mhc_residual_alpha=config.mhc_residual_alpha,
            )
            for i in range(config.num_hidden_layers)
        ])
    else:
        self.expand_streams = lambda x: x
        self.reduce_streams = lambda x: x
        self.mhc_adapters = None
```

### 3. Wrap DecoderLayer

In each `DecoderLayer.forward()`:

```python
def forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    residual = hidden_states

    # Self-attention branch with mHC
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.expand_streams(hidden_states)

    # For attention adapter, wrap the self_attn call
    if self.mhc_adapter is not None:
        hidden_states, add_attn_residual = self.mhc_adapter(hidden_states)

    # Self-attention
    self_attn_output = self.self_attn(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = self_attn_output[0] if isinstance(self_attn_output, tuple) else self_attn_output

    if add_attn_residual is not None:
        hidden_states = add_attn_residual(hidden_states)
    hidden_states = residual + hidden_states

    # MLP branch with mHC
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.expand_streams(hidden_states)

    if self.mhc_mlp_adapter is not None:
        hidden_states, add_mlp_residual = self.mhc_mlp_adapter(hidden_states)

    mlp_output = self.mlp(hidden_states)

    if add_mlp_residual is not None:
        mlp_output = add_mlp_residual(mlp_output)
    hidden_states = residual + mlp_output

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_output[1],)
    if use_cache:
        outputs += (self_attn_output[2] if isinstance(self_attn_output, tuple) else None,)
    return outputs
```

### 4. Handle Model Entry/Exit

In the top-level model (`LlamaModel.forward()`):

```python
def forward(self, input_ids, ...):
    hidden_states = self.embed_tokens(input_ids)

    # Expand at entry
    if self.mhc_enabled:
        hidden_states = self.expand_streams(hidden_states)

    # Decoder layers (each wrapped)
    for layer in self.layers:
        hidden_states = layer(hidden_states, ...)

    # Reduce at exit
    if self.mhc_enabled:
        hidden_states = self.reduce_streams(hidden_states)

    hidden_states = self.norm(hidden_states)
    ...
```

### 5. Preserve Tuple Returns

HF attention returns `(hidden_states, attn_weights, past_key_value)`. Only modify the first element:

```python
def _mhc_forward(self, hidden_states, branch_fn, *branch_args, **branch_kwargs):
    """Returns (modified_hidden, residual_adder_fn)"""
    branch_output = branch_fn(hidden_states, *branch_args, **branch_kwargs)

    if isinstance(branch_output, tuple):
        # Only modify the hidden states, preserve extras
        hidden = branch_output[0]
        extras = branch_output[1:]
        return hidden, lambda h: (h,) + extras
    return branch_output, lambda h: h
```

## Cache Handling

Ensure `past_key_value` cache is not affected by mHC. The cache is managed inside `self_attn` and should not receive the expanded hidden states.

## Verification

1. Run with `mhc_enabled=False` → should match original behavior
2. Run with `mhc_num_streams=1` → should match original behavior
3. Check H_res row/col sums ≈ 1 after first forward pass
4. Verify `use_cache=True` produces same cache shape as baseline
