# Native Torch / Custom Implementation Guide

Integration for standalone PyTorch models not using HuggingFace or llama-factory.

## When to Use This Guide

- nanoGPT, minGPT, or custom GPT implementations
- Research code with direct PyTorch
- Models with non-standard residual patterns
- Any decoder-only architecture without framework wrappers

## Core Principle

**Expand once at entry, wrap each branch, reduce once at exit.**

```
embed → expand → [layer1: attn_wrap + mlp_wrap] → ... → [layerN] → reduce → norm → head
```

## Minimal Working Example

```python
import torch
import torch.nn as nn
from functools import partial
from portable_static_mhc import (
    StaticMHCAdapter,
    expand_streams_repeat,
    reduce_streams_sum,
)

class SimpleMHCDecoder(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_streams=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(512, dim)
        self.num_streams = num_streams
        self.dim = dim

        # Create mHC adapters for each layer's attention and MLP
        self.attn_adapters = nn.ModuleList([
            StaticMHCAdapter(
                num_streams=num_streams,
                dim=dim,
                layer_index=i,
                init_mode="from_scratch",
            )
            for i in range(num_layers)
        ])
        self.mlp_adapters = nn.ModuleList([
            StaticMHCAdapter(
                num_streams=num_streams,
                dim=dim,
                layer_index=i,
                init_mode="from_scratch",
            )
            for i in range(num_layers)
        ])

        # Actual layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        # Stream expansion/reduction
        if num_streams > 1:
            self.expand = partial(expand_streams_repeat, num_streams=num_streams)
            self.reduce = partial(reduce_streams_sum, num_streams=num_streams)
        else:
            self.expand = lambda x: x
            self.reduce = lambda x: x

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        # Embed
        x = self.embed(x) + self.pos_embed(pos)

        # Expand streams
        x = self.expand(x)

        # Through layers with mHC-wrapped attention/MLP
        for i, layer in enumerate(self.layers):
            residual = x

            # Wrapped self-attention
            attn_out, add_attn = self.attn_adapters[i](x)
            x = layer.self_attn(attn_out)
            x = add_attn(x)
            x = residual + x

            # Wrapped MLP
            residual = x
            mlp_out, add_mlp = self.mlp_adapters[i](x)
            x = layer.linear1(mlp_out)
            x = layer.dropout(layer.activation(layer.linear2(x)))
            x = add_mlp(x)
            x = residual + x

        # Reduce streams
        x = self.reduce(x)

        x = self.norm(x)
        return self.head(x)
```

## Stream Expansion Modes

### expand_streams_repeat (from_scratch)

```python
# Each stream gets a copy of the hidden state
# Good for: new training
hidden = embed(tokens)  # [B, T, D]
hidden = expand_streams_repeat(hidden, num_streams=4)  # [B*4, T, D]
```

### expand_streams_identity_safe (retrofit)

```python
# Only stream 0 gets the hidden state; others get zeros
# Good for: loading existing checkpoints
hidden = embed(tokens)
hidden = expand_streams_identity_safe(hidden, num_streams=4)  # [B*4, T, D]
# Stream 0 has signal, streams 1-3 are zero
```

### reduce_streams_sum

```python
# Sum across all streams
hidden = reduce_streams_sum(hidden, num_streams=4)  # [B, T, D]
```

## Integration Checklist

- [ ] Expand streams immediately after embeddings
- [ ] Reduce streams before final norm
- [ ] Wrap both attention and MLP branches per layer
- [ ] `num_streams=1` returns standard residual behavior
- [ ] Hidden dim unchanged at entry/exit
- [ ] Logits shape matches baseline

## Testing with num_streams=1

Always verify your integration works with `num_streams=1`:

```python
# This should produce identical output to a standard model
model_1stream = MyModel(num_streams=1)
model_standard = StandardModel()  # no mHC

# Should be close to identical
assert torch.allclose(model_1stream(x), model_standard(x), atol=1e-6)
```

## Custom Branch Patterns

If your model has different branch patterns (e.g., attention + MLP + something else):

```python
# Generic wrapper pattern
class MHCBranchWrapper(nn.Module):
    def __init__(self, num_streams, dim, branch_fn, layer_index):
        super().__init__()
        self.adapter = StaticMHCAdapter(
            num_streams=num_streams,
            dim=dim,
            layer_index=layer_index,
        )
        self.branch_fn = branch_fn

    def forward(self, residual, *args, **kwargs):
        # Width connection
        branch_input, residual_out, beta = self.adapter.width_connection(residual)

        # Branch computation
        branch_out = self.branch_fn(branch_input, *args, **kwargs)

        # Depth connection
        return self.adapter.depth_connection(branch_out, residual_out, beta=beta)
```

## Common Patterns

### Pattern: Pre-norm Residual

```
x = norm(x)
x = mhc_adapter(x, branch_fn)  # returns final x
x = residual + x
```

### Pattern: Post-norm Residual

```
y = branch(norm(x))
x = x + y
# mHC wraps the addition
```

### Pattern: Multiple Branches Per Layer

```
x = norm(x)
x = mhc_adapter_1(x, branch1_fn)
x = mhc_adapter_2(x, branch2_fn)
x = residual + x
```

## Debugging

### Print H matrices during forward:

```python
def debug_mhc(adapter, stage):
    h_res = adapter.project_h_res()
    h_pre = adapter.project_h_pre()
    h_post = adapter.project_h_post()
    print(f"{stage}: H_res sum rows={h_res.sum(-1).mean():.4f}, cols={h_res.sum(-2).mean():.4f}")
    print(f"  H_pre min={h_pre.min():.4f}, max={h_pre.max():.4f}")
    print(f"  H_post min={h_post.min():.4f}, max={h_post.max():.4f}")
```

### Check gradient flow:

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if "H_" in name:
            print(f"{name}: grad={param.grad is not None}")
```
