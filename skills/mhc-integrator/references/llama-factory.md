# llama-factory Integration Guide

Integration for LLamaFactory training framework.

## llama-factory Structure

llama-factory typically has:
- `src/llamafactory/` - main package
- `model/` - model loading and wrapper
- `hparams/` - hyperparameter definitions
- `executor/` - training execution

## Integration Points

### Option 1: Model Wrapper (Recommended)

Create a wrapper that applies mHC at the model level:

```python
# src/llamafactory/model/mhc_llama_model.py

from functools import partial
from llamafactory.model.modeling_llama import LlamaForCausalLM
from llamafactory.model.utils import load_model
from portable_static_mhc import (
    StaticMHCAdapter,
    expand_streams_repeat,
    expand_streams_identity_safe,
    reduce_streams_sum,
)

class MHCLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, mhc_config):
        super().__init__(config)
        self._setup_mhc(mhc_config)

    def _setup_mhc(self, mhc_config):
        if not mhc_config.mhc_enabled or mhc_config.mhc_num_streams <= 1:
            self.expand_streams = lambda x: x
            self.reduce_streams = lambda x: x
            self.mhc_adapters = None
            return

        num_streams = mhc_config.mhc_num_streams
        init_mode = mhc_config.get("init_mode", "checkpoint_retrofit")

        expand_fn = expand_streams_identity_safe if init_mode == "checkpoint_retrofit" else expand_streams_repeat

        self.expand_streams = partial(expand_fn, num_streams=num_streams)
        self.reduce_streams = partial(reduce_streams_sum, num_streams=num_streams)

        # Wrap each layer's attention and MLP
        self.mhc_adapters = nn.ModuleList()
        for layer in self.model.layers:
            attn_adapter = StaticMHCAdapter(
                num_streams=num_streams,
                dim=config.hidden_size,
                layer_index=layer.layer_idx,
                init_mode=init_mode,
                mhc_projection=mhc_config.mhc_projection,
                mhc_sinkhorn_iters=mhc_config.mhc_sinkhorn_iters,
                mhc_sinkhorn_tau=mhc_config.mhc_sinkhorn_tau,
            )
            mlp_adapter = StaticMHCAdapter(
                num_streams=num_streams,
                dim=config.hidden_size,
                layer_index=layer.layer_idx,
                init_mode=init_mode,
                mhc_projection=mhc_config.mhc_projection,
                mhc_sinkhorn_iters=mhc_config.mhc_sinkhorn_iters,
                mhc_sinkhorn_tau=mhc_config.mhc_sinkhorn_tau,
            )
            layer.mhc_attn_adapter = attn_adapter
            layer.mhc_mlp_adapter = mlp_adapter
            self.mhc_adapters.append((attn_adapter, mlp_adapter))

    def forward(self, input_ids, ...):
        # Expand at entry
        hidden_states = self.model.embed_tokens(input_ids)
        if self.mhc_adapters is not None:
            hidden_states = self.expand_streams(hidden_states)

        # Through layers
        for i, layer in enumerate(self.model.layers):
            # Apply mHC-wrapped forward
            hidden_states = self._mhc_layer_forward(layer, hidden_states, i, ...)

        # Reduce at exit
        if self.mhc_adapters is not None:
            hidden_states = self.reduce_streams(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        return self.lm_head(hidden_states)
```

### Option 2: HParam Extension

Extend the training arguments to include mHC config:

```python
# src/llamafactory/hparams/data_args.py or training_args.py

@dataclass
class MHCArguments:
    """mHC configuration"""
    mhc_enabled: bool = field(default=False)
    mhc_num_streams: int = field(default=4)
    mhc_projection: str = field(default="sinkhorn")
    mhc_sinkhorn_iters: int = field(default=10)
    mhc_sinkhorn_tau: float = field(default=0.05)
    mhc_residual_identity_mix: bool = field(default=False)
    mhc_residual_alpha: float = field(default=0.01)
```

### Option 3: Model Loader Patch

Patch the model loading to apply mHC:

```python
# src/llamafactory/model/loader.py

def load_model_with_mhc(model_args, ...):
    model = load_model(model_args, ...)  # Original loader

    if model_args.mhc_enabled:
        from portable_static_mhc import (
            StaticMHCAdapter,
            expand_streams_repeat,
            reduce_streams_sum,
        )

        num_streams = model_args.mhc_num_streams
        model.expand_streams = partial(
            expand_streams_repeat if model_args.train_from_scratch
            else expand_streams_identity_safe,
            num_streams=num_streams
        )
        model.reduce_streams = partial(reduce_streams_sum, num_streams=num_streams)

        # Wrap each layer
        for i, layer in enumerate(model.model.layers):
            layer.mhc_attn_adapter = StaticMHCAdapter(...)
            layer.mhc_mlp_adapter = StaticMHCAdapter(...)

    return model
```

## LoRA Compatibility

If using LoRA with llama-factory:

1. mHC should be applied BEFORE LoRA injection
2. LoRA typically operates on QKV and gate_proj
3. The mHC adapter wraps around the attention module, so LoRA sees the pre-mHC hidden states
4. This ordering is usually correct since LoRA learns residual updates

```python
# Recommended ordering:
# hidden_states → mHC_pre_mix → attention_with_lora → mHC_post_mix → residual
```

## Checkpoint Saving

When saving checkpoints with mHC:

```python
def save_mhc_checkpoint(model, path):
    state_dict = model.state_dict()
    # Filter mHC-only keys for partial save
    mhc_keys = {k: v for k, v in state_dict.items() if "mhc" in k or "H_" in k}
    torch.save(mhc_keys, os.path.join(path, "mhc_weights.pt"))
    # Full save for complete checkpoint
    torch.save(state_dict, os.path.join(path, "model.safetensors"))
```

## Training Considerations

1. **Learning rate**: May need lower LR for mHC parameters initially
2. **Warmup**: Standard warmup helps stabilize stream routing
3. **Mixed precision**: mHC operations work fine with BF16/FP16
4. **Gradient checkpointing**: Compatible with mHC

## Verification

1. Test with `mhc_enabled=False` for baseline
2. Verify trainable params include H_res_logits, H_pre_logits, H_post_logits
3. Check that gradients flow to all adapter parameters
4. Monitor H_res convergence (row/col sums should stay near 1)
