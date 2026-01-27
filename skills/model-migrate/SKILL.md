---
name: model-migrate
description: Migrate third-party PyTorch model repositories to MindSpore. Use when converting standalone PyTorch projects, research code, or custom architectures.
---

# Model Migration (PyTorch to MindSpore)

Migrate third-party PyTorch model repositories to MindSpore code.

## When to Use

- Converting PyTorch research repos to MindSpore
- Migrating custom model architectures
- Porting standalone PyTorch projects
- Adapting academic paper implementations

## Instructions

(TODO: Add detailed migration workflow)

### Step 1: Analyze Source Repository

1. Clone and understand the PyTorch repository structure
2. Identify model architecture and dependencies
3. Document PyTorch-specific APIs used

### Step 2: API Mapping

Common PyTorch to MindSpore mappings:

| PyTorch | MindSpore |
|---------|-----------|
| `torch.nn.Module` | `mindspore.nn.Cell` |
| `forward()` | `construct()` |
| `torch.tensor()` | `mindspore.Tensor()` |
| `torch.nn.Linear` | `mindspore.nn.Dense` |
| `torch.nn.Conv2d` | `mindspore.nn.Conv2d` |
| `torch.optim.Adam` | `mindspore.nn.Adam` |
| `model.train()` | `model.set_train(True)` |
| `model.eval()` | `model.set_train(False)` |
| `torch.no_grad()` | `mindspore.ops.stop_gradient()` |

### Step 3: Weight Conversion

1. Save PyTorch model weights (`state_dict`)
2. Create weight mapping dictionary
3. Load weights into MindSpore model

```python
import torch
import mindspore as ms

# Load PyTorch weights
pt_ckpt = torch.load('model.pth')

# Convert to MindSpore
ms_params = []
for name, param in pt_ckpt.items():
    ms_name = convert_name(name)  # Map PyTorch names
    ms_params.append({'name': ms_name, 'data': ms.Tensor(param.numpy())})

ms.save_checkpoint(ms_params, 'model.ckpt')
```

### Step 4: Training Loop Migration

1. Replace DataLoader with MindSpore dataset
2. Convert optimizer and loss function
3. Adapt training loop to MindSpore patterns

### Step 5: Validation

1. Compare model outputs with same inputs
2. Verify gradient computation
3. Run full training comparison

## Common Pitfalls

- **Inplace operations**: MindSpore prefers functional style
- **Dynamic shapes**: May need explicit shape handling
- **Custom autograd**: Use `mindspore.ops.Custom` for custom backward
- **Device management**: MindSpore uses context-based device selection

## References

- [MindSpore Migration Guide](https://www.mindspore.cn/docs/en/master/migration_guide/index.html)
- [MindSpore API Mapping](https://www.mindspore.cn/docs/en/master/note/api_mapping.html)
