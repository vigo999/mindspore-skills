---
description: Route to Hugging Face model migration tools
---

# Hugging Face Migration

Select a library to migrate from:

| Library | Command | Target |
|---------|---------|--------|
| **Diffusers** | `/hf-diffusers-migrate` | mindone.diffusers |
| **Transformers** | `/hf-transformers-migrate` | mindone.transformers |

## Usage

```
/hf-migrate diffusers      → routes to /hf-diffusers-migrate
/hf-migrate transformers   → routes to /hf-transformers-migrate
```

## Target Repository

**mindone**: https://github.com/mindspore-lab/mindone

- `mindone.diffusers` - MindSpore implementation of HF diffusers
- `mindone.transformers` - MindSpore implementation of HF transformers

If no library specified, ask user which HF library they're migrating from.
