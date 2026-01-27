---
description: Route to Hugging Face model migration tools
---

# Hugging Face Migration

Select a library to migrate from:

| Library | Command | Target |
|---------|---------|--------|
| **Diffusers** | `/ms-hf-diffusers-migrate` | mindone.diffusers |
| **Transformers** | `/ms-hf-transformers-migrate` | mindone.transformers |

## Usage

```
/ms-hf-migrate diffusers      → routes to /ms-hf-diffusers-migrate
/ms-hf-migrate transformers   → routes to /ms-hf-transformers-migrate
```

## Target Repository

**mindone**: https://github.com/mindspore-lab/mindone

- `mindone.diffusers` - MindSpore implementation of HF diffusers
- `mindone.transformers` - MindSpore implementation of HF transformers

If no library specified, ask user which HF library they're migrating from.
