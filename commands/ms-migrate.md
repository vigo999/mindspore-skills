---
description: Route to MindSpore migration tools (HF models, third-party repos)
---

# MindSpore Migration Tools

Select a migration type:

| Type | Command | Description |
|------|---------|-------------|
| **HF Models** | `/ms-hf-migrate` | Migrate Hugging Face models to MindOne |
| **Third-party** | `/ms-model-migrate` | Migrate PyTorch repos to MindSpore |

## Usage

```
/ms-migrate hf       → routes to /ms-hf-migrate
/ms-migrate model    → routes to /ms-model-migrate
```

If no type specified, ask user what they want to migrate.
