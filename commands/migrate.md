---
description: Route to MindSpore migration tools (HF models, third-party repos)
---

# MindSpore Migration Tools

Select a migration type:

| Type | Command | Description |
|------|---------|-------------|
| **HF Models** | `/hf-migrate` | Migrate Hugging Face models to MindOne |
| **Third-party** | `/model-migrate` | Migrate PyTorch repos to MindSpore |

## Usage

```
/migrate hf       → routes to /hf-migrate
/migrate model    → routes to /model-migrate
```

If no type specified, ask user what they want to migrate.
