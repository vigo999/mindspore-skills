---
description: Route to MindSpore migration tools (HF models, third-party repos)
---

# Migrate

Use this as the top-level migration entrypoint.

Route internally to `migrate-agent` after classifying the source:

- Hugging Face Transformers model or repo
- Hugging Face Diffusers pipeline or repo
- generic PyTorch repository

## Usage

```text
/migrate migrate this Hugging Face Transformers Qwen3 model to MindSpore
/migrate port this PyTorch repo to MindSpore
```

If the user does not specify what they want to migrate, ask for the source repo
or model family before routing.
