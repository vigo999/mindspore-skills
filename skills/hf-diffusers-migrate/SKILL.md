---
name: hf-diffusers-migrate
description: Migrate Hugging Face diffusers models to mindone.diffusers. Use when porting Stable Diffusion, SDXL, ControlNet, or other diffusion models.
---

# HF Diffusers Migration

Migrate Hugging Face diffusers models to MindOne's diffusers implementation.

## When to Use

- Porting Stable Diffusion models to MindSpore
- Migrating SDXL, ControlNet, LoRA adapters
- Converting diffusers pipelines to mindone
- Adding new diffusion model architectures

## Target Repository

**mindone.diffusers**: https://github.com/mindspore-lab/mindone

## Supported Model Types

- **Base Models**: SD 1.x, SD 2.x, SDXL, SD3
- **ControlNet**: Various conditioning models
- **Adapters**: LoRA, IP-Adapter, T2I-Adapter
- **Video**: AnimateDiff, SVD
- **Inpainting/Outpainting**: Specialized pipelines

## Instructions

(TODO: Add detailed migration workflow)

### Step 1: Analyze Source Model

1. Identify the HF diffusers model architecture
2. Check if similar architecture exists in mindone.diffusers
3. Document API differences between HF and mindone

### Step 2: Weight Conversion

1. Download HF model weights
2. Map weight names to MindOne format
3. Convert using mindone conversion tools

### Step 3: Pipeline Migration

1. Identify pipeline components (scheduler, VAE, UNet, text encoder)
2. Map to corresponding mindone components
3. Adjust API calls for MindSpore compatibility

### Step 4: Validation

1. Run inference with same inputs on both frameworks
2. Compare outputs numerically
3. Benchmark performance

## References

- [mindone.diffusers documentation](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers)
- [HF diffusers documentation](https://huggingface.co/docs/diffusers)
