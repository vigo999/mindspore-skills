# MindSpore Development Agent

You are an expert MindSpore developer. Use the skills below to help build operators and migrate models.

**IMPORTANT**: Read the appropriate SKILL.md file when the user's task matches a skill description.

## Available Skills

### Operator Development

| Skill | Path | Description |
|-------|------|-------------|
| cpu-plugin-builder | skills/cpu-plugin-builder/ | Build CPU operators via ATen/libtorch in mindspore_op_plugin |
| cpu-native-builder | skills/cpu-native-builder/ | Build native CPU kernels with Eigen/SLEEF |
| gpu-builder | skills/gpu-builder/ | Build GPU operators with CUDA |
| npu-builder | skills/npu-builder/ | Build NPU operators for Huawei Ascend |

### Model Migration

| Skill | Path | Description |
|-------|------|-------------|
| hf-diffusers-migrate | skills/hf-diffusers-migrate/ | Migrate HF diffusers models to mindone.diffusers |
| hf-transformers-migrate | skills/hf-transformers-migrate/ | Migrate HF transformers models to mindone.transformers |
| model-migrate | skills/model-migrate/ | Migrate PyTorch repos to MindSpore |

## Activation Triggers

Load the appropriate SKILL.md when users mention:

**Operator Development:**
- **cpu-plugin-builder**: "ATen", "libtorch", "op_plugin", "mindspore_op_plugin", "mint.*"
- **cpu-native-builder**: "native kernel", "Eigen", "SLEEF", "CPUKernelMod"
- **gpu-builder**: "CUDA", "GPU kernel", "cuDNN", "GPU operator"
- **npu-builder**: "Ascend", "NPU", "CANN", "ACLNN", "TBE"

**Model Migration:**
- **hf-diffusers-migrate**: "diffusers", "Stable Diffusion", "SDXL", "ControlNet", "mindone.diffusers"
- **hf-transformers-migrate**: "transformers", "BERT", "GPT", "LLaMA", "mindone.transformers"
- **model-migrate**: "migrate", "PyTorch to MindSpore", "convert model", "port model"

## Quick Decision Guide

### Operator Development

1. **What platform?**
   - CPU → Go to step 2
   - GPU → Use `gpu-builder`
   - NPU (Ascend) → Use `npu-builder`

2. **Which CPU approach?**
   - Working in `mindspore_op_plugin/` → Use `cpu-plugin-builder`
   - Working in `mindspore/` → Use `cpu-native-builder`

### Model Migration

1. **What are you migrating?**
   - HF diffusers model → Use `hf-diffusers-migrate`
   - HF transformers model → Use `hf-transformers-migrate`
   - Other PyTorch repo → Use `model-migrate`

## Usage

When a user's request matches a skill:

1. Read the corresponding `skills/<name>/SKILL.md` file
2. Follow the step-by-step instructions
3. Use reference materials in `skills/<name>/reference/` if available

## Compatibility

This repository works with:

- **Claude Code**: `/plugin marketplace add vigo999/mindspore-skills`
- **OpenCode**: Clone to `~/.config/opencode/` or `.opencode/`
- **Gemini CLI**: `gemini extensions install <repo> --consent`
- **Codex**: Reads this AGENTS.md automatically
