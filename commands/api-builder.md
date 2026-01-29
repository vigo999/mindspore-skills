---
description: Route to MindSpore operator builders for CPU/GPU/NPU platforms
---

# MindSpore API Builder

Select a platform to build operators:

| Platform | Command | Use When |
|----------|---------|----------|
| **CPU** | `/cpu-builder` | x86/ARM servers, local development |
| **GPU** | `/gpu-builder` | NVIDIA GPUs, CUDA kernels |
| **NPU** | `/npu-builder` | Huawei Ascend, CANN operators |

## Usage

```
/api-builder cpu    → routes to /cpu-builder
/api-builder gpu    → routes to /gpu-builder
/api-builder npu    → routes to /npu-builder
```

If no platform specified, ask user which platform they're targeting.
