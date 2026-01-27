---
description: Route to MindSpore operator builders for CPU/GPU/NPU platforms
---

# MindSpore API Builder

Select a platform to build operators:

| Platform | Command | Use When |
|----------|---------|----------|
| **CPU** | `/ms-cpu-builder` | x86/ARM servers, local development |
| **GPU** | `/ms-gpu-builder` | NVIDIA GPUs, CUDA kernels |
| **NPU** | `/ms-npu-builder` | Huawei Ascend, CANN operators |

## Usage

```
/ms-api-builder cpu    → routes to /ms-cpu-builder
/ms-api-builder gpu    → routes to /ms-gpu-builder
/ms-api-builder npu    → routes to /ms-npu-builder
```

If no platform specified, ask user which platform they're targeting.
