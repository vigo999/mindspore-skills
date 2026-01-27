---
description: Route to CPU operator builders (plugin or native)
---

# MindSpore CPU Builder

Two approaches for CPU operators:

| Approach | Command | Use When |
|----------|---------|----------|
| **Plugin** | `/ms-cpu-plugin-builder` | Adapting ATen/libtorch ops via mindspore_op_plugin |
| **Native** | `/ms-cpu-native-builder` | Writing native kernels in mindspore/ (Eigen/SLEEF) |

## Quick Decision Guide

| Factor | Plugin (ATen) | Native |
|--------|---------------|--------|
| **Repo** | mindspore_op_plugin/ | mindspore/ |
| **Speed** | Faster (reuse PyTorch) | Slower (write from scratch) |
| **Dependency** | Requires libtorch | No external deps |
| **Control** | Limited to ATen API | Full control |
| **Best for** | Rapid prototyping | Production optimization |

## Usage

```
/ms-cpu-builder plugin    → routes to /ms-cpu-plugin-builder
/ms-cpu-builder native    → routes to /ms-cpu-native-builder
```

If unclear, ask user:
1. Are you working in `mindspore_op_plugin/` or `mindspore/` repo?
2. Do you want to adapt an existing ATen op or write from scratch?
