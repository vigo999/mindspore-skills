---
description: Build CPU operators via ATen adaptation in mindspore_op_plugin
---

# MindSpore CPU Plugin Builder

Build CPU operators by adapting ATen (libtorch) implementations.

Load the `cpu-plugin-builder` skill and follow its workflow.

## Target Repository

`mindspore_op_plugin/`

## Key Locations

- Kernels: `op_plugin/ops/kernel/<op_name>.cc`
- Tests: `tests/st/mint/test_<op_name>.py`
- ATen headers: `third_party/libtorch/include/ATen/`
