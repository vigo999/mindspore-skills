---
name: cpu-plugin-builder
description: Build MindSpore CPU operators by adapting ATen (libtorch) operators via mindspore_op_plugin. Use when implementing ops in op_plugin/ops/kernel/, writing kernel .cc files, or creating operator tests with mint.*/Tensor.* interfaces.
---

# MindSpore CPU Plugin Builder (ATen Adaptation)

This skill helps you develop CPU operators for MindSpore's op_plugin that call ATen (libtorch) operators.

## When to Use

Use this skill when:
- Implementing new CPU operators for mindspore_op_plugin
- Writing kernel `.cc` files under `op_plugin/ops/kernel/`
- Adding forward and backward (gradient) operators
- Creating functional and performance tests for operators
- Adapting ATen operators to MindSpore interfaces

## Quick Start

1. Identify the operator name from `mindspore/ops/op_def/yaml/`
2. Find the corresponding ATen operator in `third_party/libtorch/include/`
3. Implement the kernel in `op_plugin/ops/kernel/<op_name>.cc`
4. Add tests in `tests/st/mint/test_<op_name>.py`
5. Build with `bash build.sh` and test with `pytest`

## Instructions

### Step 1: Identify the Operator Name

Find the operator definition in MindSpore via cpu-find-name skill

### Step 2: Find the ATen Interface

Search for the corresponding ATen operator:

1. **Location**: `./third_party/libtorch/include/ATen/`
2. **Prefer `_out` variants**: Use `at::xxx_out()` when available to write directly to output tensor
3. **Fallback to copy**: If no `_out` variant exists, use `at::xxx()` then `copy_()` to output

Reference files:
- `aten/src/ATen/native/native_functions.yaml` for operator definitions
- `aten/src/ATen/templates/RedispatchFunctions.h` for `_out` variants

### Step 3: Implement the Kernel

Create the kernel file at `op_plugin/ops/kernel/<op_name>.cc`:

```cpp
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int OpName(int nparam, void **params, int *ndims, int64_t **shapes,
                      const char **dtypes, void *stream, void *extra) {
  // Convert MindSpore tensors to ATen tensors
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  // Parse non-tensor parameters if needed
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  // Get inputs and output (output is always last)
  auto input = tensors[0];
  auto output = tensors[nparam - 1];

  // Call ATen operator (prefer _out variant)
  at::op_name_out(output, input);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```

### Step 4: Handle Backward Operators (if needed)

Check if a gradient operator is required:

1. **Locate bprop definition**: Search `REG_BPROP_BUILDER("<PrimName>")` in `mindspore/ccsrc/frontend/expander/grad/`
2. **Check for dedicated Grad primitive**: Look for `Emit("XXXGrad", ...)` calls
3. **Composed gradients**: If bprop uses basic ops (Mul, Add, Cos), ensure those ops are available

If `XXXGrad` is needed, implement it similarly in `op_plugin/ops/kernel/<op_name>_grad.cc`.

### Step 5: Build the Plugin

```bash
cd mindspore_op_plugin
bash build.sh
source env.source
```

Verify registration in build logs: `Found operator: OpName`

### Step 6: Write Functional Tests

Create `tests/st/mint/test_<op_name>.py`:

```python
#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# ... license header ...
"""<op_name> op test case"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(input_data):
    """Generate expected output using PyTorch."""
    return torch.op_name(input_data)


def forward_func(input_data):
    """Forward function for mint.op_name."""
    return mint.op_name(input_data)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_op_name_std(mode):
    """
    Feature: pyboost function.
    Description: test function op_name.
    Expectation: expect correct result.
    """
    # Setup test data
    np.random.seed(0)
    input_data = np.random.randn(3, 4).astype(np.float32)

    # Get expected output from PyTorch
    torch_input = torch.from_numpy(input_data)
    expect = generate_expect_forward_output(torch_input)

    # Run MindSpore
    ms_input = ms.Tensor(input_data)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_func(ms_input)
    elif mode == "KBK":
        output = jit(forward_func, backend="ms_backend", jit_level="O0")(ms_input)

    # Compare results
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Step 7: Write Performance Tests

Create `tests/st/mint/test_perf_<op_name>.py`:

```python
#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# ... license header ...
"""<op_name> op performance test case"""
import time
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import pytest


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_op_name_perf(mode):
    """
    Feature: standard forward performance.
    Description: test op_name op performance.
    Expectation: expect performance OK (<=1.1x torch).
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Setup large test data
    input_data = np.random.randn(1000, 1000).astype(np.float32)
    ms_input = ms.Tensor(input_data)
    torch_input = torch.from_numpy(input_data)

    # Warm-up
    for _ in range(1000):
        _ = mint.op_name(ms_input)

    # MindSpore timing
    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        _ = mint.op_name(ms_input)
    _pynative_executor.sync()
    ms_time = time.time() - start

    # PyTorch timing
    for _ in range(1000):
        _ = torch.op_name(torch_input)
    start = time.time()
    for _ in range(1000):
        _ = torch.op_name(torch_input)
    torch_time = time.time() - start

    # Performance gate: MindSpore should be within 1.1x of PyTorch
    assert np.less(ms_time - BACKGROUND_NOISE, torch_time * 1.1).all()
```

### Step 8: Run Tests

```bash
source env.source
pytest tests/st/mint/test_<op_name>.py
pytest tests/st/mint/test_perf_<op_name>.py
```

## Test Coverage Checklist

Ensure your tests cover:
- [ ] Default arguments
- [ ] Empty tensors
- [ ] NaN and Inf values
- [ ] All supported dtypes (float16, float32, float64, etc.)
- [ ] Mixed/implicit dtype scenarios
- [ ] 0D through 8D tensors
- [ ] Non-contiguous tensors
- [ ] Dynamic shapes
- [ ] Boundary conditions and error messages
- [ ] Functional interface (mint.xxx)
- [ ] Tensor method interface (Tensor.xxx) if applicable
- [ ] vmap with batch sizes 8/16/32/64/128 where relevant
- [ ] Forward accuracy (zero deviation from torch)
- [ ] Backward accuracy (zero deviation from torch) if gradient exists
- [ ] Performance within 1.1x of PyTorch

## Test Level Marks

Use appropriate `level_mark` for different test types:

- **`level0`**: Core functional tests (standard, dtype coverage, dimensions, special values, empty tensors, non-contiguous, dynamic shapes)
- **`level1`**: vmap tests and performance tests only

Example:
```python
# Core test - level0
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', ...)
def test_op_std(mode):
    ...

# Dtype coverage - level0
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', ...)
def test_op_dtype_coverage(mode, dtype_str):
    ...

# vmap test - level1
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', ...)
def test_op_vmap(mode, batch_size):
    ...

# Performance test - level1
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', ...)
def test_op_perf(mode):
    ...
```

## vmap Testing Notes

When testing with vmap, the batch dimension is added at dim 0. This affects gradient computation:

```python
# For vmap, batch is at dim 0, so adjust dim by +1 when computing gradient
expect_grad = generate_expect_backward_output(
    torch_x,
    torch.tensor(grad, dtype=torch.float32),
    dim=dim + 1,  # Adjust for batch dimension
)
```

## Key Utilities

### ConvertToATenTensors
Converts MindSpore tensors to ATen tensors:
```cpp
auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
```

### KernelInputUtils
Parses non-tensor parameters:
```cpp
KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
KernelInputUtils input_utils(input_info);
int64_t int_param = input_utils.GetIntInput(idx);
at::Scalar scalar_param = input_utils.GetScalarInput(idx);
```

### allclose_nparray
Compares arrays with tolerance:
```python
allclose_nparray(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True)
```

## Examples

Reference implementations in `op_plugin/ops/kernel/`:
- `add.cc` - Basic arithmetic operator
- `linspace_ext.cc` - Operator with scalar parameters
- `max_dim.cc` - Operator with multiple outputs
- `inplace_sub_ext.cc` - Inplace operator

Reference tests in `tests/st/mint/`:
- `test_linspace.py` - Functional test example
- `test_perf_linspace.py` - Performance test example

## Troubleshooting

**Build not registered**: Check CMake logs for `Found operator: <Op>`, verify file is in correct directory

**Numerical mismatch**: Prefer `_out` variants, print intermediate values for debugging

**Performance issues**: Check for extra allocations, ensure warm-up runs, verify BACKGROUND_NOISE subtraction

## References

For detailed documentation, see:
- [plugin_process.md](reference/plugin_process.md) - Complete development workflow
- [isses_linspace_op.md](reference/isses_linspace_op.md) - Example issue for linspace operator
- [issuse_sub_inplace_op.md](reference/issuse_sub_inplace_op.md) - Example issue for inplace sub operator
