---
name: npu-builder
description: Build MindSpore NPU operators for Huawei Ascend chips using CANN. Use when implementing operators in mindspore/ccsrc/plugin/device/ascend/kernel/.
---

# MindSpore NPU Builder

Build NPU operators for Huawei Ascend chips using the CANN (Compute Architecture for Neural Networks) framework.

## When to Use

Use this skill when:
- Implementing operators for Huawei Ascend 910/310 chips
- Writing kernels in `mindspore/ccsrc/plugin/device/ascend/kernel/`
- Using CANN operators or TBE (Tensor Boost Engine)
- Optimizing for NPU architecture

## Quick Start

1. Identify the operator from `mindspore/ops/op_def/yaml/`
2. Create kernel in `mindspore/ccsrc/plugin/device/ascend/kernel/`
3. Register with CANN operator library
4. Add tests in `tests/st/ops/ascend/`

## Instructions

### Step 1: Create Ascend Kernel

Create `mindspore/ccsrc/plugin/device/ascend/kernel/<op_name>_ascend_kernel.cc`:

```cpp
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * ...
 */

#include "plugin/device/ascend/kernel/<op_name>_ascend_kernel.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {

bool OpNameAscendKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  // Initialize kernel parameters
  return true;
}

bool OpNameAscendKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs,
                                    void *stream_ptr) {
  auto stream = reinterpret_cast<aclrtStream>(stream_ptr);

  // Get device addresses
  auto input = inputs[0]->device_ptr();
  auto output = outputs[0]->device_ptr();

  // Call CANN operator
  // ...

  return true;
}

MS_KERNEL_FACTORY_REG(NativeAscendKernelMod, OpName, OpNameAscendKernelMod);

}  // namespace kernel
}  // namespace mindspore
```

### Step 2: Using ACLNN Operators

For operators available in ACLNN:

```cpp
#include "aclnn/<op_name>.h"

bool OpNameAscendKernelMod::Launch(...) {
  auto stream = reinterpret_cast<aclrtStream>(stream_ptr);

  // Create ACLNN tensors
  aclTensor *aclInput = CreateAclTensor(inputs[0]);
  aclTensor *aclOutput = CreateAclTensor(outputs[0]);

  // Get workspace size
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  auto ret = aclnnOpNameGetWorkspaceSize(aclInput, aclOutput,
                                          &workspaceSize, &executor);

  // Allocate workspace
  void *workspace = nullptr;
  if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Execute
  ret = aclnnOpName(workspace, workspaceSize, executor, stream);

  // Cleanup
  aclDestroyTensor(aclInput);
  aclDestroyTensor(aclOutput);

  return ret == ACL_SUCCESS;
}
```

### Step 3: Using TBE (Tensor Boost Engine)

For custom TBE operators:

```python
# Define TBE operator in Python DSL
# mindspore/python/mindspore/ops/_op_impl/tbe/<op_name>.py

from te import tik

@register_op_compute("OpName")
def op_name_compute(input_x, output_y, kernel_name="op_name"):
    tik_instance = tik.Tik()

    # Define tensors
    input_gm = tik_instance.Tensor(dtype, shape, name="input_gm", scope=tik.scope_gm)
    output_gm = tik_instance.Tensor(dtype, shape, name="output_gm", scope=tik.scope_gm)

    # Implement computation
    # ...

    tik_instance.BuildCCE(kernel_name, inputs=[input_gm], outputs=[output_gm])
    return tik_instance
```

### Step 4: Write Tests

Create `tests/st/ops/ascend/test_<op_name>_op.py`:

```python
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
import torch


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_op_name_ascend():
    """Test op_name on Ascend NPU."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    input_np = np.random.randn(3, 4).astype(np.float32)

    ms_output = ops.op_name(Tensor(input_np))

    # Reference computation (CPU or PyTorch)
    expected = np.op_name(input_np)  # or torch reference

    assert np.allclose(ms_output.asnumpy(), expected, rtol=1e-3, atol=1e-3)
```

## Key Locations

| Component | Location |
|-----------|----------|
| Ascend kernels | `mindspore/ccsrc/plugin/device/ascend/kernel/` |
| ACLNN wrappers | `mindspore/ccsrc/plugin/device/ascend/kernel/opapi/` |
| TBE definitions | `mindspore/python/mindspore/ops/_op_impl/tbe/` |
| Tests | `tests/st/ops/ascend/` |

## CANN Operator Categories

| Category | Description | When to Use |
|----------|-------------|-------------|
| **ACLNN** | Pre-built CANN operators | Most common ops |
| **TBE** | Custom DSL operators | When ACLNN lacks support |
| **AICPU** | AI CPU operators | Ops not suitable for NPU |

## Performance Tips

1. **Use ACLNN** when available - highly optimized
2. **Batch operations** - NPU excels at batched computation
3. **Minimize data transfer** - Keep data on device
4. **Use appropriate precision** - FP16 is often faster
5. **Profile with CANN tools** - Identify bottlenecks

## TODO

This skill is a placeholder. Detailed instructions will be added based on actual NPU kernel development patterns in MindSpore and CANN SDK documentation.
