
### HOW TO WRITE BACKWARD OP

When you know the forward op name, use api-helper skill to get all backward ops needed for that forward op.

## Coding Rules
- Ensure there is one operator in one .cc file
- Parse the full `REG_BPROP_BUILDER` body; do not stop at the first `Emit("XXXGrad", ...)`.
- Before adding a new kernel, check `op_plugin/ops/kernel/` to avoid duplicate implementation.

#### Case 1: Standalone Grad Operator:
 - If the backward uses `Emit("XXXGrad", ...)`, it is dedicated grad operator.
 - write xxx_grad.cc in `op_plugin/ops/kernel/xxx_grad.cc`
 - e.g. for gelu_ext_grad.cc

```
#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int GeluGradExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t approximate_enum = input_utils.GetIntInput(2);
  c10::string_view approximate = (approximate_enum == 1) ? "tanh" : "none";

  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto grad = tensors[0];
  auto input = tensors[1];
  auto dinput = tensors[nparam - 1];

  at::gelu_backward_out(dinput, grad.contiguous(), input.contiguous(), approximate);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```

#### Case 2: Multiple Primitive Operators

 - For `AcosExt` backward which uses `ib->Neg(dout) * ib->Rsqrt(ib->Sub(..., ib->Square(x)))`:
 - backward ops include `Neg`, `Rsqrt`, `Sub`, `Square` (and expression math like mul), so write missing primitive files such as `neg.cc`, `rsqrt.cc`, `sub.cc`, `square.cc` under `op_plugin/ops/kernel/`
 - e.g. for neg.cc

```
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int Neg(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  // Parameter list: [input, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  // Call ATen interface: output = -input
  at::neg_out(at_output, at_input);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

```

#### Case 3: Mixed Composite Backward Chain

 - Some backward bodies mix helper primitives and emitted grad operators.
 - Example: `AdaptiveMaxPool1D` backward chain may include:
   - `ExpandDims`
   - `Emit("AdaptiveMaxPool2DGrad", ...)`
   - `Reshape`
   - `OutZeros(output_size)`
 - Classify operators before implementation:
   - kernel-required primitives: `ExpandDims`, `Reshape`, `Squeeze`, `Transpose`, `Cast`, and emitted grad ops (for example `AdaptiveMaxPool2DGrad`) if missing
   - graph/meta helpers (no kernel file): `OutZeros`, `ShapeCalc`, `TupleGetItem`, `EmitValue`
 - Implement only missing kernel-required operators.

### NOTES:
 - if BinopGradCommon() , backward op are SumExt/ReduceSum, so sum_ext.cc and reshape.cc are needed.
 - check `op_plugin/ops/kernel/` first; if ops are already there, no need to write, but notify in the report.
