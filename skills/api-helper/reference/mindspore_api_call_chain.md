### Understand Mindspore API Call Chains

This guide helps you locate and identify API/Operator names in the MindSpore codebase

### Instructions

#### Step 1: Identify Forward Operator Name
 - mint api import -> `mindspore/python/mindspore/mint/__init__.py`
 - API definition -> `mindspore/ops/api_def/`
 - OP definition -> ``mindspore/ops/op_def/yaml/`

```python
# Example: mint.linspace
from mindspore.ops.function.math_func import linspace_ext as linspace
```
This tells you:
- **Module path**: `mindspore.ops.function.math_func`
- **Internal function name**: `linspace_ext`
- **Public API name**: `linspace`
- **Operator name**: `LinSpaceExt` (CamelCase of `linspace_ext`)

##### Case 1: Overloaded Operators
e.g. `mint.max` has three implementations. this info can be found in `mindspore/ops/api_def/max`.
and it is import from function_overload module in `mindspore/python/mindspore/mint/__init__.py`

```
from mindspore.ops.functional_overload import max
```
e.g. yaml format is below

max:
  - op_yaml: max_op.yaml
    py_method: tensor_max
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: max_dim_op.yaml
    py_method: tensor_maxdim
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: dim
    interface: tensor, function

  - op_yaml: maximum_op.yaml
    py_method: tensor_maximum
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: function

- max_op_yaml -> `mindspore/ops/op_def/max_yaml`.
- max_dim_op.yaml -> `mindspore/ops/op_def/max_dim_op.yaml`.
- maximum_op.yaml -> `mindspore/ops/op_def/maximum_op.yaml`.

##### Case 2: Inplace Operators

`Tensor.sub_` (inplace subtraction) requires:
- Forward: `inplace_sub_ext.cc`
- The `_` suffix in Python indicates inplace operation

##### Case 3: Wrapper/Module APIs

Some APIs are class/module wrappers and are neither overload nor inplace.

Example: `mint.nn.AdaptiveMaxPool1d`
- `mint.nn.layer.pooling.AdaptiveMaxPool1d.construct`
- `mint.nn.functional.adaptive_max_pool1d`
- `ops.auto_generate.gen_ops_prim.adaptive_max_pool1d_op(...)`
- primitive from YAML/class name: `AdaptiveMaxPool1D`

#### Step 2: Identify Backward Operator Name
1. Search `mindspore/ccsrc/frontend/expander/grad/` for the **CORRECT operator name** from Step 1
2. Look for `REG_BPROP_BUILDER("<CorrectOpName>")` registration (e.g., "AcosExt" not "ACos"). parse the full REG_BPROP_BUILDER body.
3. **EXTRACT AND LIST each individual operator call from the gradient computation**:
   - Look for `ib->OperatorName(...)` patterns in the BODYFUNC
   - List each operator name (e.g., Mul, Muls, Exp, PowTensorScalar, Add, Sub, etc.)
   - Show the line of code for each operator call
   - Enumerate all `ib->...` operators in bprop body, not only `Emit("XXXGrad", ...)`.

##### Case 1: Standalone Grad Operator:
 - only uses `Emit("XXXGrad", ...)`, it is dedicated grad operator.

##### Case 2: Multiple Primitive Operators
 - For `AcosExt` backward which uses `ib->Neg(dout) * ib->Rsqrt(ib->Sub(..., ib->Square(x)))`

##### Case 3: Mixed Composite Backward Chain
 - Some bprop bodies mix helper primitives and emitted grad ops.
 - Example `AdaptiveMaxPool1D` backward chain:
  - `ExpandDims`
  - `Emit("AdaptiveMaxPool2DGrad", ...)`
  - `Reshape`
  - `OutZeros(output_size)`
 - Scope rules:
  - Kernel-required primitives: `ExpandDims`, `Reshape`, `Squeeze`, `Transpose`, `Cast`, etc.
  - Graph/meta helpers (no kernel file): `OutZeros`, `ShapeCalc`, `TupleGetItem`, `EmitValue`.

### Key Search Locations

| Purpose                  | Path                                                           |
| ------------------------ | -------------------------------------------------------------- |
| mint.* imports           | `mindspore/python/mindspore/mint/__init__.py`                  |
| Tensor.* methods         | `mindspore/python/mindspore/common/tensor/tensor.py`           |
| Function implementations | `mindspore/python/mindspore/ops/function/*.py`                 |
| Primitive definitions    | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` |
| Op YAML definitions      | `mindspore/ops/op_def/yaml/*_op.yaml`                          |
| Gradient definitions     | `mindspore/ccsrc/frontend/expander/grad/`                      |


### Examples

**Step 0: Verify operator name**
```bash
# Read mint/__init__.py line 308:
from mindspore.ops.function.math_func import acos_ext as acos
```
→ `mint.acos` uses `acos_ext` → Operator name is `AcosExt` (NOT `ACos`!)

**Step 1: Find backward with CORRECT name**
```bash
# Search for AcosExt backward:
grep -A 10 "REG_BPROP_BUILDER(\"AcosExt\")" mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc
```

**Result**:
```cpp
REG_BPROP_BUILDER("AcosExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  dx = ib->Neg(dout) * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  return {dx};
});
```

**Answer**: `mint.acos` (which uses `AcosExt`) has NO dedicated backward operator. It computes gradient inline using:
1. `Neg` - negation
2. `Mul` - multiplication
3. `Rsqrt` - reciprocal square root
4. `Sub` - subtraction
5. `Square` - squaring
