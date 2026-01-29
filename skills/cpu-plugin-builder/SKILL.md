---
name: generate-kernel-cc
description: Build MindSpore CPU operators by adapting ATen (libtorch) operators. Use when implementing new CPU operators for mindspore_op_plugin, writing kernel .cc files, adding forward/backward ops, or creating operator tests. For tasks involving mint.*, Tensor.*, operator adaptation, or CPU kernel development.
---

# Generate Kernel CC File

This skill helps you generate a C++ kernel file for adapting a MindSpore operator to ATen backend in mindspore_op_plugin.

## When to use this skill

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

### Step 1: Identify the Operator

The source of truth is the export in `mindspore/python/mindspore/mint/__init__.py`.

1) Locate mint export:
```bash
rg -n "^(from|import|\\w+\\s*=).*\\b<api_name>\\b" mindspore/python/mindspore/mint/__init__.py
```

2) Determine overload vs non-overload:

### Case A: from `mindspore.ops.functional_overload` (overload)
1. Open `mindspore/ops/api_def/<api_name>.yaml`
2. Collect all `op_yaml:` entries, ignore any with `deprecated/`
3. For each `op_yaml`, locate:
   - `mindspore/ops/op_def/yaml/<op_yaml>`
4. If any required YAML is missing, stop: **`can't find ops yaml`**

Notes:
- One mint API can map to multiple op_defs (e.g. `max.yaml` → `max_op.yaml`, `max_dim_op.yaml`).
- Kernel filename usually matches `op_yaml` without `_op.yaml`: `max_dim_op.yaml` → `max_dim.cc`.

### Case B: not from `mindspore.ops.functional_overload` (non-overload)
1. Find the alias in `mint/__init__.py`:
   - Example: `from mindspore.ops.auto_generate import cummin_ext as cummin`
2. Locate the primitive class in:
   - `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py`
3. Map primitive class → `op_def` YAML:
   - `CamelCase` → `snake_case` + `_op.yaml`
   - Example: `CumminExt` → `cummin_ext_op.yaml`
4. Locate the op definition:
   - `mindspore/ops/op_def/yaml/<snake_name>_op.yaml`
5. If not found, stop: **`can't find ops yaml`**

### conditions (before any implementation)
- If `mindspore/ops/api_def/<api_name>.yaml` has `alias:` → **need to implement the alias API**
- If kernel already exists:
  - `mindspore_op_plugin/op_plugin/ops/kernel/<kernel_name>.cc`
  - no need to implement double

Locate Backward Dependencies

MindSpore backward ops come in two forms:

1. **Composite bprop**: Gradient composed of basic ops (e.g., `Sin` grad uses `Cos` + `Mul`) - no separate `SinGrad` needed
2. **Standalone Grad primitive**: bprop calls `XXXGrad` explicitly - need to implement `XXXGrad` kernel

**How to find backward dependencies:**

```bash
# Search C++ expander (priority)
rg 'REG_BPROP_BUILDER\("<PrimName>"\)' mindspore/ccsrc/frontend/expander/grad/

# Search Python grad registration
rg 'bprop_getters\.register' mindspore/python/mindspore/ops/_grad_experimental/
```

**From bprop implementation:**
- If `Emit("XXXGrad", {...})` or `P.XXXGrad()` appears → implement `XXXGrad` kernel
- If only basic ops (`Mul`/`Add`/`Cos`/`Neg`) → ensure those ops are already implemented

**If `XXXGrad` exists, find its signature:**
```bash
cat mindspore/ops/op_def/yaml/xxx_grad_op.yaml
```
### Step 2: Classify the Operator

Before implementing, classify the operator to determine which template to use. This helps ensure consistent code patterns.

#### Category Classification Table

| Category | Description | Key Characteristics | Template File |
|----------|-------------|---------------------|---------------|
| **01_unary** | Unary operations | Single input, single output, element-wise | `reference/01_unary_ops.cc` |
| **02_binary** | Binary operations | Two inputs, single output, element-wise/broadcast | `reference/02_binary_ops.cc` |
| **03_tensor_scalar** | Tensor-Scalar mixed | Tensor + Scalar inputs | `reference/03_tensor_scalar_ops.cc` |
| **04_reduction** | Reduction operations | Reduce along dims, has keepdims | `reference/04_reduction_ops.cc` |
| **05_generator** | Generator operations | No input or scalars only, creates new tensor | `reference/05_generator_ops.cc` |
| **06_random** | Random operations | Requires seed/offset generator | `reference/06_random_ops.cc` |
| **07_multi_output** | Multi-output operations | Multiple output tensors | `reference/07_multi_output_ops.cc` |
| **08_gradient** | Gradient operations | Backward pass computation | `reference/08_gradient_ops.cc` |
| **09_inplace** | Inplace operations | Uses xxx_() to modify input | `reference/09_inplace_ops.cc` |
| **10_indexing** | Indexing operations | Involves index tensor | `reference/10_indexing_ops.cc` |
| **11_shape** | Shape operations | Changes tensor shape/layout | `reference/11_shape_ops.cc` |
| **12_nn** | Neural network operations | Complex NN layers (conv, norm) | `reference/12_nn_ops.cc` |
| **13_comparison** | Comparison operations | Outputs bool tensor | `reference/13_comparison_ops.cc` |
| **14_linalg** | Linear algebra operations | Matrix operations, decompositions | `reference/14_linalg_ops.cc` |

#### Classification Examples

| Operator | Category | Reason |
|----------|----------|--------|
| `abs`, `sin`, `cos`, `exp`, `sqrt`, `sigmoid` | 01_unary | Single input element-wise |
| `add`, `sub`, `mul`, `div`, `maximum` | 02_binary | Two inputs element-wise |
| `add_scalar`, `sub_scalar`, `pow_tensor_scalar` | 03_tensor_scalar | Tensor + Scalar mixed |
| `sum`, `mean`, `max` (global), `reduce_max` | 04_reduction | Reduce along dimensions |
| `arange`, `linspace`, `zeros`, `ones`, `empty`, `eye` | 05_generator | Create tensor from scalars |
| `randn`, `rand`, `bernoulli`, `uniform`, `normal` | 06_random | Requires random generator |
| `max_dim`, `cummax`, `topk`, `sort`, `svd` | 07_multi_output | Returns values + indices |
| `sigmoid_grad`, `sqrt_grad`, `convolution_grad` | 08_gradient | Backward gradient computation |
| `inplace_copy`, `inplace_fill`, `inplace_add` | 09_inplace | Modifies tensor inplace |
| `index_select`, `gather`, `scatter`, `narrow` | 10_indexing | Uses index operations |
| `concat`, `stack`, `tile`, `flatten`, `transpose` | 11_shape | Shape transformation |
| `conv2d`, `batch_norm`, `softmax`, `pooling` | 12_nn | Neural network layers |
| `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `isclose` | 13_comparison | Returns bool tensor |
| `matmul`, `bmm`, `mm`, `dot`, `addmm`, `outer` | 14_linalg | Linear algebra operations |

#### How to Classify

1. **Check input/output pattern**:
   - Single tensor in → **unary** or **reduction**
   - Two tensors in → **binary**, **comparison**, or **linalg**
   - Scalars only → **generator**
   - Has seed/offset → **random**
   - Multiple outputs → **multi_output**

2. **Check operation type**:
   - Modifies input directly → **inplace**
   - Uses index tensor → **indexing**
   - Changes shape → **shape**
   - Neural network layer → **nn**
   - Has "Grad" suffix → **gradient**

3. **Select the matching template** from `reference/` directory and adapt it.

### Step 3: Find the ATen Interface

Search for the corresponding ATen operator:

1. **Location**: `./third_party/libtorch/include/ATen/`
2. **Prefer `_out` variants**: Use `at::xxx_out()` when available to write directly to output tensor
3. **Fallback to copy**: If no `_out` variant exists, use `at::xxx()` then `copy_()` to output

Reference files:
- `aten/src/ATen/native/native_functions.yaml` for operator definitions
- `aten/src/ATen/templates/RedispatchFunctions.h` for `_out` variants

### Step 4: Create the Kernel File

Location: `op_plugin/ops/kernel/{operator_name}.cc`

1. Copy the matching template from `reference/` directory based on Step 2 classification
2. Replace `{{OperatorName}}` with actual operator name (CamelCase)
3. Replace `{{aten_function}}` with actual ATen function name
4. Adjust parameter indices according to YAML definition
5. Handle optional parameters if needed

## Code Quality Requirements

1. **Line length**: Maximum 120 characters per line
2. **File ending**: Must end with a newline character
3. **No trailing whitespace**: Remove all trailing spaces/tabs
4. **Comments in extern "C"**: Use `/* */` style, NOT `//`
5. **Copyright header**: Must use 2026 (current year)

### Quality Check Commands

```bash
# Check line length (should be <= 120)
awk 'length > 120 {print NR": "length" chars"}' op_plugin/ops/kernel/{operator_name}.cc

# Remove trailing whitespace
sed -i 's/[ \t]*$//' op_plugin/ops/kernel/{operator_name}.cc

# Ensure file ends with newline
[ -n "$(tail -c1 op_plugin/ops/kernel/{operator_name}.cc)" ] && echo >> op_plugin/ops/kernel/{operator_name}.cc
```
