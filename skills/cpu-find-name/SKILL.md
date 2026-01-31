---
name: cpu-find-name
description: Find MindSpore CPU operators name for mindspore_op_plugin/op_plugin/ops/kernel/op_name.cc.
---

# MindSpore CPU Operator Name Finder

This skill helps you locate and identify CPU operator names in the MindSpore codebase for use with mindspore_op_plugin development.

## When to Use

Use this skill when:
- Finding the internal operator name for a `mint.*` function
- Finding the internal operator name for a `Tensor.*` method
- Tracing the call chain from Python API to C++ operator
- Identifying all operator variants needed for a single API (e.g., overloads, inplace versions)
- Preparing to implement a new operator in mindspore_op_plugin

## Quick Start

1. Identify the target API (e.g., `mint.linspace`, `Tensor.sub_`)
2. Search the corresponding `__init__.py` for the import path
3. Trace the call chain to find the primitive class name
4. Convert the primitive name to the kernel file name

## Instructions

### Step 1: Identify the API Entry Point

Determine whether the target is a `mint.*` function or a `Tensor.*` method:

| API Type | Entry Point Location |
|----------|---------------------|
| `mint.*` | `mindspore/python/mindspore/mint/__init__.py` |
| `Tensor.*` | `mindspore/python/mindspore/common/tensor/tensor.py` |

### Step 2: Trace the Import Chain

For `mint.*` functions, search the import statement in `mint/__init__.py`:

```python
# Example: mint.linspace
from mindspore.ops.function.math_func import linspace_ext as linspace
```

This tells you:
- **Module path**: `mindspore.ops.function.math_func`
- **Internal function name**: `linspace_ext`
- **Public API name**: `linspace`

### Step 3: Find the Primitive Class

Navigate to the function implementation and locate the primitive call:

```python
# In mindspore/ops/function/math_func.py
def linspace_ext(start, end, steps, *, dtype=None):
    return lin_space_ext_op(start, end, steps, dtype)
```

Then find the operator definition:

```python
lin_space_ext_op = LinSpaceExt()
```

### Step 4: Derive the Kernel Name

Convert the primitive class name to the kernel file name:

| Primitive Class | Kernel File Name |
|----------------|------------------|
| `LinSpaceExt` | `linspace_ext.cc` |
| `SubExt` | `sub_ext.cc` |
| `MaximumExt` | `maximum_ext.cc` |

**Naming convention**: CamelCase to snake_case, preserving the `_ext` suffix if present.

### Step 5: Check for Operator Variants

Some APIs require multiple kernel implementations:

#### Overloaded Operators

`mint.max` has multiple behaviors:
- `max(input)` - Returns maximum value: `max.cc`
- `max(input, dim)` - Returns max along dimension: `max_dim.cc`
- `max(input, other)` - Element-wise maximum: `maximum.cc`

#### Inplace Operators

`Tensor.sub_` (inplace subtraction) requires:
- Forward: `inplace_sub_ext.cc`
- The `_` suffix in Python indicates inplace operation

#### Operators with Gradients

If the operator supports backpropagation, you may also need:
- Forward: `<op_name>.cc`
- Backward: `<op_name>_grad.cc`

## Examples

### Example 1: Finding `mint.linspace`

```
User: find mint.linspace

Step 1: Search mint/__init__.py
  -> from mindspore.ops.function.math_func import linspace_ext as linspace

Step 2: Search linspace_ext in math_func.py
  -> return lin_space_ext_op(start, end, steps, dtype)

Step 3: Find the operator
  -> lin_space_ext_op = LinSpaceExt()

Result: Kernel name is "linspace_ext"
        File: op_plugin/ops/kernel/linspace_ext.cc
```

### Example 2: Finding `Tensor.sub_`

```
User: find Tensor.sub_

Step 1: Search tensor.py for sub_
  -> def sub_(self, other, *, alpha=1):
  ->     return mint.nn.functional.sub_(self, other, alpha=alpha)

Step 2: Search mint/nn/functional/__init__.py
  -> from mindspore.ops.function.math_func import sub_ as sub_

Step 3: Search sub_ in math_func.py
  -> return inplace_sub_ext_op(input, other, alpha)

Step 4: Find the operator
  -> inplace_sub_ext_op = InplaceSubExt()

Result: Kernel name is "inplace_sub_ext"
        File: op_plugin/ops/kernel/inplace_sub_ext.cc
```

### Example 3: Finding `mint.max` (Multiple Variants)

```
User: find mint.max

This operator has multiple variants:

Variant 1: max(input) - Global maximum
  -> MaxExt() -> "max_ext.cc"

Variant 2: max(input, dim) - Maximum along dimension
  -> MaxDimExt() -> "max_dim_ext.cc"

Variant 3: max(input, other) - Element-wise maximum
  -> MaximumExt() -> "maximum_ext.cc"

Result: Three kernel files may be needed:
        - op_plugin/ops/kernel/max_ext.cc
        - op_plugin/ops/kernel/max_dim_ext.cc
        - op_plugin/ops/kernel/maximum_ext.cc
```

## Key Search Locations

| Purpose | Path |
|---------|------|
| mint.* imports | `mindspore/python/mindspore/mint/__init__.py` |
| Tensor.* methods | `mindspore/python/mindspore/common/tensor/tensor.py` |
| Function implementations | `mindspore/python/mindspore/ops/function/*.py` |
| Primitive definitions | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` |
| Op YAML definitions | `mindspore/ops/op_def/yaml/*_op.yaml` |
| Gradient definitions | `mindspore/ccsrc/frontend/expander/grad/` |

## Common Patterns

### Standard Operators
- Python API: `mint.xxx` or `Tensor.xxx`
- Primitive: `XxxExt`
- Kernel: `xxx_ext.cc`

### Inplace Operators
- Python API: `Tensor.xxx_` (trailing underscore)
- Primitive: `InplaceXxxExt`
- Kernel: `inplace_xxx_ext.cc`

### Operators with Dimension Parameter
- Python API: `mint.xxx(input, dim=...)`
- Primitive: `XxxDimExt`
- Kernel: `xxx_dim_ext.cc`

## Troubleshooting

**Cannot find import in mint/__init__.py**: The operator may be in a submodule. Check:
- `mint.nn.functional.__init__.py`
- `mint.linalg.__init__.py`
- `mint.special.__init__.py`

**Function uses multiple primitives**: Some high-level functions compose multiple primitives. Trace each primitive call separately.

**Operator not yet defined**: If the primitive class doesn't exist, you may need to define it first in `gen_ops_prim.py` and corresponding YAML files.

## References

After finding the operator name, use the [cpu-plugin-builder](/cpu-plugin-builder) skill to implement the kernel.
