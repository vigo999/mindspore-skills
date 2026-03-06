# Scope Resolver

Resolve exact implementation scope before writing kernels.

## Inputs
- target API name (for example `mint.broadcast_to`, `mint.nn.AdaptiveMaxPool1d`)
- `MS_CPU_PLUGIN_SCOPE_KEY`
- `MS_CPU_PLUGIN_SCOPE_FILE`

## Steps
1. Use `api-helper` to resolve:
   - real forward primitive
   - bprop registration op name
2. Verify forward primitive existence:
   - check `mindspore/ops/api_def`
   - check `mindspore/ops/op_def/yaml`
   - check generated prims
3. Parse full backward body:
   - enumerate all primitive calls, not only `Emit("XXXGrad", ...)`
   - include mixed chains (`ExpandDims`, `Reshape`, etc.)
4. Classify operators:
   - `forward_ops`
   - `backward_emit_ops`
   - `backward_required_primitives`
   - `backward_non_kernel_helpers` (`OutZeros`, `ShapeCalc`, `TupleGetItem`, `EmitValue`)
5. Check existing kernel files in `mindspore_op_plugin/op_plugin/ops/kernel/`.
6. Produce missing-op lists.

## Output Artifact
Write `MS_CPU_PLUGIN_SCOPE_FILE` with keys:
- `scope_key`
- `api_name`
- `forward_primitive`
- `forward_ops`
- `backward_emit_ops`
- `backward_required_primitives`
- `backward_non_kernel_helpers`
- `missing_forward_ops`
- `missing_backward_ops`
- `notes`

Write atomically:
1. write JSON to a temp file under `$(dirname "$MS_CPU_PLUGIN_SCOPE_FILE")`
2. `mv` temp file to `MS_CPU_PLUGIN_SCOPE_FILE`

## Hard Rule
Writers may implement only operators listed in `missing_forward_ops` and `missing_backward_ops`.
`scope_key` in artifact must equal `MS_CPU_PLUGIN_SCOPE_KEY`.
