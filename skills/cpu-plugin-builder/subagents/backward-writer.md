# Backward Writer

Implement backward kernels only from scope artifact.

## Required Input
- `mindspore_op_plugin/.skill_artifacts/op_scope.json`

## Steps
1. Read `missing_backward_ops`.
2. For each missing op:
   - map to ATen interface (`how_to_find_aten_interface.md`)
   - implement one operator per `.cc` file under `op_plugin/ops/kernel/`
3. Follow backward coding guide:
   - `reference/how_to_write_backward_op.md`
4. Mixed-chain rule:
   - if `ExpandDims`, `Reshape`, `Squeeze`, `Transpose`, `Cast` appear in missing backward ops, implement them.
   - do not treat these primitive ops as auto-skippable helpers.
5. Keep scope boundary:
   - do not add operators not listed in `missing_backward_ops`.

## Output
- list of created/updated backward kernel files
