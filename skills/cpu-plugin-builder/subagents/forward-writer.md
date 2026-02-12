# Forward Writer

Implement forward kernels only from scope artifact.

## Required Input
- `MS_CPU_PLUGIN_SCOPE_KEY`
- `MS_CPU_PLUGIN_SCOPE_FILE`

## Steps
1. Read `MS_CPU_PLUGIN_SCOPE_FILE` and verify `scope_key == MS_CPU_PLUGIN_SCOPE_KEY`.
2. Read `missing_forward_ops`.
3. For each missing op:
   - map to ATen interface (`how_to_find_aten_interface.md`)
   - implement one operator per `.cc` file under `op_plugin/ops/kernel/`
4. Follow forward coding guide:
   - `reference/how_to_write_forward_op.md`
5. Keep scope boundary:
   - do not add operators not listed in `missing_forward_ops`.

## Output
- list of created/updated forward kernel files
