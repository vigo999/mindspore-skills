# Test Writer

Write/complete functional tests for target API.

## Inputs
- target API name
- `MS_CPU_PLUGIN_SCOPE_KEY`
- scope artifact `MS_CPU_PLUGIN_SCOPE_FILE`

## Steps
1. Read `MS_CPU_PLUGIN_SCOPE_FILE` and verify `scope_key == MS_CPU_PLUGIN_SCOPE_KEY`.
2. Implement/modify `tests/st/mint/test_{API_name}.py`.
3. Follow:
   - `reference/how to write the functional test`
   - `reference/how_to_review_code` functional checklist
4. Cover both forward and backward.
5. Cover both `pynative` and `KBK`.
6. Include required special cases when supported:
   - dynamic shape/rank
   - non-contiguous input
   - dtype matrix (including `bf16` where supported)
   - edge values (`inf`, `nan`, empty tensor if valid expectation is exception)

## Output
- list of test cases added/updated
- known unsupported scenarios and reason
