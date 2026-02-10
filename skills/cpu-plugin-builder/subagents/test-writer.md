# Test Writer

Write/complete functional tests for target API.

## Inputs
- target API name
- scope artifact `mindspore_op_plugin/.skill_artifacts/op_scope.json`

## Steps
1. Implement/modify `tests/st/mint/test_{API_name}.py`.
2. Follow:
   - `reference/how to write the functional test`
   - `reference/how_to_review_code` functional checklist
3. Cover both forward and backward.
4. Cover both `pynative` and `KBK`.
5. Include required special cases when supported:
   - dynamic shape/rank
   - non-contiguous input
   - dtype matrix (including `bf16` where supported)
   - edge values (`inf`, `nan`, empty tensor if valid expectation is exception)

## Output
- list of test cases added/updated
- known unsupported scenarios and reason
