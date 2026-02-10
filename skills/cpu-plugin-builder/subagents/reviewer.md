# Reviewer

Run scope compliance and quality review before final report.

## Inputs
- scope artifact `mindspore_op_plugin/.skill_artifacts/op_scope.json`
- changed kernel/test files

## Checks
1. Scope compliance:
   - implemented forward ops == `missing_forward_ops`
   - implemented backward ops == `missing_backward_ops`
   - no extra out-of-scope ops
2. Coding compliance:
   - follow `reference/how_to_review_code`
   - one operator per kernel file
   - prefer ATen `_out` variant when appropriate
3. Test compliance:
   - follow `reference/how to write the functional test`
   - ensure forward/backward and `pynative`/`KBK` are covered
4. Execution:
   - `bash build.sh`
   - `python tests/run_tests.py --type functional --op op_name`

## Output
- findings by severity (file + reason)
- pass/fail decision
- residual risks
