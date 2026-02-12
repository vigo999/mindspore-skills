# Reviewer

Run scope compliance and quality review before final report.

## Inputs
- `MS_CPU_PLUGIN_SCOPE_KEY`
- scope artifact `MS_CPU_PLUGIN_SCOPE_FILE`
- changed kernel/test files

## Checks
1. Artifact integrity:
   - `scope_key` in `MS_CPU_PLUGIN_SCOPE_FILE` must equal `MS_CPU_PLUGIN_SCOPE_KEY`
2. Scope compliance:
   - implemented forward ops == `missing_forward_ops`
   - implemented backward ops == `missing_backward_ops`
   - no extra out-of-scope ops
3. Coding compliance:
   - follow `reference/how_to_review_code`
   - one operator per kernel file
   - prefer ATen `_out` variant when appropriate
4. Test compliance:
   - follow `reference/how to write the functional test`
   - ensure forward/backward and `pynative`/`KBK` are covered
5. Execution:
   - `bash build.sh`
   - `python tests/run_tests.py --type functional --op op_name`

## Output
- findings by severity (file + reason)
- pass/fail decision
- residual risks
