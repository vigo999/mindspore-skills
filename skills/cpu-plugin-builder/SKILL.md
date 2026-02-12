---
name: cpu-plugin-builder
description: Build MindSpore CPU operators by adapting ATen (libtorch) operators via mindspore_op_plugin. Use when implementing op_plugin/ops/kernel/*.cc and mint tests. This skill orchestrates internal sub-agents for scope resolving, forward/backward kernel writing, test writing, and review.
---

# CPU Plugin Builder

Use this orchestrator skill for CPU plugin work.

## Session Isolation (mandatory)

Before Step 1, initialize a session-unique scope artifact path and reuse it for all steps.

```bash
export MS_CPU_PLUGIN_SCOPE_KEY="${MS_CPU_PLUGIN_SCOPE_KEY:-${CODEX_SESSION_ID:-$(date +%s)-$$}}"
export MS_CPU_PLUGIN_SCOPE_FILE="mindspore_op_plugin/.skill_artifacts/${MS_CPU_PLUGIN_SCOPE_KEY}/op_scope.json"
mkdir -p "$(dirname "$MS_CPU_PLUGIN_SCOPE_FILE")"
```

Rules:
- Never use a shared fixed path such as `mindspore_op_plugin/.skill_artifacts/op_scope.json`.
- Read and write scope artifact only through `MS_CPU_PLUGIN_SCOPE_FILE`.
- Scope artifact must include `scope_key` and it must match `MS_CPU_PLUGIN_SCOPE_KEY`.
- Write atomically: write to temp file in the same directory, then `mv` to `MS_CPU_PLUGIN_SCOPE_FILE`.

## Internal Sub-Agents

- `subagents/scope-resolver.md`
- `subagents/forward-writer.md`
- `subagents/backward-writer.md`
- `subagents/test-writer.md`
- `subagents/reviewer.md`

## Instructions

### Step 1: Resolve scope first (mandatory)
- Read `subagents/scope-resolver.md`.
- Use `api-helper` to resolve call chain and backward chain.
- Write scope artifact to `MS_CPU_PLUGIN_SCOPE_FILE`.

### Step 2: Implement forward kernels only from scope artifact
- Read `subagents/forward-writer.md`.
- Implement only operators listed in `missing_forward_ops` from `MS_CPU_PLUGIN_SCOPE_FILE`.

### Step 3: Implement backward kernels only from scope artifact
- Read `subagents/backward-writer.md`.
- Implement only operators listed in `missing_backward_ops` from `MS_CPU_PLUGIN_SCOPE_FILE`.
- Do not skip primitive ops in mixed chains (for example `ExpandDims`, `Reshape`) if listed.

### Step 4: Write/update functional tests
- Read `subagents/test-writer.md`.
- Implement in `mindspore_op_plugin/tests/st/mint/test_{API_name}.py`.

### Step 5: Build and run tests
cd `mindspore_op_plugin`
build with `bash build.sh`
get env ready : `source env.source`
run test : `python tests/run_tests.py --type functional --op op_name`

### Step 6: Review and enforce scope compliance
- Read `subagents/reviewer.md`.
- Confirm implemented ops exactly match the scope artifact and review checklist.

### Step 7: Write report
Report must include:
- `scope_key` and scope artifact path
- `forward_ops` and kernel file names
- `backward_ops` and kernel file names
- test command and results
- any out-of-scope request and why it was rejected
