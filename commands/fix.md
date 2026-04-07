---
description: Diagnose and fix a training failure, accuracy problem, or performance bottleneck by routing to the right specialist skill in fix mode
---

# Fix

Use this as the top-level fix entrypoint for post-run problems.

Do lightweight deterministic routing first, then load exactly one specialist
skill in `fix` mode:

- `failure-agent`
- `accuracy-agent`
- `performance-agent`

`/fix` must include diagnosis as its first phase. It must not apply any change
until the diagnosis is presented and the user explicitly confirms the proposed
fix.

## Routing Rules

Classify from the user's wording and any directly visible evidence:

- failure keywords:
  - crash
  - error
  - failed
  - operator
  - CANN
  - ACLNN
  - RuntimeError
  - hang
  - timeout
  - OOM
  - not implemented
- accuracy keywords:
  - accuracy
  - drift
  - regression
  - eval
  - wrong result
  - mismatch
  - loss diverge
  - NaN
  - Inf
- performance keywords:
  - throughput
  - latency
  - slow
  - memory
  - bottleneck
  - utilization
  - profiler
  - tok/s

## Routing Decision

- if the problem is a crash, runtime failure, unsupported op, backend abort,
  hang, or OOM, load `failure-agent`
- if the workload runs but outputs, metrics, or convergence are wrong, load
  `accuracy-agent`
- if the workload runs and results are correct but speed, utilization, or
  memory behavior are poor, load `performance-agent`

If classification is ambiguous, ask the user to choose exactly one:

1. training crashed or failed to start
2. accuracy or evaluation results are wrong
3. performance is worse than expected

## Execution Contract

After choosing the specialist skill, tell it explicitly:

- run in `fix` mode
- diagnose first
- propose one concrete fix plan with expected impact
- wait for user confirmation before applying anything
- apply the minimum necessary change only after confirmation
- verify the result and report before/after evidence

If the user actually needs pre-run environment validation instead of diagnosis,
redirect to `readiness-agent`.
