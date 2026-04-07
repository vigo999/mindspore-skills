---
description: Diagnose a training failure, accuracy problem, or performance bottleneck by routing to the right specialist skill in diagnose mode
---

# Diagnose

Use this as the top-level diagnosis entrypoint for post-run problems.

Do lightweight deterministic routing first, then load exactly one specialist
skill in `diagnose` mode:

- `failure-agent`
- `accuracy-agent`
- `performance-agent`

This command is analyze-only. It must stop after diagnosis, root cause, and
reporting. It must not apply fixes.

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

- run in `diagnose` mode
- collect evidence first
- diagnose the root cause
- stop after the diagnosis report
- do not edit code, config, or environment

If the user actually needs pre-run environment validation instead of diagnosis,
redirect to `readiness-agent`.
