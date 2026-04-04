# Use Cases

This document summarizes the most common use cases currently targeted by MindSpore Skills.

## 1. Readiness before first run

Typical questions:

- Is the local training workspace ready?
- Are the dependency versions compatible?
- Is the execution environment complete enough to start?

Primary skills:

- `readiness-agent`

## 2. Failure after training start

Typical questions:

- Why did training fail?
- Which layer is responsible: script, framework, operator, runtime, or environment?
- What evidence supports the root-cause hypothesis?

Primary skills:

- `failure-agent`

## 3. Accuracy mismatch after successful execution

Typical questions:

- Why do results drift from a baseline?
- Why is loss or metric behavior inconsistent?
- Is the difference caused by data, API behavior, operator behavior, or accumulation effects?

Primary skills:

- `accuracy-agent`

## 4. Performance bottleneck analysis

Typical questions:

- Why is throughput low?
- Why is utilization poor?
- Is the bottleneck in dataloading, host/device interaction, memory, communication, or operator behavior?

Primary skills:

- `performance-agent`

## 5. Migration and adaptation routing

Typical questions:

- How should a HuggingFace or third-party repo be migrated to MindSpore?
- What is the right adaptation path?
- Where does operator or algorithm work belong?

Primary skills:

- `migrate-agent`
- `operator-agent`
- `algorithm-agent`
