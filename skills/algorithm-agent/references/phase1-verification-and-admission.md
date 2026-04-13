# Phase-1 Verification and Admission

Keep verification scaffold guidance and factory/template admission guidance
combined in phase 1 unless reuse clearly justifies splitting.

## Verification Scaffold

Minimum repeatable checks:

- `torch` smoke forward
- `torch` backward / training-step sanity when relevant
- `torch_npu` smoke forward
- `torch_npu` backward / training-step sanity when relevant
- MindSpore NPU smoke forward
- MindSpore NPU backward / training-step sanity when relevant
- shape/dtype consistency checks
- feature on/off regression checks
- standard accuracy-drift classification output

## Handoff Rules

Hand off to:

- `readiness-agent` for environment/run readiness blockers
- `accuracy-agent` for wrong-result, mismatch, or drift after successful
  execution

## Combined-helper default

Phase 1 should default to one combined helper/scaffold script covering adjacent:
- intake artifact generation
- code-map artifact generation
- verification report generation

Only split the helper if execution proves the combined shape is no longer
maintainable.

## Runnable vs Scaffold-only Boundary

### Must be runnable in phase 1
- intake artifact generation
- code-map artifact generation
- verification report generation with expected check slots and statuses
- at least one worked proving-case pass through the scaffold path

### May remain scaffold-only in phase 1
- full automation of all torch / torch_npu / MindSpore execution steps
- automatic collection of every metric/check without manual assistance
- broad multi-case generation beyond the proving set

## Admission Gate

### Hard blockers
A feature must not enter factory/template if any of these fail:

- minimum verification scaffold is incomplete
- baseline-off behavior is not preserved
- code-map artifact is missing
- verification artifact is missing
- target-family integration touchpoints are not identified
- unresolved correctness/drift issue exists without explicit
  `accuracy-agent` handoff status

### Warnings
A feature may enter with warnings recorded when:

- only one model-family instance has been validated
- performance characterization is incomplete
- some optional robustness checks are pending
- paper/code ambiguity remains but does not block the validated path

### Minimum artifact set before admission
- intake artifact
- code-map artifact
- verification artifact
- one family-specific template or case instance
- admission checklist result with blockers/warnings explicitly recorded
