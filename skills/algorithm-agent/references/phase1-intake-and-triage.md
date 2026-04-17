# Phase-1 Intake and Triage

Status: not part of the current live `algorithm-agent` skill flow.

Keep phase-1 paper intake compact.

This reference combines:
- intake artifact definition
- intake scoring / triage rubric
- default routing behavior
- first proving-case calibration guidance

## Purpose

Convert paper discovery into:
- a structured intake artifact
- a ranked shortlist
- a recommended next action

Use DeepXiv as the preferred/default paper-intake assistant, but keep intake
source-flexible when other sources improve coverage.

## Intake Artifact

Record at least:

- `source_type`
- `paper_title`
- `paper_url`
- `code_url`
- `claimed_contribution`
- `target_task`
- `target_model_family`
- `feature_bucket`
- `likely_integration_surface`
- `dependency_complexity`
- `verification_risk`
- `migration_blockers`
- `qualification_basis`
- `source_status`
- `recommended_next_action`

## Scoring / Triage Rubric

Score or tag each candidate on:

- `feature_bucket`
- `paper_clarity`
- `code_availability`
- `target_family_fit`
- `integration_surface_clarity`
- `bridge_value`
- `verification_value`
- `phase1_tractability`
- `recommended_next_action`

### Hard gates before extraction

The following must not be low for a phase-1 proving candidate:

- `code_availability`
- `target_family_fit`
- `phase1_tractability`

If any of these are low, default to `reject` or `watchlist`.

### Ranking factors

Use these to rank candidates that pass the gates:

- `bridge_value`
- `verification_value`
- `integration_surface_clarity`
- `paper_clarity`

### High / Medium / Low guidance

#### `code_availability`
- high: maintained public repo with reusable core logic
- medium: partial or demo code with recoverable implementation logic
- low: no usable public code path

#### `target_family_fit`
- high: natural fit for Qwen / DeepSeek-like decoder-only models
- medium: adaptable with moderate reshaping
- low: poor or unclear fit

#### `bridge_value`
- high: strongly exercises paper -> code-map -> patch-plan
- medium: only some bridge steps are informative
- low: mostly template reuse with little bridge learning

#### `verification_value`
- high: meaningfully stresses torch / torch_npu / MindSpore scaffold
- medium: validates only part of the scaffold
- low: little new scaffold value

#### `phase1_tractability`
- high: realistic proving-case scope now
- medium: feasible with constrained scope
- low: too large or ambiguous for phase 1

## Default routing behavior

- any hard-gate failure -> `reject` or `watchlist`
- passes gates but weak ranking value -> `reference-code extraction`
- passes gates with strong `bridge_value` + `verification_value` + at least
  medium `integration_surface_clarity` -> `proving candidate`
- promising but blocked by timing or ambiguity -> `watchlist`

## Phase-1 proving-case calibration

Use `TransMLA` as the first worked example for calibrating:
- intake scoring
- artifact shape
- minimum helper behavior
- minimum file split decisions

Use `mHC` and `AttnRes` as reuse anchors for existing route-pack patterns, not
as reasons to expand file structure early.

