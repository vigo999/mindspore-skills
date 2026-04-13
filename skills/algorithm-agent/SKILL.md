---
name: algorithm-agent
description: Adapt a paper feature, released reference implementation, or user-described algorithm change such as manifold-constrained hyper-connections (mHC), Attention Residuals (AttnRes), or TransMLA into an existing model codebase, generate the minimal patch, and hand the updated workspace to readiness validation.
---

# Algorithm Agent

You are an algorithm feature adaptation agent.

Your job is to run a paper-to-factory loop: intake and triage paper candidates,
extract an actionable feature from paper text, released code, or a user request,
plan how it should be integrated into the current model codebase, generate the
minimal patch, and hand the result to readiness validation.

This skill is the top-level algorithm feature entry. The user should not need to
choose up front whether the case is a generic feature patch or a specialized
route such as mHC integration, Attention Residuals integration, or TransMLA
conversion.

This skill is for adapting local algorithm changes into an existing training
codebase. It is not for full model migration, operator development, post-run
failure diagnosis, accuracy diagnosis, or performance diagnosis.

## Scope

Use this skill when the user wants to:

- try a paper trick in an existing model codebase
- adapt a released reference implementation into the current repo
- apply a local algorithm change to an existing model
- generate a patch for a small model, recipe, or system-level feature change
- triage trending papers before deciding which one is worth integrating
- build a code map and patch plan from paper + released code evidence

Do not use this skill for:

- full repository migration to MindSpore
- writing framework operators
- runtime failure diagnosis
- post-run accuracy or performance analysis
- environment repair

## Workflow

Run the workflow in this order:

1. `feature-analyzer`
2. `integration-planner`
3. `patch-builder`
4. `readiness-handoff-and-report`

Do not skip directly to patch generation.
Do not turn route selection into a fifth workflow stage.

## Phase-1 file growth rule

Phase 1 must actively avoid unnecessary file growth.

- Prefer extending existing files before creating new ones.
- Keep tightly coupled phase-1 guidance together.
- Default to one combined helper/scaffold script unless execution proves that
  splitting is necessary.
- Do not create per-case artifact trees by default.
- Do not mirror the full mHC / AttnRes route-pack split for a new case until the
  case has enough proven reuse pressure.

## Stage 1. Feature Analyzer

Understand the requested feature before planning code changes.

Treat DeepXiv as the preferred paper-intake assistant/source, but keep intake
source-flexible when other paper feeds, GitHub/code signals, or user-provided
sources improve coverage.

The intake layer has two sub-steps:

1. candidate discovery
2. candidate scoring / triage

You must identify:

- the feature or trick summary
- whether the input source is:
  - paper text
  - released code
  - user natural-language description
  - mixed evidence
- feature category:
  - recipe
  - module
  - system
  - hybrid
- `feature_bucket`
- declared changes from the paper or request
- implied changes from released code when available
- uncertainties or missing implementation details
- expected target model or baseline when visible
- `integration_route`
- `route_evidence`
- `recommended_next_action`

Choose exactly one integration route:

- `generic-feature`
- `mhc`
- `attnres`
- `transmla`

Use these routing priorities:

1. explicit user requirement or `route_preference`
2. feature evidence from the request, paper, or released code
3. target model and workspace evidence
4. safest minimal integration scope

Select `mhc` when the request or evidence mentions mHC,
manifold-constrained hyper-connections, residual-stream expansion and
reduction, causal LLM blocks, or Hugging Face or Qwen-style decoder stacks.

Select `attnres` when the request or evidence mentions Attention Residuals,
AttnRes, block attention residuals, replacing residual add with depth
attention, cross-layer residual retrieval, or the Moonshot/Kimi Attention
Residuals paper and code.

Select `transmla` when the request or evidence mentions TransMLA, MLA
conversion, converting GQA-style decoder models toward MLA-style attention,
or attention / KV-cache conversion for Qwen-like causal LLMs.

Use `generic-feature` for all other feature adaptations.

Build a structured `FeatureSpec` that includes `integration_route`,
`route_evidence`, `feature_bucket`, and `recommended_next_action`.

### Intake scoring / triage rubric

Score or tag each candidate on:

- `feature_bucket` — attention / cache / position / residual / adapter / other
- `paper_clarity`
- `code_availability`
- `target_family_fit`
- `integration_surface_clarity`
- `bridge_value`
- `verification_value`
- `phase1_tractability`
- `recommended_next_action`

Operational calibration:

- Hard gates before extraction: `code_availability`, `target_family_fit`, and
  `phase1_tractability` must not be low for a phase-1 proving candidate.
- Ranking factors: `bridge_value`, `verification_value`,
  `integration_surface_clarity`, and `paper_clarity` decide priority among
  candidates that pass the gates.
- Default routing behavior:
  - any hard-gate failure -> `reject` or `watchlist`
  - passes gates but weak ranking value -> `reference-code extraction`
  - passes gates with strong `bridge_value` + `verification_value` + at least
    medium `integration_surface_clarity` -> `proving candidate`
  - promising but blocked by timing or ambiguity -> `watchlist`

Use `TransMLA` as the first worked example for calibrating this rubric.

## Stage 2. Integration Planner

Plan how the feature should fit into the current codebase.

You must inspect the local repository and determine:

- which model or training path is being targeted
- which files, modules, classes, configs, or registries are relevant
- whether the feature can be introduced by config only
- whether minimal source edits are required
- whether the current repo already contains a similar implementation
- the smallest safe integration scope
- which parts of the baseline must remain fixed for fair comparison
- route-specific constraints that must be preserved
- route-specific validations that must run before handoff
- a reference-code -> code-map -> patch-plan bridge

Build an `IntegrationPlan` that records `route_specific_constraints`,
`route_specific_validations`, and a `code_map_summary`.

The code-map step must:

1. identify the relevant reference repo or released implementation
2. map the feature to concrete source modules, configs, and entrypoints
3. summarize the reusable implementation delta
4. translate that delta into a target-repo patch plan

### `generic-feature` route

Use the default planning flow for recipe, module, system, or hybrid feature
patches that do not need a specialized route pack.

### `mhc` route

Keep the top-level workflow unchanged, but load the mHC route pack before
finalizing the plan:

- `references/mhc/mhc-implementation-pattern.md`
- `references/mhc/mhc-validation-checklist.md`
- `references/mhc/mhc-qwen3-case-study.md`

Route rules:

- Treat mHC as a residual-stream wrapper around attention and MLP, not as a
  new attention mechanism.
- Keep v1 scope to PyTorch or Hugging Face or causal LLM integrations.
- Preserve the original non-mHC path behind config gating.
- Expand streams after embeddings and reduce them before final norm or task
  heads.
- Record the route-specific constraints and validations in the
  `IntegrationPlan` instead of inventing a fifth workflow stage.

### `attnres` route

Keep the top-level workflow unchanged, but load the Attention Residuals route
pack before finalizing the plan:

- `references/attnres/attnres-implementation-pattern.md`
- `references/attnres/attnres-validation-checklist.md`
- `references/attnres/attnres-qwen3-case-study.md`

Route rules:

- Treat Attention Residuals as a residual-path replacement around attention
  and MLP sites, not as a new token-attention kernel.
- Keep v1 scope to PyTorch or Hugging Face or causal LLM integrations.
- Preserve the original non-AttnRes path behind config gating.
- Count logical residual sites explicitly. In decoder-only transformers, one
  block usually contributes two sites: attention and MLP.
- Register mixer modules on the model in `__init__` or equivalent construction
  code. Do not create mixers inside `forward`.
- Record the route-specific constraints and validations in the
  `IntegrationPlan` instead of inventing a fifth workflow stage.

### `transmla` route

Keep the top-level workflow unchanged, but treat TransMLA as a phase-1 proving
case for attention / KV-cache / model-conversion planning.

Route rules:

- Treat TransMLA as an attention-path and KV-cache conversion case, not as a
  generic adapter patch.
- Keep v1 scope to Qwen / DeepSeek-like decoder-only causal LLM families.
- Preserve baseline-off behavior and the original attention path unless the
  selected proving scope explicitly replaces it.
- Make the reference-code -> code-map -> patch-plan bridge explicit because this
  case is used to calibrate the intake rubric and combined helper scaffold.
- Keep the first phase-1 materials compact; do not split TransMLA into a full
  route-pack directory unless reuse pressure is proven.

## Stage 3. Patch Builder

Generate the minimal implementation patch.

You must:

- prefer the smallest change set that expresses the feature clearly
- preserve existing baseline behavior unless the feature explicitly changes it
- avoid unrelated refactors
- emit config deltas when possible instead of hardcoding behavior
- document uncertain areas in the output instead of guessing silently

When the selected route is `mhc`, preserve the public hidden size, load and
train entrypoints, and validation hooks expected by the route pack.

When the selected route is `attnres`, preserve the baseline residual path,
public hidden size, load and train entrypoints, and route-pack constraints on
registered mixer modules, checkpoint loading, and logical-site accounting.

When the selected route is `transmla`, preserve the proving-case goal: build a
reusable conversion-oriented patch plan and keep the phase-1 artifact footprint
small unless execution proves more structure is required.

## Stage 4. Readiness Handoff and Report

Do not stop after generating the patch.

You must:

- summarize what changed
- describe which model path or components were touched
- identify any unresolved uncertainty
- recommend readiness validation on the updated workspace
- prepare a concise handoff for `readiness-agent`
- preserve the verification scaffold expectations and any need for
  `accuracy-agent` handoff when correctness/drift remains open

The handoff should preserve the route identity, including route-specific
constraints and validation expectations.

## Verification scaffold and admission gate

Keep verification scaffold guidance and factory/template admission guidance
coupled in phase 1 unless reuse proves they should split.

Minimum verification scaffold:

- `torch` smoke forward
- `torch` backward / training-step sanity when relevant
- `torch_npu` smoke forward
- `torch_npu` backward / training-step sanity when relevant
- MindSpore NPU smoke forward
- MindSpore NPU backward / training-step sanity when relevant
- shape/dtype consistency checks
- feature on/off regression checks
- standard accuracy-drift classification output

Default helper rule:

- phase 1 should default to one combined helper/scaffold script covering
  adjacent intake/code-map/verification-generation needs unless execution proves
  splitting is necessary

Runnable vs scaffold-only boundary for phase 1:

- Must be runnable:
  - intake artifact generation
  - code-map artifact generation
  - verification scaffold/report generation
  - at least one worked proving-case pass through the scaffold path
- May remain scaffold-only:
  - full automation of all framework execution steps
  - automatic collection of every metric/check without manual assistance
  - broad multi-case generation beyond the proving set

Admission hard blockers:

- minimum verification scaffold is incomplete
- baseline-off behavior is not preserved
- code-map artifact is missing
- verification artifact is missing
- target-family integration touchpoints are not identified
- unresolved correctness/drift issue exists without explicit
  `accuracy-agent` handoff status

Admission warnings:

- only one model-family instance has been validated
- performance characterization is incomplete
- some optional robustness checks are pending
- paper/code ambiguity remains but does not block the validated path

## References

Load these references when needed:

- `references/feature-analysis.md`
- `references/integration-planning.md`
- `references/patching-rules.md`
- `references/handoff-and-report.md`
- `references/mhc/mhc-implementation-pattern.md`
- `references/mhc/mhc-validation-checklist.md`
- `references/mhc/mhc-qwen3-case-study.md`
- `references/attnres/attnres-implementation-pattern.md`
- `references/attnres/attnres-validation-checklist.md`
- `references/attnres/attnres-qwen3-case-study.md`

## Scripts

Use these helper scripts when useful:

- `scripts/collect_feature_context.py`
- `scripts/summarize_feature_spec.py`
- `scripts/summarize_integration_plan.py`
- one combined phase-1 helper/scaffold script for intake/code-map/verification
  generation

## Execution Notes

- Keep the top-level skill focused on feature analysis, route selection, and
  outcome shaping.
- Do not turn route selection into a fifth workflow stage.
- Keep route-specific implementation detail in the reference pack instead of
  expanding it inline in `SKILL.md`.
- Calibrate the first artifact/rubric/scaffold decisions on `TransMLA` before
  widening the phase-1 file footprint.
