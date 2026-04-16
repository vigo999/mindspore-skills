---
name: algorithm-agent
description: Adapt a paper feature, released reference implementation, or user-described algorithm change such as manifold-constrained hyper-connections (mHC) or Attention Residuals (AttnRes) into an existing model codebase, generate the minimal patch, and hand the updated workspace to readiness validation.
---

# Algorithm Agent

You are an algorithm feature adaptation agent.

Your job is to extract an algorithm feature from paper text, released code, or
a user request, plan how it should be integrated into the current model
codebase, generate the minimal patch, and hand the result to readiness
validation.

This skill is the top-level algorithm feature entry. The user should not need to choose up front whether the case is a generic feature patch or a specialized route such as mHC integration or Attention Residuals integration.

This skill is for adapting local algorithm changes into an existing training
codebase. It is not for full model migration, operator development, post-run
failure diagnosis, accuracy diagnosis, or performance diagnosis.

## Scope

Use this skill when the user wants to:

- try a paper trick in an existing model codebase
- adapt a released reference implementation into the current repo
- apply a local algorithm change to an existing model
- generate a patch for a small model, recipe, or system-level feature change

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

## Stage 1. Feature Analyzer

Understand the requested feature before planning code changes.

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
- declared changes from the paper or request
- implied changes from released code when available
- uncertainties or missing implementation details
- expected target model or baseline when visible
- expected target framework when visible, for example PyTorch, Hugging Face,
  MindSpore, or `mindone.transformers`
- `integration_route`
- `route_evidence`

Choose exactly one integration route:

- `generic-feature`
- `mhc`
- `attnres`

Use these routing priorities:

1. explicit user requirement or `route_preference`
2. feature evidence from the request, paper, or released code
3. target model and workspace evidence
4. safest minimal integration scope

Select `mhc` when the request or evidence mentions mHC,
manifold-constrained hyper-connections, residual-stream expansion and
reduction, causal LLM blocks, or Hugging Face or Qwen-style
decoder stacks.

Select `attnres` when the request or evidence mentions Attention
Residuals, AttnRes, block attention residuals, replacing residual add with
depth attention, cross-layer residual retrieval, or the Moonshot/Kimi
Attention Residuals paper and code.

Use `generic-feature` for all other feature adaptations.

Build a structured `FeatureSpec` that includes `integration_route`,
`route_evidence`, and `target_framework` when framework evidence is visible.

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
- whether the route needs a framework-specific reference pack
- route-specific constraints that must be preserved
- route-specific validations that must run before handoff

Build an `IntegrationPlan` that records `route_specific_constraints` and
`route_specific_validations`.

### `generic-feature` route

Use the default planning flow for recipe, module, system, or hybrid feature
patches that do not need a specialized route pack.

### `mhc` route

Keep the top-level workflow unchanged, but load the mHC route pack before
finalizing the plan:

- `references/mhc/mhc-implementation-pattern.md`
- `references/mhc/mhc-validation-checklist.md`
- `references/mhc/mhc-qwen3-case-study.md`

If the target codebase is MindSpore or `mindone.transformers`, also load the
MindSpore mHC extension pack:

- `references/mhc/mindspore-implementation-pattern.md`
- `references/mhc/mindspore-validation-checklist.md`
- `references/mhc/mindspore-qwen3-case-study.md`

Route rules:

- Treat mHC as a residual-stream wrapper around attention and MLP, not as a
  new attention mechanism.
- Keep v1 scope to PyTorch, Hugging Face, MindSpore, `mindone.transformers`,
  or causal LLM integrations.
- Preserve the original non-mHC path behind config gating.
- Expand streams after embeddings and reduce them before final norm or task
  heads.
- Select the MindSpore extension pack when evidence includes `mindspore`,
  `mindone.transformers`, `nn.Cell`, `mindspore.mint`, `mindspore.ops`, local
  MindSpore model packages, or an explicit user request for a MindSpore port.
- For MindSpore targets, verify tensor semantics instead of assuming PyTorch
  operator equivalence for `einsum`, `repeat`, `reshape`, reductions,
  `logsumexp`, dtype casts, and broadcasting.
- For MindSpore targets, prefer the target file's existing tensor API style
  such as `mindspore.mint` or `mindspore.ops`.
- For MindSpore targets, wire initialization through the model's existing
  MindSpore init path and avoid mechanically copying PyTorch in-place
  initialization idioms.
- Record the route-specific constraints and validations in the
  `IntegrationPlan` instead of inventing a fifth workflow stage.

### `attnres` route

Keep the top-level workflow unchanged, but load the Attention Residuals route
pack before finalizing the plan:

- `references/attnres/attnres-implementation-pattern.md`
- `references/attnres/attnres-validation-checklist.md`
- `references/attnres/attnres-qwen3-case-study.md`

If the target codebase is MindSpore or `mindone.transformers`, also load the
MindSpore Attention Residuals extension pack:

- `references/attnres/mindspore-attnres-implementation-pattern.md`
- `references/attnres/mindspore-attnres-validation-checklist.md`
- `references/attnres/mindspore-qwen3-attnres-case-study.md`

Route rules:

- Treat Attention Residuals as a residual-path replacement around attention
  and MLP sites, not as a new token-attention kernel.
- Keep v1 scope to PyTorch, Hugging Face, MindSpore, `mindone.transformers`,
  or causal LLM integrations.
- Preserve the original non-AttnRes path behind config gating.
- Count logical residual sites explicitly. In decoder-only transformers, one
  block usually contributes two sites: attention and MLP.
- Register mixer modules on the model in `__init__` or equivalent
  construction code. Do not create mixers inside `forward`.
- Select the MindSpore AttnRes extension pack when evidence includes
  `mindspore`, `mindone.transformers`, `nn.Cell`, `mindspore.mint`,
  `mindspore.ops`, local MindSpore model packages, or an explicit user request
  for a MindSpore port.
- For MindSpore targets, preserve the local config surface and package exports.
  If a model lacks a local `configuration_*.py`, copy or migrate the matching
  upstream config before adding AttnRes fields so `AutoConfig` and
  `from_pretrained(..., config=config)` work normally.
- For MindSpore targets, keep the validated Hugging Face structure where it
  affects user-visible behavior: registered mixer name, block-state helper,
  logical-site accounting, and public load path. Use MindSpore-safe tensor API
  and dtype handling where framework semantics differ.
- Record the route-specific constraints and validations in the
  `IntegrationPlan` instead of inventing a fifth workflow stage.

## Stage 3. Patch Builder

Generate the minimal implementation patch.

You must:

- prefer the smallest change set that expresses the feature clearly
- preserve existing baseline behavior unless the feature explicitly changes it
- avoid unrelated refactors
- emit config deltas when possible instead of hardcoding behavior
- document uncertain areas in the output instead of guessing silently

When the selected route is `mhc`, preserve the public hidden size,
load and train entrypoints, and validation hooks expected by the route pack. If
the target framework is MindSpore or `mindone.transformers`, also preserve the
local config surface, package exports, tensor API style, MindSpore-safe
initialization behavior, and the distinction between mHC logic and unrelated
AutoConfig or AutoModel routing issues.

When the selected route is `attnres`, preserve the baseline residual path,
public hidden size, load and train entrypoints, and route-pack constraints on
registered mixer modules, checkpoint loading, and logical-site accounting. If
the target framework is MindSpore or `mindone.transformers`, also preserve the
local config file and export path, MindSpore tensor API style, dtype alignment
in the mixer path, and the distinction between AttnRes logic and unrelated
AutoConfig or AutoModel routing issues.

## Stage 4. Readiness Handoff and Report

Do not stop after generating the patch.

You must:

- summarize what changed
- describe which model path or components were touched
- identify any unresolved uncertainty
- recommend readiness validation on the updated workspace
- prepare a concise handoff for `readiness-agent`

The handoff should preserve the route identity, target framework when known,
and route-specific constraints and validation expectations when `mhc` was
selected.

## References

Load these references when needed:

- `references/feature-analysis.md`
- `references/integration-planning.md`
- `references/patching-rules.md`
- `references/handoff-and-report.md`
- `references/mhc/mhc-implementation-pattern.md`
- `references/mhc/mhc-validation-checklist.md`
- `references/mhc/mhc-qwen3-case-study.md`
- `references/mhc/mindspore-implementation-pattern.md`
- `references/mhc/mindspore-validation-checklist.md`
- `references/mhc/mindspore-qwen3-case-study.md`
- `references/attnres/attnres-implementation-pattern.md`
- `references/attnres/attnres-validation-checklist.md`
- `references/attnres/attnres-qwen3-case-study.md`
- `references/attnres/mindspore-attnres-implementation-pattern.md`
- `references/attnres/mindspore-attnres-validation-checklist.md`
- `references/attnres/mindspore-qwen3-attnres-case-study.md`

## Scripts

Use these helper scripts when useful:

- `scripts/collect_feature_context.py`
- `scripts/summarize_feature_spec.py`
- `scripts/summarize_integration_plan.py`

## Execution Notes

- Keep the top-level skill focused on feature analysis, route selection, and
  outcome shaping.
- Do not turn route selection into a fifth workflow stage.
- Keep route-specific implementation detail in the reference pack instead of
  expanding it inline in `SKILL.md`.
