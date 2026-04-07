# Contributing to MindSpore Skills

Thank you for contributing to MindSpore Skills.

MindSpore Skills is an open skill library for AI infra and model training workflows.
The current contribution model focuses on two paths:

- content contribution
- problem collaboration

We intentionally keep the public contribution surface clear and practical.

## 1. What to contribute

### 1.1 Content contribution

Good content contributions include:

- new skills
- workflow templates
- examples and demos
- docs improvements
- diagnose patterns
- prompts, rules, and recipes
- usage clarification and boundary notes

This is the main path for turning repeated training experience into reusable public capabilities.

### 1.2 Problem collaboration

Good problem collaboration includes:

- high-quality issue reports
- reproducible cases
- environment details
- command traces
- relevant logs
- narrowing hints
- fix verification
- regression feedback

This is the main path for turning user problems into clearer, actionable inputs.

## 2. What is currently in scope

In scope:

- skills under `skills/`
- public command surface under `commands/` when clearly justified
- examples under `examples/`
- docs under `docs/`
- diagnose patterns
- consistency and helper scripts under `tools/`

## 3. What is not the main public contribution surface

The following are currently not the main public contribution surface unless specifically requested by maintainers:

- deep host-specific runtime behavior outside this repo
- broad generic coding-agent features
- unrelated large-scale orchestration redesign
- large-scope changes without a clear training workflow use case

If you want to propose one of these, open a proposal issue first.

## 4. Repository structure

- `skills/` — reusable domain skills
- `commands/` — intentionally small public slash command surface
- `docs/` — concepts, contracts, architecture notes
- `examples/` — examples and demos
- `tests/contract/` — cross-skill contract tests
- `tools/` — consistency and helper scripts

## 5. Adding a new skill

When adding a new skill:

- create `skills/<skill-name>/SKILL.md`
- add examples if applicable
- update docs if behavior or scope needs explanation
- update `README.md` if it changes the public capability map
- update `AGENTS.md`, `CLAUDE.md`, or `gemini-extension.json` if integration behavior changes
- run consistency checks

Example:

```bash
python tools/check_consistency.py
```

Optional local setup:

```bash
python tools/install_git_hooks.py
make hooks
```

## 6. Style expectations

Please keep contributions:

- focused
- scenario-driven
- easy to review
- aligned with AI infra and model training workflows
- clear about scope and limitations

Avoid vague contributions that do not explain:

- the target scenario
- the expected inputs and outputs
- how the behavior should be validated

## 7. Issues

For user problems, please include as much of the following as possible:

- environment details
- reproduction steps
- command lines
- relevant logs
- expected behavior
- actual behavior
- impact summary

Use issue templates when available.

## 8. Pull requests

A good PR should explain:

- what problem it solves
- why the change belongs in this repo
- what files were changed
- how it was validated
- whether `README`/docs/integration files also need updates

## 9. Contribution philosophy

We prefer contributions that:

- make public capabilities clearer
- improve issue reproducibility
- turn repeated experience into reusable skills
- keep the public surface understandable
- help real training workflows move forward

Thanks for helping make model training workflows easier to use and easier to debug.
