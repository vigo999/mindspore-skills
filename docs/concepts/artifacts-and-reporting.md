# Artifacts and Reporting Specification

Version: 1.0.0
Status: Active
Last Updated: 2026-03-02

## 1. Purpose

This document defines the runtime output contract for all skills.
Every skill run must produce a consistent, machine-readable result plus a
human-readable summary.

## 2. Output Location and Directory Contract

Recommended run directory:

`runs/<run_id>/out/`

Required structure:

```text
out/
  report.json
  report.md
  logs/
    run.log
    build.log        # required if build step exists
    verify.log       # required if verify step exists
  artifacts/
    ...              # generated files if any
    README.md        # required only when no artifact is produced
  meta/
    env.json
    inputs.json
```

Rules:
- Paths in `report.json.logs` and `report.json.artifacts` must be relative to `out/`.
- Files referenced in `logs` and `artifacts` arrays must exist.
- `runs/` output must not be committed to git.

## 3. Single Source of Truth (SSOT)

- `report.json` is the primary source of truth.
- `report.md` is a readable projection of `report.json`.
- `report.md` must not conflict with `report.json`.

## 4. report.json Contract

`report.json` must validate against:

`skills/_shared/contract/report.schema.json`

Required fields:
- `schema_version`
- `skill`
- `run_id`
- `status`
- `start_time`
- `end_time`
- `duration_sec`
- `steps`
- `logs`
- `artifacts`
- `env_ref`
- `inputs_ref`

Status values:
- `success`
- `failed`
- `partial`

Error model:
- `error` is optional for success/partial, required for failed.
- Structure:
  - `code`: stable error code (`E_ENV`, `E_BUILD`, `E_VERIFY`, `E_INTERNAL`)
  - `message`: short human-readable summary
  - `details`: optional object for diagnostics

## 5. report.md Contract

Required sections (fixed order):
1. Summary
2. What
3. How
4. Verify
5. Artifacts
6. Environment
7. Logs
8. Next

## 6. meta Files Contract

`meta/env.json` minimum keys:
- `mindspore_version`
- `cann_version`
- `driver_version`
- `python_version`
- `platform`
- `git_commit`

`meta/inputs.json` minimum keys:
- `skill`
- `run_id`
- `parameters`

Security and privacy:
- `inputs.json` must mask secrets and tokens.

## 7. Exit Code Mapping

- `status=success` -> process exit code `0`
- `status=partial` -> process exit code `0` by default (can be elevated by CI policy)
- `status=failed` -> process exit code non-zero

## 8. Cross-Contract Alignment Rule

`report.json.skill` must equal `skill.yaml.name`.

## 9. Schema Versioning Policy

- Use semantic versioning for `schema_version`.
- Minor/patch updates must stay backward-compatible.
- Major updates may break compatibility and must be documented.

## 10. Minimal Example

```json
{
  "schema_version": "1.0.0",
  "skill": "operator-agent",
  "run_id": "20260324_131500_op_build",
  "status": "success",
  "start_time": "2026-03-24T13:15:00Z",
  "end_time": "2026-03-24T13:17:00Z",
  "duration_sec": 120,
  "steps": [
    {"name": "build", "status": "success"},
    {"name": "verify", "status": "success"}
  ],
  "logs": ["logs/run.log", "logs/build.log", "logs/verify.log"],
  "artifacts": ["artifacts/op_build.so"],
  "env_ref": "meta/env.json",
  "inputs_ref": "meta/inputs.json"
}
```
