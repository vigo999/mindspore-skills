# Skills Contract Specification

Version: 1.0.0
Status: Active
Last Updated: 2026-03-02

## 1. Purpose

This document defines the contract for each skill package.
It is the canonical human-readable definition for `skill.yaml`.

## 2. Required Files Per Skill

Each `skills/<skill-name>/` folder must contain:
- `SKILL.md`
- `skill.yaml`
- runner entry file referenced by `skill.yaml.entry.path` (`run.py` or `run.sh`, or `manual` mode)

## 3. skill.yaml Schema

`skill.yaml` must validate against:

`skills/_shared/contract/skill.schema.json`

Minimum required fields:
- `schema_version`
- `name`
- `display_name`
- `description`
- `entry`
- `inputs`
- `outputs`
- `permissions`
- `dependencies`

## 4. Field Semantics

- `name`: stable machine id, lowercase kebab-case.
- `display_name`: human-facing name.
- `entry`: execution mode and entry path.
- `inputs`: accepted runtime parameters.
- `outputs`: output contract linkage.
- `permissions`: declared runtime capability needs.
- `dependencies`: required tools and optional python packages.

## 5. Output Contract Linkage

Every skill must declare:
- `outputs.report_schema = skills/_shared/contract/report.schema.json`
- `outputs.out_dir_layout = runs/<run_id>/out/`

Cross-contract alignment rule:
- `report.json.skill` must equal `skill.yaml.name`.

## 6. Ownership Boundary with Artifacts Contract

- `skills-contract.md` owns skill identity, entry, inputs, dependencies, permissions.
- `artifacts-and-reporting.md` owns runtime output (`out/`, `report.json`, `report.md`, `meta/`, `logs/`, `artifacts/`).

Do not duplicate field definitions across both docs.

## 7. Versioning

- `schema_version` in `skill.yaml` follows semantic versioning.
- Backward-compatible updates must not require immediate skill rewrites.
- Breaking changes require a major version bump and migration notes.
