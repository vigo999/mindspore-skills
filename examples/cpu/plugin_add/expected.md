# plugin_add Expected Output

## Goal
- Validate the minimal CPU plugin example flow and output contract generation.

## Success Criteria
- `run.sh` exits with code `0`.
- `runs/<run_id>/out/` is generated.
- Required files exist:
  - `report.json`
  - `report.md`
  - `logs/run.log`
  - `logs/build.log`
  - `logs/verify.log`
  - `meta/env.json`
  - `meta/inputs.json`
- `report.json.status = success`
- `report.json.skill = cpu-plugin-builder`

## One-Command Run
```bash
bash examples/cpu/plugin_add/run.sh
```

## Smoke Test
```bash
python -m pytest -q examples/cpu/plugin_add/test_smoke.py
```
