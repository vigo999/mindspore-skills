import json
import os
import subprocess
from pathlib import Path

import pytest


def test_report_json_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "skills" / "_shared" / "contract" / "report.schema.json"
    schema = json.loads(schema_path.read_text())

    example_root = repo_root / "examples" / "cpu" / "plugin_add"
    run_id = "ci_contract_report_schema"
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    subprocess.run(["bash", str(example_root / "run.sh")], check=True, env=env)

    report_path = example_root / "runs" / run_id / "out" / "report.json"
    report = json.loads(report_path.read_text())
    jsonschema.validate(instance=report, schema=schema)
