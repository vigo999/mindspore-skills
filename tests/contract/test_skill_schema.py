import json
from pathlib import Path

import pytest


def test_skill_manifest_template_compatible_with_schema():
    yaml = pytest.importorskip("yaml")
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "skills" / "_shared" / "contract" / "skill.schema.json"
    tmpl_path = repo_root / "skills" / "_shared" / "templates" / "skill.yaml.tmpl"

    schema = json.loads(schema_path.read_text())
    content = tmpl_path.read_text()
    content = (
        content.replace("{{ skill_name }}", "readiness-agent")
        .replace("{{ display_name }}", "Readiness Agent")
        .replace("{{ description }}", "Check whether a single-machine training workspace is ready to run.")
    )
    manifest = yaml.safe_load(content)
    jsonschema.validate(instance=manifest, schema=schema)
