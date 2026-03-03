import json
import subprocess
from pathlib import Path


def test_plugin_add_smoke():
    root = Path(__file__).resolve().parent
    run_id = "test_plugin_add_smoke"
    subprocess.run(
        ["bash", str(root / "run.sh")],
        check=True,
        env={"RUN_ID": run_id},
    )

    out_dir = root / "runs" / run_id / "out"
    assert out_dir.exists()

    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    assert report_json.exists()
    assert report_md.exists()

    data = json.loads(report_json.read_text())
    assert data["schema_version"] == "1.0.0"
    assert data["skill"] == "cpu-plugin-builder"
    assert data["status"] == "success"
    assert data["env_ref"] == "meta/env.json"
    assert data["inputs_ref"] == "meta/inputs.json"

    for rel in data["logs"] + data["artifacts"]:
        assert (out_dir / rel).exists(), f"Missing output path: {rel}"
