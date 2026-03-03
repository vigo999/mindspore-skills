import json
import os
import subprocess
from pathlib import Path


def test_example_output_layout():
    repo_root = Path(__file__).resolve().parents[2]
    example_root = repo_root / "examples" / "cpu" / "plugin_add"
    run_id = "ci_contract_layout"

    env = os.environ.copy()
    env["RUN_ID"] = run_id
    subprocess.run(["bash", str(example_root / "run.sh")], check=True, env=env)

    out_dir = example_root / "runs" / run_id / "out"
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "logs" / "run.log").exists()
    assert (out_dir / "meta" / "env.json").exists()
    assert (out_dir / "meta" / "inputs.json").exists()

    data = json.loads((out_dir / "report.json").read_text())
    for rel in data["logs"] + data["artifacts"]:
        assert (out_dir / rel).exists(), f"Missing path from report.json: {rel}"
