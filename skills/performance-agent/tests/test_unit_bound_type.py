"""Unit tests for detect_bound_type.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS

# Allow importlib to resolve `from perf_common import ...`
sys.path.insert(0, str(SCRIPTS))


def _write_step_trace(root: Path, step_times: list[dict]) -> Path:
    """Write a step_trace_time.csv with given rows."""
    csv_path = root / "step_trace_time.csv"
    header = "Step ID,ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)"
    rows = [header]
    for i, st in enumerate(step_times):
        rows.append(
            f"{i},{st.get('compute', 50)},{st.get('comm', 20)},"
            f"{st.get('idle', 10)},{st.get('total', 80)}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def _write_trace_view(root: Path, events: list[dict]) -> Path:
    """Write a trace_view.json with given events."""
    json_path = root / "trace_view.json"
    json_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return json_path


def test_analyze_step_trace_device_bound():
    """Low free time ratio (<10%) → device_bound."""
    script = SCRIPTS / "detect_bound_type.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dbt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"Step ID": "1", "StepTime(ms)": "100", "ComputeTime(ms)": "60",
         "CommunicationTime(ms)": "30", "IdleGap(ms)": "5"},
        {"Step ID": "2", "StepTime(ms)": "95", "ComputeTime(ms)": "55",
         "CommunicationTime(ms)": "32", "IdleGap(ms)": "4"},
    ]
    result = mod.analyze_step_trace_bound(rows)
    assert result["bound_type"] == "device_bound"
    assert result["free_time_ratio"] < 0.10


def test_analyze_step_trace_host_bound():
    """High free time ratio (>=20%) → host_bound strong."""
    script = SCRIPTS / "detect_bound_type.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dbt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"Step ID": "1", "StepTime(ms)": "100", "ComputeTime(ms)": "30",
         "CommunicationTime(ms)": "10", "IdleGap(ms)": "55"},
        {"Step ID": "2", "StepTime(ms)": "100", "ComputeTime(ms)": "28",
         "CommunicationTime(ms)": "12", "IdleGap(ms)": "58"},
    ]
    result = mod.analyze_step_trace_bound(rows)
    assert result["bound_type"] == "host_bound"
    assert result["free_time_ratio"] >= 0.20


def test_analyze_step_trace_moderate_host_bound():
    """Free time ratio between 10% and 20% → host_bound moderate."""
    script = SCRIPTS / "detect_bound_type.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dbt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"Step ID": "1", "StepTime(ms)": "100", "ComputeTime(ms)": "40",
         "CommunicationTime(ms)": "30", "IdleGap(ms)": "15"},
    ]
    result = mod.analyze_step_trace_bound(rows)
    assert result["bound_type"] == "host_bound"
    assert result["severity"] == "moderate"
    assert 0.10 <= result["free_time_ratio"] < 0.20


def test_analyze_trace_view_host_bound():
    """High idle ratio in trace events → host_bound."""
    script = SCRIPTS / "detect_bound_type.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dbt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    trace_data = {
        "events": [
            {"name": "idle_gap", "duration_ms": 80},
            {"name": "compute_kernel", "duration_ms": 40},
            {"name": "host_dispatch", "duration_ms": 30},
        ]
    }
    result = mod.analyze_trace_view_bound(trace_data)
    assert result["bound_type"] == "host_bound"
    assert result["idle_ratio"] > 0.20


def test_analyze_trace_view_device_bound():
    """High compute ratio in trace events → device_bound."""
    script = SCRIPTS / "detect_bound_type.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dbt", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    trace_data = {
        "events": [
            {"name": "compute_matmul", "duration_ms": 70},
            {"name": "forward_pass", "duration_ms": 50},
            {"name": "backward_pass", "duration_ms": 60},
            {"name": "idle_gap", "duration_ms": 5},
        ]
    }
    result = mod.analyze_trace_view_bound(trace_data)
    assert result["bound_type"] == "device_bound"
    assert result["device_ratio"] > result["host_ratio"]


def test_full_pipeline_step_trace_host_bound(tmp_path: Path):
    """Full CLI pipeline: step_trace with high idle gap → host_bound."""
    profiler_root = tmp_path / "host_bound_run"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True)

    (ascend / "step_trace_time.csv").write_text(
        "Step ID,ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)\n"
        + "\n".join(f"{i},30,15,60,105" for i in range(5)) + "\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "bound.json"
    run_script(
        "detect_bound_type.py",
        "--trace-root", str(profiler_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["bound_type"] == "host_bound"
    assert "recommended_actions" in result
    assert any("flame graph" in a.lower() or "cpu" in a.lower()
               for a in result["recommended_actions"])


def test_full_pipeline_trace_view_device_bound(tmp_path: Path):
    """Full CLI pipeline: trace_view with dominant compute → device_bound."""
    profiler_root = tmp_path / "device_bound_run"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True)

    (ascend / "trace_view.json").write_text(
        json.dumps({
            "events": [
                {"name": "compute_matmul", "duration_ms": 80},
                {"name": "forward_compute", "duration_ms": 60},
                {"name": "backward_compute", "duration_ms": 70},
                {"name": "launch_dispatch", "duration_ms": 10},
            ]
        }),
        encoding="utf-8",
    )

    output_json = tmp_path / "bound.json"
    run_script(
        "detect_bound_type.py",
        "--trace-root", str(profiler_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["bound_type"] == "device_bound"
    assert any("hotspot" in a.lower() for a in result["recommended_actions"])


def test_no_input_data_exits_with_error(tmp_path: Path):
    """No step_trace_time.csv or trace_view.json should exit with error."""
    empty_root = tmp_path / "empty_run"
    (empty_root / "ASCEND_PROFILER_OUTPUT").mkdir(parents=True)

    output_json = tmp_path / "bound.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_bound_type.py"),
         "--trace-root", str(empty_root), "--output-json", str(output_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
