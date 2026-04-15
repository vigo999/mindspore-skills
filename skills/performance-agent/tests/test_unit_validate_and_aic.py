"""Unit tests for validate_profiler_data.py and summarize_aic_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def run_validate(*args: str) -> subprocess.CompletedProcess[str]:
    """Run validate_profiler_data.py without check=True since it returns 1 for critical."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_profiler_data.py"), *args],
        check=False, text=True, capture_output=True,
    )


def _write_good_profiler(root: Path) -> Path:
    """Write a profiler export that passes stop_check (valid profiler_info.json)."""
    profiler_root = root / "worker_0_20260325_ascend_ms"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)

    (profiler_root / "profiler_info.json").write_text(
        json.dumps({"chip_name": "910b1", "status": "completed"}), encoding="utf-8"
    )
    (profiler_root / "profiler_metadata.json").write_text("{}", encoding="utf-8")
    (profiler_root / "profiler_info_0.json").write_text(
        json.dumps({"chip_name": "910b1", "status": "completed"}), encoding="utf-8"
    )

    (ascend / "step_trace_time.csv").write_text(
        "Step ID,StepTime(ms)\n1,86\n2,83\n3,89\n", encoding="utf-8"
    )
    (ascend / "kernel_details.csv").write_text(
        "Kernel Name,Duration(ms)\nAllReduce,44\n", encoding="utf-8"
    )
    (ascend / "trace_view.json").write_text(
        json.dumps({"events": []}), encoding="utf-8"
    )
    return profiler_root


# -- validate_profiler_data.py --


def test_validate_good_framework_profiler(tmp_path: Path):
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["quality_level"] in ("excellent", "good", "fair")
    assert result["stop_check"] is True
    assert result["data_type"] is not None


def test_validate_nonexistent_path_returns_1(tmp_path: Path):
    output_json = tmp_path / "validation.json"
    result = run_validate("--trace-root", str(tmp_path / "nonexistent"), "--output-json", str(output_json))
    assert result.returncode == 1


def test_validate_empty_directory_reports_issues(tmp_path: Path):
    empty_dir = tmp_path / "empty_profiler"
    empty_dir.mkdir()
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(empty_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["quality_level"] in ("critical", "poor")
    assert len(result["issues"]) > 0


def test_validate_reports_deliverables(tmp_path: Path):
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    deliverables = result["key_deliverables"]
    assert isinstance(deliverables, dict)
    assert "step_trace_time.csv" in deliverables
    assert deliverables["step_trace_time.csv"] is True


# -- summarize_aic_metrics.py --


def test_aic_no_data_returns_gracefully(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is False


def test_aic_with_percentage_values(tmp_path: Path):
    profiler_root = tmp_path / "prof_with_aic"
    device_data = profiler_root / "PROF_0" / "device_0" / "data"
    device_data.mkdir(parents=True)

    (device_data / "aic_metrics_0.csv").write_text(
        "Op Name,Cube Utilization,Duration(us),L2 Hit Rate\n"
        "MatMul_0,5.2%,1000,30.5%\n"
        "MatMul_1,45.0%,500,60.0%\n"
        "MatMul_2,75.0%,200,85.0%\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is True
    assert result["analyzed_operators"] == 3
    assert result["severity_summary"]["critical"] >= 1
    top = result["top_bottlenecks"][0]
    assert top["severity"] == "critical"
    assert top["cube_utilization"] < 1.0


def test_aic_with_fraction_values(tmp_path: Path):
    profiler_root = tmp_path / "prof_frac"
    device_data = profiler_root / "PROF_0" / "device_0" / "data"
    device_data.mkdir(parents=True)

    (device_data / "aic_metrics_0.csv").write_text(
        "Op Name,cube_utilization,l2_hit_rate\n"
        "MatMul,0.85,0.90\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is True
    top = result["top_bottlenecks"][0]
    assert top["cube_utilization"] == 0.85
    assert top["l2_hit_rate"] == 0.90
