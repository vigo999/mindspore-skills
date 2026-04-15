#!/usr/bin/env python3
"""Validate profiler data integrity before performance analysis.

Checks collection status, parse status, and key deliverable presence.
Returns exit code: 0=valid, 1=invalid(needs recollection), 2=unparsed(needs parsing).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import write_json


def detect_data_type(profiler_path: Path, _depth: int = 0) -> Optional[str]:
    """Detect profiler data type from the top-level path."""
    if _depth > 3:
        return None
    name = profiler_path.name.lower()
    if name.endswith("_ascend_pt") or "_ascend_pt_" in name:
        return "framework_profiler_pt"
    if name.endswith("_ascend_ms") or "_ascend_ms_" in name:
        return "framework_profiler_ms"
    # Check for PROF_{} directory pattern (msprof)
    if profiler_path.name.startswith("prof_"):
        return "msprof"
    if (profiler_path / "ASCEND_PROFILER_OUTPUT").exists():
        # Framework profiler output detected by content
        subdirs = [p for p in profiler_path.iterdir() if p.is_dir()]
        for sub in subdirs:
            if sub.name.lower().endswith("_ascend_pt"):
                return "framework_profiler_pt"
            if sub.name.lower().endswith("_ascend_ms"):
                return "framework_profiler_ms"
        # If ASCEND_PROFILER_OUTPUT exists at root, treat as framework profiler
        return "framework_profiler"
    if any(profiler_path.glob("PROF_*")):
        return "msprof"
    # Check if it's a cluster directory containing multiple profiler dirs
    subdirs = [p for p in profiler_path.iterdir() if p.is_dir()]
    types_found = set()
    for sub in subdirs:
        sub_type = detect_data_type(sub, _depth + 1)
        if sub_type:
            types_found.add(sub_type)
    if types_found:
        return f"cluster_{types_found.pop()}"
    return None


def check_stop_framework(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if framework profiler collection completed normally."""
    issues = []
    # Look for profiler_info.json or profiler_info_{Rank_ID}.json
    info_files = list(profiler_path.glob("profiler_info.json"))
    info_files.extend(profiler_path.glob("profiler_info_*.json"))
    if not info_files:
        issues.append("profiler_info.json not found; collection may not have completed normally")
        return False, issues
    # Verify it has content
    for info_file in info_files[:1]:
        try:
            data = json.loads(info_file.read_text(encoding="utf-8"))
            if not data:
                issues.append(f"{info_file.name} is empty")
                return False, issues
        except Exception as e:
            issues.append(f"Failed to read {info_file.name}: {e}")
            return False, issues
    return True, issues


def check_stop_msprof(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if msprof collection completed normally."""
    issues = []
    # Look for end_info files under device subdirectories
    prof_dirs = [profiler_path]
    if not profiler_path.name.startswith("PROF_"):
        prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]
        if not prof_dirs:
            issues.append("No PROF_* directory found")
            return False, issues

    failed_dirs: list[str] = []
    for prof_dir in prof_dirs:
        device_dirs = [p for p in prof_dir.iterdir() if p.is_dir() and p.name.startswith("device_")]
        if not device_dirs:
            failed_dirs.append(f"No device_* directory in {prof_dir.name}")
            continue
        # Check for end_info in first device
        device_dir = device_dirs[0]
        end_info_files = list(device_dir.glob("end_info.*"))
        if not end_info_files:
            failed_dirs.append(f"end_info not found in {device_dir.name}")

    if len(failed_dirs) == len(prof_dirs):
        # All PROF_ directories failed
        issues.extend(failed_dirs)
        return False, issues
    if failed_dirs:
        # Some PROF_ directories failed — warn but proceed
        issues.extend([f"Partial: {msg}" for msg in failed_dirs])
    return True, issues


def check_parse_framework(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if framework profiler data has been parsed."""
    issues = []
    output_dir = profiler_path / "ASCEND_PROFILER_OUTPUT"
    if not output_dir.exists():
        issues.append("ASCEND_PROFILER_OUTPUT directory not found; data may not be parsed")
        return False, issues
    if not any(output_dir.iterdir()):
        issues.append("ASCEND_PROFILER_OUTPUT is empty; data may not be parsed")
        return False, issues
    return True, issues


def check_parse_msprof(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if msprof data has been parsed/exported."""
    issues = []
    prof_dirs = [profiler_path]
    if not profiler_path.name.startswith("PROF_"):
        prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]

    for prof_dir in prof_dirs[:1]:
        ms_output = prof_dir / "mindstudio_profiler_output"
        if not ms_output.exists():
            issues.append(f"mindstudio_profiler_output not found in {prof_dir.name}; run 'msprof --export=on'")
            return False, issues
        if not any(ms_output.iterdir()):
            issues.append(f"mindstudio_profiler_output is empty in {prof_dir.name}")
            return False, issues
    return True, issues


def check_key_deliverables(profiler_path: Path, data_type: str) -> dict[str, bool]:
    """Check presence of key analysis files."""
    files = {}

    if data_type.startswith("framework_profiler"):
        base = profiler_path / "ASCEND_PROFILER_OUTPUT"
        files["step_trace_time.csv"] = (base / "step_trace_time.csv").exists()
        files["kernel_details.csv"] = (base / "kernel_details.csv").exists()
        files["trace_view.json"] = (base / "trace_view.json").exists()
        files["communication.json"] = (base / "communication.json").exists()
        files["memory_record.csv"] = (base / "memory_record.csv").exists()
        files["operator_memory.csv"] = (base / "operator_memory.csv").exists()
        files["dataset.csv"] = (base / "dataset.csv").exists()
        # Check for DB files
        db_files = list(profiler_path.glob("*_profiler_*.db"))
        files["profiler_db"] = len(db_files) > 0
    elif data_type == "msprof" or data_type.startswith("cluster_msprof"):
        prof_dirs = [profiler_path]
        if not profiler_path.name.startswith("PROF_"):
            prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]
        for prof_dir in prof_dirs[:1]:
            ms_output = prof_dir / "mindstudio_profiler_output"
            files["op_summary"] = bool(list(ms_output.glob("op_summary_*.csv")))
            files["task_time"] = bool(list(ms_output.glob("task_time_*.csv")))
            db_files = list(prof_dir.glob("msprof_*.db"))
            files["profiler_db"] = len(db_files) > 0

    return files


def check_aic_metrics(profiler_path: Path) -> bool:
    """Check if AIC PMU metrics data is available."""
    # AIC metrics are typically in PROF_{}/device_{}/data/ or similar locations
    for pattern in ("**/aic_metrics_*", "**/*aic*pmu*", "**/*pmu*"):
        if list(profiler_path.rglob(pattern)):
            return True
    return False


def determine_quality_level(
    stop_ok: bool,
    parse_ok: bool,
    deliverables: dict[str, bool],
) -> str:
    """Determine overall data quality level."""
    if not stop_ok:
        return "critical"
    if not parse_ok:
        return "poor"
    present = sum(1 for v in deliverables.values() if v)
    critical_files = ["step_trace_time.csv", "kernel_details.csv", "trace_view.json"]
    critical_present = sum(1 for f in critical_files if deliverables.get(f, False))
    if critical_present >= 2 and present >= 3:
        return "excellent"
    if critical_present >= 1 and present >= 2:
        return "good"
    if present >= 1:
        return "fair"
    return "poor"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate profiler data integrity")
    parser.add_argument("--trace-root", required=True, help="Profiler data root directory")
    parser.add_argument("--output-json", required=True, help="Path to write validation report JSON")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()
    if not trace_root.exists():
        print(f"Error: path does not exist: {trace_root}", file=sys.stderr)
        return 1

    # 1. Detect data type
    data_type = detect_data_type(trace_root)

    # 2. Check collection status
    if data_type and data_type.startswith("framework_profiler"):
        stop_ok, stop_issues = check_stop_framework(trace_root)
    elif data_type == "msprof" or (data_type and data_type.startswith("cluster_msprof")):
        stop_ok, stop_issues = check_stop_msprof(trace_root)
    else:
        # Try both approaches
        fw_ok, fw_issues = check_stop_framework(trace_root)
        ms_ok, ms_issues = check_stop_msprof(trace_root)
        if fw_ok:
            stop_ok, stop_issues = True, []
            data_type = data_type or "framework_profiler"
        elif ms_ok:
            stop_ok, stop_issues = ms_ok, ms_issues
            data_type = data_type or "msprof"
        else:
            stop_ok = False
            stop_issues = fw_issues + ms_issues
            data_type = data_type or "unknown"

    # 3. Check parse status
    if data_type and data_type.startswith("framework_profiler"):
        parse_ok, parse_issues = check_parse_framework(trace_root)
    elif data_type in ("msprof",) or (data_type and data_type.startswith("cluster_msprof")):
        parse_ok, parse_issues = check_parse_msprof(trace_root)
    else:
        parse_ok, parse_issues = False, ["Unknown data type, cannot check parse status"]

    # 4. Check key deliverables
    deliverables = check_key_deliverables(trace_root, data_type or "unknown")

    # 5. Check AIC metrics (optional)
    aic_available = check_aic_metrics(trace_root)

    # 6. Determine quality level
    quality = determine_quality_level(stop_ok, parse_ok, deliverables)

    # 7. Build issues list
    all_issues = []
    if stop_issues:
        all_issues.extend([{"severity": "critical", "message": msg} for msg in stop_issues])
    if parse_issues:
        all_issues.extend([{"severity": "warning", "message": msg} for msg in parse_issues])
    for name, present in deliverables.items():
        if not present:
            severity = "warning" if name in ("step_trace_time.csv", "kernel_details.csv", "trace_view.json") else "note"
            all_issues.append({"severity": severity, "message": f"{name} not found"})

    # 8. Build report
    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "data_type": data_type,
        "quality_level": quality,
        "stop_check": stop_ok,
        "parse_check": parse_ok,
        "key_deliverables": deliverables,
        "aic_metrics_available": aic_available,
        "issues": all_issues,
        "recommended_action": _recommend(quality),
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({"quality_level": quality, "data_type": data_type, "issues": len(all_issues)}, indent=2))

    if quality == "critical":
        return 1
    if quality == "poor" and not parse_ok:
        return 2
    return 0


def _recommend(quality: str) -> str:
    if quality == "critical":
        return "Data collection did not complete normally. Recollect profiler data before analysis."
    if quality == "poor":
        return "Data is not parsed or key files are missing. Parse the profiler data first."
    if quality == "fair":
        return "Limited data available. Analysis will be constrained. Consider collecting more complete profiling data."
    return "Data quality is sufficient for performance analysis. Proceed with Stage 1."


if __name__ == "__main__":
    raise SystemExit(main())
