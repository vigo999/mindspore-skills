from pathlib import Path
import json
import subprocess
import sys


SKILL_ROOT = Path(__file__).resolve().parents[1]
CANN_ERROR_DB = SKILL_ROOT / "reference" / "index" / "cann_error_index.db"
CANN_ACLNN_DB = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.db"
QUERY_SCRIPT = SKILL_ROOT / "scripts" / "query_cann_index.py"


def query(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_query_meta_returns_db_kind_for_error_db():
    payload = query("--db", str(CANN_ERROR_DB), "meta")
    assert payload["ok"] is True
    assert payload["db_kind"] == "error_index"
    assert payload["meta"]["entry_count"] >= 1


def test_query_meta_returns_db_kind_for_aclnn_db():
    payload = query("--db", str(CANN_ACLNN_DB), "meta")
    assert payload["ok"] is True
    assert payload["db_kind"] == "aclnn_index"
    assert payload["meta"]["api_count"] >= 1


def test_error_code_lookup_returns_stable_record():
    payload = query("--db", str(CANN_ERROR_DB), "code", "--value", "561003")
    assert payload["ok"] is True
    assert payload["entry"]["code"] == 561003


def test_error_search_returns_related_records():
    payload = query("--db", str(CANN_ERROR_DB), "search", "--text", "aclnn")
    assert payload["ok"] is True
    assert payload["results"]


def test_aclnn_api_lookup_returns_complete_record():
    payload = query("--db", str(CANN_ACLNN_DB), "api", "--name", "aclnnAbs")
    assert payload["ok"] is True
    assert payload["api"]["workspace_api"] == "aclnnAbsGetWorkspaceSize"
    assert payload["api"]["execute_api"] == "aclnnAbs"


def test_aclnn_resolve_understands_workspace_symbol():
    payload = query("--db", str(CANN_ACLNN_DB), "resolve", "--symbol", "aclnnAbsGetWorkspaceSize")
    assert payload["ok"] is True
    assert payload["api"]["api_name"] == "aclnnAbs"
    assert payload["resolved_symbol"] == "aclnnAbsGetWorkspaceSize"


def test_command_db_mismatch_returns_explicit_error():
    payload = query("--db", str(CANN_ERROR_DB), "api", "--name", "aclnnAbs")
    assert payload["ok"] is False
    assert payload["error"] == "command/db mismatch"
    payload = query("--db", str(CANN_ACLNN_DB), "code", "--value", "561003")
    assert payload["ok"] is False
    assert payload["error"] == "command/db mismatch"


def test_query_script_reports_missing_index():
    missing_db = SKILL_ROOT / "reference" / "index" / "missing_cann.db"
    payload = query("--db", str(missing_db), "meta")
    assert payload["ok"] is False
    assert payload["error"] == "index unavailable"
