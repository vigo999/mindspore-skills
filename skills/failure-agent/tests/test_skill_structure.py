from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = SKILL_ROOT / "SKILL.md"
SHOWCASE = SKILL_ROOT / "reference" / "failure-showcase.md"
PTA_DIAGNOSIS = SKILL_ROOT / "reference" / "pta-diagnosis.md"
MS_DIAGNOSIS = SKILL_ROOT / "reference" / "mindspore-diagnosis.md"
MS_API = SKILL_ROOT / "reference" / "mindspore-api-reference.md"
CANN_API = SKILL_ROOT / "reference" / "cann-api-reference.md"
MINT_DB = SKILL_ROOT / "reference" / "index" / "mint_api_index.db"
CANN_ERROR_DB = SKILL_ROOT / "reference" / "index" / "cann_error_index.db"
CANN_ACLNN_DB = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.db"
LEGACY_MS_DIAGNOSIS_NAME = "mindspore" + "-dianosis.md"
LEGACY_MS_DIAGNOSIS = SKILL_ROOT / "reference" / LEGACY_MS_DIAGNOSIS_NAME


def test_skill_markers_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Workflow" in text
    assert "## Stage 1. Failure Analyzer" in text
    assert "## Stage 2. Root-Cause Validator" in text
    assert "## Stage 3. Snapshot Builder" in text
    assert "## Stage 4. Report Builder" in text


def test_reference_and_script_files_exist():
    assert (SKILL_ROOT / "reference" / "failure-taxonomy.md").exists()
    assert (SKILL_ROOT / "reference" / "root-cause-validation.md").exists()
    assert (SKILL_ROOT / "reference" / "index").exists()
    assert CANN_ERROR_DB.exists()
    assert CANN_ACLNN_DB.exists()
    assert MINT_DB.exists()
    assert (SKILL_ROOT / "reference" / "index" / "mint_api_methodology.md").exists()
    assert (SKILL_ROOT / "scripts" / "collect_failure_context.py").exists()
    assert (SKILL_ROOT / "scripts" / "query_cann_index.py").exists()
    assert (SKILL_ROOT / "scripts" / "summarize_traceback.py").exists()
    assert (SKILL_ROOT / "scripts" / "query_mint_api_index.py").exists()
    assert (SKILL_ROOT / "scripts" / "index_builders" / "generate_cann_failure_index.py").exists()
    assert (SKILL_ROOT / "scripts" / "index_builders" / "generate_mindspore_failure_index.py").exists()
    assert not (SKILL_ROOT / "scripts" / "index_builders" / ".tmp").exists()
    assert not LEGACY_MS_DIAGNOSIS.exists()


def test_showcase_contains_curated_pta_and_mindspore_patterns():
    text = SHOWCASE.read_text(encoding="utf-8")
    assert "pta-aclnn-interface-missing-or-capability-floor" in text
    assert "pta-mixed-cpu-npu-device-placement" in text
    assert "ms-graph-vs-pynative-divergence" in text
    assert "ms-backward-only-bprop-failure" in text
    assert "operator-unsupported-misread-instead-of-preconditions" in text


def test_pta_reference_contains_route_and_index_guidance():
    text = PTA_DIAGNOSIS.read_text(encoding="utf-8")
    assert "version compatibility" in text
    assert "device placement" in text
    assert "ASCEND_LAUNCH_BLOCKING=1" in text
    assert "dispatcher or registration before generic runtime" in text
    assert "reference/index/cann_error_index.db" in text
    assert "reference/index/cann_aclnn_api_index.db" in text
    assert "scripts/query_cann_index.py" in text


def test_mindspore_reference_contains_layered_routing_and_index_guidance():
    text = MS_DIAGNOSIS.read_text(encoding="utf-8")
    assert "Platform" in text
    assert "Scripts" in text
    assert "Framework" in text
    assert "Backend" in text
    assert "Graph vs PyNative" in text
    assert "backward or `bprop`" in text
    assert "reference/index/mint_api_index.db" in text
    assert "scripts/query_mint_api_index.py" in text
    assert "reference/index/mint_api_methodology.md" in text
    assert "prefer the general MindSpore route first" in text
    assert "rebuild" in text
    assert "generate_mindspore_failure_index.py" in text


def test_api_references_require_db_index_first():
    ms_text = MS_API.read_text(encoding="utf-8")
    cann_text = CANN_API.read_text(encoding="utf-8")
    assert "mindspore.mint" in ms_text
    assert "this index is usually not the first thing to read" in ms_text
    assert "reference/index/mint_api_index.db" in ms_text
    assert "scripts/query_mint_api_index.py" in ms_text
    assert "skip the mint index query" in ms_text
    assert "regenerate a fresh" in ms_text
    assert "generate_mindspore_failure_index.py" in ms_text
    assert "structured SQLite indexes are the primary runtime inputs" in cann_text
    assert "reference/index/cann_error_index.db" in cann_text
    assert "reference/index/cann_aclnn_api_index.db" in cann_text
    assert "scripts/query_cann_index.py" in cann_text


def test_no_source_files_reference_legacy_mindspore_dianosis_name():
    for path in (
        SKILL_MD,
        SHOWCASE,
        PTA_DIAGNOSIS,
        MS_DIAGNOSIS,
        MS_API,
        CANN_API,
    ):
        assert LEGACY_MS_DIAGNOSIS_NAME not in path.read_text(encoding="utf-8")
