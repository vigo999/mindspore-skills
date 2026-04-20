from pathlib import Path
import importlib.util
import sqlite3
import sys
import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
CANN_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_cann_failure_index.py"
MINDSPORE_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_mindspore_failure_index.py"
CANN_INDEX = SKILL_ROOT / "reference" / "index" / "cann_error_index.yaml"
CANN_ACLNN_INDEX = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.yaml"
CANN_ERROR_DB = SKILL_ROOT / "reference" / "index" / "cann_error_index.db"
CANN_ACLNN_DB = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.db"
MINT_DB = SKILL_ROOT / "reference" / "index" / "mint_api_index.db"
MINT_INDEX = SKILL_ROOT / "reference" / "index" / "mint_api_index.yaml"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def load_mint_builder_module():
    spec = importlib.util.spec_from_file_location("mint_index_builder", MINDSPORE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cann_builder_defaults_to_minimal_yaml_outputs():
    text = CANN_BUILDER.read_text(encoding="utf-8")
    assert "cann_error_index.db" in text
    assert "cann_aclnn_api_index.db" in text
    assert "--no-tags" in text
    assert "--with-error-yaml" in text
    assert "--with-aclnn-yaml" in text
    assert "--with-source-docs" in text
    assert "--with-compact" in text
    assert "--deterministic" in text
    assert "aclError.md" in text
    assert "aclnnApiError.md" in text


def test_mindspore_builder_defaults_and_source_modes_are_declared():
    text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    assert "https://atomgit.com/mindspore/mindspore.git" in text
    assert "mint_api_index.db" in text
    assert "mint_api_index.yaml" in text
    assert "--depth" in text
    assert "--no-tags" in text
    assert "--repo" in text
    assert "--branch" in text
    assert "--deterministic" in text
    assert "--with-yaml" in text
    assert "--with-evidence" in text
    assert "--with-review" in text
    assert "--with-rulebook" in text


def test_builders_emit_provenance_fields():
    cann_text = CANN_BUILDER.read_text(encoding="utf-8")
    mint_text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    for field in (
        "generated_at",
        "generator_name",
        "generator_version",
        "index_schema_version",
        "source_mode",
        "source_repo_url",
        "source_branch",
        "source_commit",
        "source_repository_count",
        "source_repositories",
    ):
        assert field in cann_text
        assert field in mint_text


def test_builders_emit_cleanup_and_workspace_controls():
    cann_text = CANN_BUILDER.read_text(encoding="utf-8")
    mint_text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    for marker in ("--keep-workspace", "--deterministic", ".tmp"):
        assert marker in cann_text
        assert marker in mint_text
    assert "run_workspace_root" in cann_text
    assert "prune_empty_parents" in cann_text
    assert "safe_rmtree(workspace_root)" in cann_text
    assert "run_workspace_root" in mint_text
    assert "prune_empty_parents" in mint_text
    assert "safe_rmtree(workspace_root)" in mint_text


def test_generated_indexes_include_provenance_meta():
    for path, count_field in ((CANN_ERROR_DB, "entry_count"), (CANN_ACLNN_DB, "api_count")):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            meta = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
            assert meta is not None
            for field in (
                "generated_at",
                "generator_name",
                "generator_version",
                "schema_version",
                "source_mode",
                "source_repo_url",
                "source_branch",
                "source_commit",
                "source_repository_count",
                count_field,
            ):
                assert field in meta.keys()
        finally:
            conn.close()
    conn = sqlite3.connect(MINT_DB)
    try:
        conn.row_factory = sqlite3.Row
        meta = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
        assert meta is not None
        for field in (
            "generated_at",
            "generator_name",
            "generator_version",
            "schema_version",
            "source_mode",
            "source_repo_url",
            "source_branch",
            "source_commit",
            "mindspore_version_hint",
        ):
            assert field in meta.keys()
    finally:
        conn.close()


def test_generated_mint_db_schema_is_slimmed():
    conn = sqlite3.connect(MINT_DB)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api)").fetchall()
        }
        assert "summary" in columns
        assert "category" in columns
        assert "trust_level" in columns
        assert "grad_differentiable" in columns
        for removed in (
            "semantic_kind",
            "graph_support_kind",
            "primitive_count",
            "evidence_ref",
            "status_summary",
            "support_summary",
            "llm_summary",
            "llm_warning",
            "confidence",
        ):
            assert removed not in columns
        grad_primitive_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_grad_primitive)").fetchall()
        }
        assert grad_primitive_columns == {"api_id", "ordinal", "primitive_name", "origin_kind"}
        api_path_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_path)").fetchall()
        }
        assert "anchor" in api_path_columns
        path_kinds = {
            row[0]
            for row in conn.execute("SELECT DISTINCT path_kind FROM api_path").fetchall()
        }
        assert "primary_path" not in path_kinds
        assert {
            "api_def_path",
            "dispatch_path",
            "implementation_path",
            "op_def_path",
            "kernel_path_pynative_ascend",
            "kernel_path_pynative_cpu",
            "kernel_path_pynative_gpu",
            "kernel_path_graph_kbk_o0_ascend",
            "kernel_path_graph_kbk_o0_cpu",
            "kernel_path_graph_kbk_o0_gpu",
            "infer_path",
        }.issubset(path_kinds)
    finally:
        conn.close()


def test_generated_indexes_use_deterministic_timestamp():
    for path in (CANN_ERROR_DB, CANN_ACLNN_DB):
        conn = sqlite3.connect(path)
        try:
            value = conn.execute("SELECT generated_at FROM schema_meta WHERE id = 1").fetchone()
            assert value is not None
            assert value[0] == DETERMINISTIC_TIMESTAMP
        finally:
            conn.close()
    conn = sqlite3.connect(MINT_DB)
    try:
        value = conn.execute("SELECT generated_at FROM schema_meta WHERE id = 1").fetchone()
        assert value is not None
        assert value[0] == DETERMINISTIC_TIMESTAMP
    finally:
        conn.close()


def load_cann_builder_module():
    spec = importlib.util.spec_from_file_location("cann_index_builder", CANN_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optional_cann_yaml_matches_db_snapshot():
    module = load_cann_builder_module()
    error_payload_from_db = module.load_error_index_payload_from_db(CANN_ERROR_DB)
    error_payload_from_yaml = yaml.safe_load(CANN_INDEX.read_text(encoding="utf-8"))
    assert error_payload_from_db == error_payload_from_yaml
    aclnn_payload_from_db = module.load_aclnn_index_payload_from_db(CANN_ACLNN_DB)
    aclnn_payload_from_yaml = yaml.safe_load(CANN_ACLNN_INDEX.read_text(encoding="utf-8"))
    assert aclnn_payload_from_db == aclnn_payload_from_yaml


def test_optional_mint_yaml_matches_db_snapshot():
    module = load_mint_builder_module()
    payload_from_db = module.canonicalize_for_yaml(module.load_main_payload_from_db(MINT_DB))
    payload_from_yaml = yaml.safe_load(MINT_INDEX.read_text(encoding="utf-8"))
    assert payload_from_db == payload_from_yaml


def test_cann_indexes_preserve_utf8_text():
    error_payload = yaml.safe_load(CANN_INDEX.read_text(encoding="utf-8"))
    aclnn_payload = yaml.safe_load(CANN_ACLNN_INDEX.read_text(encoding="utf-8"))
    assert error_payload["entries"][0]["meaning"] == "执行成功。"
    assert aclnn_payload["apis"][0]["summary"] == "接口功能：为输入张量的每一个元素取绝对值。"
