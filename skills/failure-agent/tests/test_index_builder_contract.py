from pathlib import Path
import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
CANN_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_cann_failure_index.py"
MINDSPORE_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_mindspore_failure_index.py"
CANN_INDEX = SKILL_ROOT / "reference" / "index" / "cann_error_index.yaml"
CANN_ACLNN_INDEX = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.yaml"
MINT_INDEX = SKILL_ROOT / "reference" / "index" / "mint_api_index.yaml"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def test_cann_builder_defaults_to_minimal_yaml_outputs():
    text = CANN_BUILDER.read_text(encoding="utf-8")
    assert "cann_error_index.yaml" in text
    assert "cann_aclnn_api_index.yaml" in text
    assert "--with-source-docs" in text
    assert "--with-compact" in text
    assert "--deterministic" in text
    assert "aclError.md" in text
    assert "aclnnApiError.md" in text


def test_mindspore_builder_defaults_and_source_modes_are_declared():
    text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    assert "https://atomgit.com/mindspore/mindspore.git" in text
    assert "mint_api_index.yaml" in text
    assert "mint_api_methodology.md" in text
    assert "--repo" in text
    assert "--branch" in text
    assert "--deterministic" in text
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
    for path in (CANN_INDEX, CANN_ACLNN_INDEX, MINT_INDEX):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
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
            assert field in meta


def test_generated_indexes_use_deterministic_timestamp():
    for path in (CANN_INDEX, CANN_ACLNN_INDEX, MINT_INDEX):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["meta"]["generated_at"] == DETERMINISTIC_TIMESTAMP


def test_cann_indexes_preserve_utf8_text():
    error_payload = yaml.safe_load(CANN_INDEX.read_text(encoding="utf-8"))
    aclnn_payload = yaml.safe_load(CANN_ACLNN_INDEX.read_text(encoding="utf-8"))
    assert error_payload["entries"][0]["meaning"] == "执行成功。"
    assert aclnn_payload["apis"][0]["summary"] == "接口功能：为输入张量的每一个元素取绝对值。"
