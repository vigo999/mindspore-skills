from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "references").is_dir()
    assert (ROOT / "scripts").is_dir()


def test_mhc_route_pack_exists():
    assert (ROOT / "references" / "mhc" / "mhc-implementation-pattern.md").exists()
    assert (ROOT / "references" / "mhc" / "mhc-validation-checklist.md").exists()
    assert (ROOT / "references" / "mhc" / "mhc-qwen3-case-study.md").exists()


def test_attnres_route_pack_exists():
    assert (ROOT / "references" / "attnres" / "attnres-implementation-pattern.md").exists()
    assert (ROOT / "references" / "attnres" / "attnres-validation-checklist.md").exists()
    assert (ROOT / "references" / "attnres" / "attnres-qwen3-case-study.md").exists()


def test_phase1_combined_references_and_helper_exist():
    assert (ROOT / "references" / "phase1-intake-and-triage.md").exists()
    assert (ROOT / "references" / "phase1-verification-and-admission.md").exists()
    assert (ROOT / "scripts" / "phase1_artifact_helper.py").exists()
