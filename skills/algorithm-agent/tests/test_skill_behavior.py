from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_workflow_stages_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "1. `feature-analyzer`" in text
    assert "2. `integration-planner`" in text
    assert "3. `patch-builder`" in text
    assert "4. `readiness-handoff-and-report`" in text
