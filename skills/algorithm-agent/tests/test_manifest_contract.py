from pathlib import Path


SKILL_YAML = Path(__file__).resolve().parents[1] / "skill.yaml"


def test_manifest_has_expected_name():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'name: "algorithm-agent"' in text
    assert 'display_name: "Algorithm Agent"' in text
