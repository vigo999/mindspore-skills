#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "skills"
COMMANDS_DIR = ROOT / "commands"
README = ROOT / "README.md"
AGENTS = ROOT / "AGENTS.md"
GEMINI = ROOT / "gemini-extension.json"

# Public slash commands are a small curated surface and do not need matching
# skill directories.
ROUTER_COMMANDS = {
    "diagnose",
    "fix",
    "migrate",
}


def load_skills():
    skills = set()
    for path in SKILLS_DIR.iterdir():
        if path.is_dir() and (path / "SKILL.md").exists():
            skills.add(path.name)
    return skills


def load_commands():
    return {p.stem for p in COMMANDS_DIR.glob("*.md")}


def parse_agents_skills():
    skills = set()
    if not AGENTS.exists():
        return skills
    for line in AGENTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "|" in line:
            cols = [c.strip() for c in line.strip("|").split("|")]
            if not cols or not cols[0]:
                continue
            if cols[0] == "Skill":
                continue
            if set(cols[0]) <= {"-"}:
                continue
            skills.add(cols[0])
    return skills


def parse_readme_skills():
    skills = set()
    if not README.exists():
        return skills
    pattern = re.compile(r"^\|\s*`([^`]+)`\s*\|")
    for line in README.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            token = match.group(1)
            if not token.startswith("/"):
                skills.add(token)
    return skills


def parse_readme_commands():
    commands = set()
    if not README.exists():
        return commands
    pattern = re.compile(r"^\|\s*`/([^`]+)`\s*\|")
    for line in README.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            commands.add(match.group(1))
    return commands


def parse_gemini_skills():
    skills = set()
    if not GEMINI.exists():
        return skills
    data = json.loads(GEMINI.read_text(encoding="utf-8"))
    for item in data.get("skills", []):
        name = item.get("name")
        if name:
            skills.add(name)
    return skills


def main():
    skills = load_skills()
    commands = load_commands()
    agents_skills = parse_agents_skills()
    readme_skills = parse_readme_skills()
    readme_commands = parse_readme_commands()
    gemini_skills = parse_gemini_skills()

    issues = []

    extra_commands = sorted((commands - skills) - ROUTER_COMMANDS)
    if extra_commands:
        issues.append(("commands_without_skills", extra_commands))

    missing_agents = sorted(skills - agents_skills)
    if missing_agents:
        issues.append(("skills_missing_in_agents", missing_agents))

    extra_agents = sorted(agents_skills - skills)
    if extra_agents:
        issues.append(("agents_extra_skills", extra_agents))

    missing_readme_skills = sorted(skills - readme_skills)
    if missing_readme_skills:
        issues.append(("skills_missing_in_readme", missing_readme_skills))

    extra_readme_skills = sorted(readme_skills - skills)
    if extra_readme_skills:
        issues.append(("readme_extra_skills", extra_readme_skills))

    missing_readme_commands = sorted(commands - readme_commands)
    if missing_readme_commands:
        issues.append(("commands_missing_in_readme", missing_readme_commands))

    extra_readme_commands = sorted(readme_commands - commands)
    if extra_readme_commands:
        issues.append(("readme_extra_commands", extra_readme_commands))

    missing_gemini = sorted(skills - gemini_skills)
    if missing_gemini:
        issues.append(("skills_missing_in_gemini", missing_gemini))

    extra_gemini = sorted(gemini_skills - skills)
    if extra_gemini:
        issues.append(("gemini_extra_skills", extra_gemini))

    if issues:
        print("Consistency issues found:")
        for key, values in issues:
            print(f"- {key}: {', '.join(values)}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
