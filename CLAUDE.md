# MindSpore Skills Repository

This repository contains agent skills for MindSpore development.

## Repository Structure

- `skills/*/SKILL.md` - Individual skill definitions
- `.claude-plugin/marketplace.json` - Marketplace registry for Claude Code
- `AGENTS.md` - Universal agent instructions for Codex/other tools

## Validation

When adding or modifying skills, ensure:

1. SKILL.md has valid YAML frontmatter with `name` and `description`
2. Skill name in SKILL.md matches the directory name
3. `.claude-plugin/plugin.json` references the skills directory correctly
4. marketplace.json is updated with new skills
5. AGENTS.md includes the skill in the table

## Testing Skills

Test locally before committing:

```bash
# Claude Code
/plugin install ./skills/cpu-plugin-builder

# Verify activation
/cpu-plugin-builder "describe the workflow"
```
