# MindSpore Skills Repository

This repository contains agent skills for MindSpore development.

## Repository Structure

- `skills/*/SKILL.md` - Individual skill definitions
- `.claude-plugin/marketplace.json` - Marketplace registry for Claude Code
- `AGENTS.md` - Universal agent instructions for Codex/other tools

## Validation and maintenance

When adding a new skill:
1. Add `skills/<skill-name>/SKILL.md` with matching frontmatter and directory name
2. Add a slash command in `commands/<command-name>.md` only if the skill belongs on the small public command surface (`diagnose`, `fix`, `migrate`, or a future replacement)
3. Ensure SKILL.md has valid YAML frontmatter with `name` and `description`
4. Ensure the skill name in SKILL.md matches the directory name
5. Update `AGENTS.md` (skill table + activation triggers)
6. Update `README.md` (skill list and commands)
7. Update `gemini-extension.json` with name/path/description
8. Ensure `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json` align with the skill list

When modifying an existing skill:
1. Update `skills/<skill-name>/SKILL.md` and any referenced files
2. Refresh `AGENTS.md` triggers if scope/keywords changed
3. Update `README.md` if descriptions or commands changed
4. Update `gemini-extension.json` if name/path/description changed

## Testing Skills

Test locally before committing:

```bash
# Claude Code

# Verify activation
/diagnose "my qwen3 lora run crashes with operator not implemented"
/fix "accuracy dropped after switching to ascend"
/migrate "port this HuggingFace model repo to MindSpore"
```
