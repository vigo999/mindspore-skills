# MindSpore Skills

MindSpore development skills for AI coding agents. Build CPU/GPU/NPU operators and migrate models with guided workflows.

Compatible with **Claude Code**, **OpenCode**, **Gemini CLI**, and **Codex**.

## Installation

### Claude Code

Register the marketplace and install:

```
/plugin marketplace add vigo999/mindspore-skills
/plugin install cpu-plugin-builder@mindspore-skills
```

Or install all skills:

```
/plugin install mindspore-skills
```

Then use slash commands:

```
/ms-api-builder
/ms-cpu-plugin-builder
```

### OpenCode

Clone to your global config:

```bash
git clone https://github.com/vigo999/mindspore-skills.git ~/.config/opencode/mindspore-skills
ln -s ~/.config/opencode/mindspore-skills/skills ~/.config/opencode/skills
ln -s ~/.config/opencode/mindspore-skills/commands ~/.config/opencode/commands
```

Or for a specific project:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .opencode
```

Then in OpenCode:

```
/ms-api-builder
```

See [OpenCode Skills docs](https://opencode.ai/docs/skills) for more details.

### Gemini CLI

Install as an extension:

```bash
gemini extensions install https://github.com/vigo999/mindspore-skills.git --consent
```

Or from local clone:

```bash
git clone https://github.com/vigo999/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more details.

### Codex

Clone to your project root:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .mindspore-skills
```

Codex reads `AGENTS.md` automatically. Verify with:

```bash
codex "Summarize the MindSpore skills available."
```

See [Codex AGENTS guide](https://developers.openai.com/codex/guides/agents-md) for more details.

## Available Skills

### Operator Development

| Skill | Description |
|-------|-------------|
| `cpu-plugin-builder` | Build CPU operators via ATen/libtorch adaptation |
| `cpu-native-builder` | Build native CPU kernels with Eigen/SLEEF |
| `gpu-builder` | Build GPU operators with CUDA |
| `npu-builder` | Build NPU operators for Huawei Ascend |

### Model Migration

| Skill | Description |
|-------|-------------|
| `hf-diffusers-migrate` | Migrate HF diffusers models to mindone.diffusers |
| `hf-transformers-migrate` | Migrate HF transformers models to mindone.transformers |
| `model-migrate` | Migrate PyTorch repos to MindSpore |

## Available Commands

### Operator Development

| Command | Description |
|---------|-------------|
| `/ms-api-builder` | Platform router (CPU/GPU/NPU) |
| `/ms-cpu-builder` | CPU approach router (plugin/native) |
| `/ms-cpu-plugin-builder` | ATen adaptation workflow |
| `/ms-cpu-native-builder` | Native kernel workflow |
| `/ms-gpu-builder` | CUDA kernel workflow |
| `/ms-npu-builder` | Ascend NPU workflow |

### Model Migration

| Command | Description |
|---------|-------------|
| `/ms-migrate` | Migration router (HF/third-party) |
| `/ms-hf-migrate` | HF library router (diffusers/transformers) |
| `/ms-hf-diffusers-migrate` | HF diffusers migration workflow |
| `/ms-hf-transformers-migrate` | HF transformers migration workflow |
| `/ms-model-migrate` | PyTorch repo migration workflow |

## Usage Examples

### Build a CPU operator

```
/ms-cpu-plugin-builder

> Help me implement the linspace operator for MindSpore CPU
```

### Choose platform interactively

```
/ms-api-builder

> I need to build a softmax operator
```

## Repository Structure

```
mindspore-skills/
├── .claude-plugin/          # Claude Code plugin config
├── commands/                # Slash commands
│   ├── ms-api-builder.md    # Operator platform router
│   ├── ms-cpu-builder.md    # CPU approach router
│   ├── ms-migrate.md        # Migration router
│   ├── ms-hf-migrate.md     # HF library router
│   └── ...
├── skills/                  # Skill definitions
│   ├── cpu-plugin-builder/  # ATen/libtorch operators
│   ├── cpu-native-builder/  # Native CPU kernels
│   ├── gpu-builder/         # CUDA operators
│   ├── npu-builder/         # Ascend NPU operators
│   ├── hf-diffusers-migrate/   # HF diffusers migration
│   ├── hf-transformers-migrate/ # HF transformers migration
│   └── model-migrate/       # PyTorch repo migration
├── AGENTS.md                # Codex instructions
└── gemini-extension.json    # Gemini CLI config
```

## Contributing

1. Fork this repository
2. Add your skill to `skills/<skill-name>/SKILL.md`
3. Add a command to `commands/<command-name>.md`
4. Update `AGENTS.md` with the new skill
5. Submit a pull request

## License

Apache 2.0
