# API Memory Consistency Analysis

Diagnose why a single torch_npu API uses more device memory than the
equivalent GPU API, using dedicated memory benchmarks, plog analysis, and
known-issue matching.

## When to Load

Load this reference when:

- Stage 2 identifies a specific operator dominating peak memory (Branch F)
  and the user asks why it uses more memory on NPU than on GPU
- the user explicitly invokes `/api_memory_analyze`
- the user reports that a single API shows higher NPU memory than GPU
  without a broader workload performance problem
- the user provides a memory test script and asks to analyze memory
  consistency (treat the script as a conforming NPU script; skip Generate Script step and always use Path B in Run tests)

## Stack Routing

Before proceeding, confirm the runtime stack from the `PerformanceProfile`
or from user context:

- `pta` (torch_npu): proceed with the analysis below
- `mindspore`: inform the user that API-level memory consistency analysis
  is not yet supported for MindSpore workloads and will be added in a
  future version. Return to the main performance workflow without
  interruption.

## Execution Environment

Detect whether the agent is running on the Ascend (NPU) server itself or
on a separate development machine. This determines how test scripts and
builds are executed.

Detection method: run locally:

```bash
npu-smi info
```

| Detection result | Mode | Behavior |
|---|---|---|
| Succeeds | `local-npu` | NPU tests and builds run locally; GPU tests still use SSH |
| Fails | `remote` | All tests and builds run via SSH (current default) |

Record the detected mode as `execution_mode`. Pass `--local-npu` to
execution scripts when mode is `local-npu`.

GPU tests always run via SSH to the GPU server regardless of execution
environment.

## Entry Point Routing

| User provides | Start from | Test runner |
|---|---|---|
| API name + input spec (no script) | Generate Script | `run_remote_mem_test.py` (Path A) |
| NPU script path (no results yet) | Prepare and Collect | `execute_mem_test.py` (Path B) |
| Both `mem_results_*.md` AND `filtered_plog_*.log` | Analyze Root Cause | — |
| Only one of the above (incomplete) | Prepare and Collect | same as original entry |

If the user provides a `pytorch_npu` source path at any point, record it
for source code analysis in Analyze Root Cause Step 2.

---

## Generate Script

Entry condition: user provides a torch API name and input specification
**but no test script**. Do NOT enter this section if the user already
provided a script — go to Prepare and Collect directly.

Extract from the user request: API full path (required), input shapes
(required), input dtype (default `torch.float32`), and extra API
arguments (if any). Ask for any missing required items. Use the
user-provided shapes as-is — do not generate additional shape variants.

Generate **both NPU and GPU scripts** following the contract in
`references/mem-test-script-contract.md` (read it before generating).

Write both scripts to the working directory, then proceed to Prepare and
Collect with both script paths.

---

## Prepare and Collect

### Server configuration

Before running any test, read `<skill_directory>/references/servers.json`.

**If the file exists and is valid JSON** — present the configuration to the user in a table showing Role, Host, User, and Remote dir for each server, then say:

*"Server config loaded from `<absolute_path>`. I will use it directly. Feel free to edit this file if you need to change any settings."*

Proceed to "Run tests" without waiting for confirmation.

**If the file is missing or malformed** — ask the user to provide server info in a single message(do NOT ask fields one at a time). `remote` mode requires both NPU and GPU; `local-npu` requires GPU only. Each server needs: host (IP or hostname), user, password, remote_dir (remote working directory), and optionally env_script (absolute path to env setup script, defaults to `~/.bashrc`). 

After collecting, present the info back to the user for confirmation. Once confirmed, write `<skill_directory>/references/servers.json` with keys `servers.npu` and `servers.gpu` (each containing `host`, `user`, `password`, `remote_dir`, `description`, and optionally `env_script`) and top-level `"default": "npu"`. In `local-npu` mode the `npu` entry may be omitted.

### Run tests

Do not read script source — just execute and inspect output. Add
`--local-npu` when `execution_mode` is `local-npu`.

**Path A — Agent-generated scripts** (both NPU and GPU scripts exist):

```bash
python <skill_directory>/scripts/api_memory/run_remote_mem_test.py \
    <npu_script> <gpu_script> --api-name <api> [--local-npu]
```

**Path B — User-provided NPU script** (GPU script not yet created):

```bash
python <skill_directory>/scripts/api_memory/execute_mem_test.py \
    <npu_script> [--local-npu]
```

`execute_mem_test.py` automatically generates the GPU script via
`convert_npu_to_gpu.py` and then calls `run_remote_mem_test.py`.

Interpret output:

- `[SUCCESS]` — extract `api_name` and file paths from output, proceed
  to Check Results below
- `[SCRIPT_ERROR]` — the test script has a runtime issue. Show the error
  to the user, then ask: "(A) diagnose and fix, or (B) skip and proceed
  with whatever results are available?"
- `[ERROR]` (without `SCRIPT_`) — a preset validation error. Show error
  and stop.

Hard gate: you must have `mem_results_<api_name>.md` +
`filtered_plog_<api_name>.log` before entering Analyze Root Cause.

### Check Results

Read `mem_results_<api_name>.md`. Key metrics:

| Metric | Description |
|---|---|
| Memory Benchmark heading | Extract `torch.xxx` as `TARGET_API` |
| total_driver_GB | Actual driver-level memory delta |
| reserved_GB | PTA CachingAllocator reserved (NPU) or gpu_reserved_GB (GPU) |
| activated_GB | PTA CachingAllocator peak allocated (most important) |
| Ratio | NPU / GPU — above 1.05x warrants investigation |

Early exit check on activated_GB ratio:

- 1.05 or below: inform user memory is normal, ask to verify test script
  correctness, skip remaining analysis, and return to main workflow
- above 1.05: proceed to Analyze Root Cause

---

## Analyze Root Cause

### Step 1. Plog analysis and known-issue lookup

Perform plog analysis and known-issue lookup together before presenting
any conclusions to the user.

#### 1a. Plog analysis

Inputs (all required — do NOT read any other files in this step):

| Source | What to extract |
|---|---|
| `mem_results_<api_name>.md` — Key Code | torch API call, input shape/dtype → estimate expected memory baseline |
| `mem_results_<api_name>.md` — Table | NPU vs GPU metrics (total_driver_GB, reserved_GB, activated_GB, ratio) |
| `filtered_plog_<api_name>.log` — Summary | `Workspace allocs: #N: ... bytes ... \| op: aclnnXXX` |
| `filtered_plog_<api_name>.log` — Events | `PTA CachingAllocator malloc/free`, `DevMalloc`, `workspaceSize_:N` → identify peak allocated |

Goal: pinpoint which NPU-side aclnn ops account for extra memory beyond
what the API semantically needs. For each suspected interface, record:
aclnn interface name + estimated memory impact + suspected root cause.

Common patterns:

| Pattern | Plog signature | Root cause |
|---|---|---|
| Internal Cast | Cast node in workspace (e.g. FP32 → FP16) | aclnn op does dtype conversion internally |
| Large workspace | `workspaceSize_` >> input size | aclnn algorithm needs large scratch buffer |
| Redundant Contiguous | Multiple `Contiguous` calls per op | Non-contiguous tensor triggers extra copy |

After analysis, classify the plog outcome:

| Outcome | Definition |
|---|---|
| Target attribution | Overconsumption traced to aclnn ops of `TARGET_API` itself |
| Non-target attribution | Source is identifiable but not from `TARGET_API` (e.g. preprocessing, implicit Cast before/after the target aclnn kernel) |
| Inconclusive | Memory allocation purpose unclear (e.g. abnormal memory consumption is neither from preprocessing/postprocessing nor from `TARGET_API` itself, or no aclnn kernel calls related to `TARGET_API` appear in plog) |

If the outcome is Non-target attribution or Inconclusive, you MUST
include the category name **and** a one-sentence explanation in
downstream output (1c progress checkpoint, Step 3 Present Findings).
Do NOT omit or downplay these outcomes — they are as important as
Target attribution for the user's decision-making.

If the plog outcome is **Inconclusive**, skip 1b (no suspected interfaces
to look up) and proceed directly to 1c.

#### 1b. Known-issue lookup

For each suspected aclnn interface from 1a, use Grep to search
`references/memory_consistency_issue_cases.md` by interface name. Do not
read the full file.

Name extraction: from plog entries like
`aclnn[OpName]_[Num]_[InnerKernel]`, extract base name `aclnn[OpName]`
(e.g. `aclnnInplaceNormal_1_CastAiCore` → `aclnnInplaceNormal`).
Deduplicate before searching.

Classify each suspected interface:

| Search result | Classification |
|---|---|
| Known issue found, corroborates plog | Closed-loop |
| Known issue found, does not corroborate | Needs source analysis |
| No match, plog clearly indicates CANN issue | Closed-loop (CANN) |
| No match, root cause unclear | Needs source analysis |

#### 1c. Progress checkpoint and decision

Briefly inform the user of plog outcome, known-issue match status, and
preliminary judgment (closed-loop vs needs deeper analysis). Then:

- All interfaces closed-loop → skip to Present Findings
- Source path already known → proceed to Step 2 immediately
- Source path not yet known → ask user for the `pytorch_npu` source path
- User declines to provide source → skip to Present Findings with
  plog-level conclusions only (warn that root cause may be incomplete)


### Step 2. Source code analysis and fix verification

Perform source code analysis for every suspected interface that was not
closed-loop in Step 1. If a torch_npu code fix is feasible, build and
verify it remotely through an iterative cycle.

#### 2a. Source analysis

Prerequisite: `pytorch_npu` source path (collected in Step 1c or earlier
from user context).

- If already provided, use it directly.
- If the user cannot provide it, skip source code analysis, present only
  plog-level conclusions, and warn the user.

If source is available:

1. Search `pytorch_npu/third_party/op-plugin/` for:
   - The operator kernel implementation
   - The dispatch path (`op_plugin_functions.yaml`)
   - Whether a composite decomposition is used
2. Check if an optimization exists in compiled mode but not eager mode
   (search `torch_npu/_inductor/`)
3. Determine fix location:
   - torch_npu code issue → propose code fix, apply locally, continue
     to 2b
   - CANN issue → recommend filing an issue to the CANN team, skip to
     Present Findings

#### 2b. Build, verify, and iterate

Entry condition: a code fix was proposed and applied locally in 2a. If 2a concluded with a CANN issue recommendation or no code change, skip to Present Findings.

Inform the user that the code fix will now be compiled and verified on the NPU server.
Resolve script arguments — only ask the user for values marked **ask**:

| Argument | `remote` | `local-npu` | Source |
|---|---|---|---|
| `--local-pta-path` | reuse | reuse | pytorch_npu source path from Step 2a |
| `--servers-json` | reuse | — | "Server configuration" step |
| `--remote-pta-path` | **ask** | — | pytorch_npu source root on remote NPU server |
| `--container-name` | **ask** (optional) | **ask** (optional) | Docker container for build; omit to build on host |
| `--verify-cmd` | reuse | reuse | NPU test script from "Run tests" (i.e. `python <npu_script>`) |

If the user chooses to skip compilation, record current findings and go to Present Findings.

Run `scripts/api_memory/src_code_build_verify.py`:

```bash
# remote mode，it auto-generates a patch from local changes, uploads, builds, installs, verifies, then reverts
python src_code_build_verify.py --local-pta-path <local_pta_path> \
    --remote --remote-pta-path <path> --servers-json <json> \
    [--container-name <name>] --verify-cmd "<verify_cmd>"

# local-npu mode
python src_code_build_verify.py --local-pta-path <pta_path> \
    [--container-name <name>] --verify-cmd "<verify_cmd>"
```

Iterate until resolved or the user decides to stop:

- activated_GB ratio ≤ 1.05x → report success, proceed to Present Findings.
- Issue persists or new issue appears → analyze the new output, revise the source fix, and re-run.
- Build fails → diagnose the build error, fix, and re-run.

### Step 3. Present findings

This step is mandatory. Do not skip it and do not wait for user
permission.

Collect all conclusions and report:

```
## Analysis: {TARGET_API} Memory Overconsumption

### Memory Overview
- NPU peak activated: X GiB | GPU baseline: Y GiB | Ratio: X/Y

### Root Causes
For each suspect:
- Interface: aclnnXxx | Owner: torch_npu / CANN | Impact: N GiB
- Evidence: plog analysis summary
- Known issue: ISSUE-xxx (corroborated) / None
- Fix: description or code change

### Code Changes (if applicable)
- File: path → Change: description

### Summary
[1-2 sentences on primary cause of NPU-GPU memory gap]
```

---

## Validate and Accumulate

### Ask user for validation

Ask: "Does the above analysis resolve your issue?"

- User confirms → if root cause was already a known issue from the knowledge base,
end the workflow directly (no need to accumulate again); otherwise proceed to
Accumulate Experience
- User says no → present two options:
  1. "I have specific concerns or new evidence" → user provides details,
     return to Step 2 (source code analysis and fix verification) with
     the updated context. If the user has not yet provided a `pytorch_npu`
     source path, ask for it now.
  2. "Abandon further analysis" → end workflow without knowledge
     accumulation

### Accumulate experience

Ask for the ISSUE number. If none, generate an internal ID:
`INT-YYYYMMDD-NNN`.

Append to `references/memory_consistency_issue_cases.md`:

```yaml
### <issue_number>
- issue_number: "<issue_number>"
- description: "<brief issue description>"
- aclnn_interface: "<related aclnn interface>"
- root_cause: "<root cause analysis>"
- solution: "<resolution strategy>"
- category: "<issue classification>"
```

Category values:

- `internal-cast-overhead` — internal Cast inserted by aclnn op
- `missing-aclnn-kernel` — no dedicated aclnn kernel for the operator
- `worst-case-prealloc` — pre-allocating for worst-case output size
- `oversized-workspace` — aclnn workspace disproportionate to input
- `torch-npu-logic-defect` — missing optimization in torch_npu eager path
- `other` — other

