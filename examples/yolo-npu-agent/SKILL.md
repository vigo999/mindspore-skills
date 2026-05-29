---
name: ultralytics-npu-support
description: Guide agents to add, maintain, or resynchronize Huawei Ascend NPU support in Ultralytics YOLO repositories through torch_npu, including device selection, training, inference, export, AMP, memory, AutoBatch, backend boundaries, and validation.
---

# Ultralytics Ascend NPU Support

Use this skill when working on an Ultralytics YOLO repository and the user wants to add, repair, validate, or resynchronize Huawei Ascend NPU support. The target backend is PyTorch with `torch_npu`.

Typical triggers:

- add NPU or Ascend support to Ultralytics;
- support `device=npu` or `device=npu:0`;
- port an existing Ultralytics NPU patch to a newer upstream version;
- keep an NPU fork in sync after Ultralytics updates;
- diagnose missing NPU support in train, predict, val, export, or AutoBackend paths.

## Core Goal

Make Ultralytics treat Ascend NPU as a first-class accelerator without changing existing CPU, CUDA, or MPS behavior.

The implementation should follow the existing CUDA behavior as the reference, but it must not blindly reuse CUDA-only branches for NPU. For every CUDA-specific path, choose one explicit NPU decision:

- implement an equivalent `torch.npu` path;
- keep the feature CUDA-only and reject `device=npu` clearly;
- fall back deliberately with logging;
- leave unsupported with a clear error.

Silent accidental CUDA fallback is not acceptable when the user explicitly requests `device=npu`.

## Scope

The target is Ultralytics NPU support across the accelerator surfaces users expect from the project.

Complete NPU support targets:

- Huawei Ascend NPU through `torch_npu`;
- single-card and multi-card execution where Ultralytics has equivalent accelerator flows;
- `device=npu`, indexed NPU devices, and multi-NPU spellings with clear normalization rules;
- train, val, predict, benchmark, export, and PyTorch model runtime paths;
- AMP if supported by the local PyTorch and `torch_npu` versions, otherwise safe downgrade;
- memory, cache, timing, profiling, and OOM handling through NPU APIs;
- export behavior with explicit per-format rules;
- distributed training through the appropriate Ascend communication stack when the local environment and upstream architecture support it;
- real NPU tests that skip cleanly when no Ascend device is present.

Natural non-NPU boundaries still need explicit handling:

- interpreting `device=0` as NPU;
- TensorRT, DLA, or other NVIDIA-specific runtimes cannot become NPU runtimes and should reject `device=npu` clearly;
- CUDA-specific providers and packages must not be selected for NPU;
- unsupported third-party formats should fail clearly or use a deliberate documented CPU fallback;
- a broad accelerator abstraction refactor is optional only if it makes full NPU coverage safer and easier to maintain.

## User-Facing Device Semantics

Support and normalize:

- `device=npu` selects `npu:0`;
- `device=npu:0` selects NPU index 0;
- multi-NPU spellings such as `device=npu:0,npu:1` should be supported when the training/runtime path supports distributed NPU execution.

If a user requests multi-NPU in a code path that has not yet been implemented or cannot run distributed NPU, fail clearly at the boundary and name the unsupported path. Do not silently run single-card NPU, CPU, or CUDA instead.

Preserve existing CUDA semantics:

- `device=0` means CUDA device 0;
- `device=0,1` means CUDA multi-GPU;
- `device=-1` keeps idle CUDA GPU selection behavior;
- CPU and MPS behavior remain unchanged.

When no device is specified, preserve Ultralytics accelerator auto-selection with this order:

1. CUDA if available.
2. Ascend NPU if `torch_npu` imports successfully, `torch.npu` exists, and `torch.npu.is_available()` is true.
3. MPS if available.
4. CPU.

Automatic NPU probing must be best-effort. If importing or probing `torch_npu` raises an import, ABI, driver, CANN, or native-extension error, ignore NPU for omitted-device auto-selection and continue to MPS or CPU. Do not break CPU/CUDA users.

## Workflow

### 1. Establish Baselines

Before editing code:

1. Check the current branch and worktree status.
2. Identify the upstream base and whether this is a fresh port or a resync of an existing NPU fork.
3. Run or record the existing CPU/CUDA behavior needed to protect the migration.
4. Record the local `torch`, `torch_npu`, CANN, driver, and hardware versions when they are available.

Do not rely on fixed line numbers from an older Ultralytics version. Upstream changes frequently; locate behavior by responsibility.

### 2. Scan Accelerator Touchpoints

Search for CUDA, accelerator, and third-party backend touchpoints before editing. Include terms such as `select_device`, `torch.cuda`, `CUDA_VISIBLE_DEVICES`, `GradScaler`, `autocast`, `empty_cache`, `memory_reserved`, `OutOfMemoryError`, `AutoBackend`, `onnxruntime`, `TensorRT`, `paddle`, and `device.type`.

Build a capability checklist covering:

- device parsing and logging;
- trainer device setup and distributed execution;
- AMP, autocast, and GradScaler;
- memory usage, cache clearing, timing, profiling, and OOM detection;
- AutoBatch;
- predictor and validator tensor movement;
- AutoBackend runtime dispatch;
- ONNX Runtime, TensorRT, Paddle, and other third-party backend boundaries;
- export;
- environment and system information;
- tests.

### 3. Verify Official API Mappings

Do not migrate CUDA APIs by intuition. Before changing a CUDA, accelerator, or NVIDIA-specific call site, check official documentation for the target `torch_npu` and CANN environment.

Use these official documentation sources:

- Ascend Extension for PyTorch documentation: `https://www.hiascend.com/document/detail/zh/Pytorch`
- Common PyTorch model migration and replacement interfaces.
- `torch_npu.npu` API reference.
- PyTorch native API support status on Ascend.

Use the generic Ascend PyTorch documentation entry point above, then select the latest version or the version matching the local `torch_npu` / CANN / PyTorch stack. Version-specific URLs are examples only, not fixed sources of truth.

### 4. Produce the API Migration Decision Table

Create an API migration decision table before editing code:

```text
Location | CUDA/API usage | Official NPU mapping | Decision | Test
```

Every CUDA, accelerator, or NVIDIA-related touchpoint must be classified as one of:

- `NPU supported and migrated`;
- `NPU unsupported and explicitly rejected`;
- `Conditional fallback`;
- `Not applicable`;
- `Needs verification`.

The decision must cite the official mapping or support status used for that call site. If no official mapping exists, mark the point as `Needs verification` or choose an explicit rejection/fallback boundary.

### 5. Migrate by Module

Cover these implementation areas:

- device selection and normalization;
- training and distributed execution;
- inference, validation, benchmark, and data movement;
- AMP, autocast, and gradient scaling;
- memory, cache, timing, profiling, and OOM handling;
- AutoBatch;
- AutoBackend runtime dispatch;
- export;
- environment and system information;
- tests.

Prefer changes that match the current Ultralytics architecture. Small explicit NPU branches beside CUDA logic are acceptable when they are easy to review. A broader accelerator abstraction is acceptable only when it reduces duplication and makes NPU behavior safer to maintain.

Each module change should preserve this invariant: explicit NPU requests must either run on NPU, use a deliberate documented fallback, or fail with a precise boundary error. They must not accidentally select CUDA-only code paths.

### 6. Close the Migration Loop

After module migration:

1. Run real Ascend NPU smoke tests.
2. Check accuracy alignment against the preserved CPU/CUDA baseline where comparable.
3. Add or update the dedicated NPU tests.
4. Produce a capability matrix listing each area as supported, rejected, fallback, not applicable, or unverified.

## Resynchronizing After Upstream Updates

When applying NPU support to a newer Ultralytics version:

1. Do not blindly paste or cherry-pick old patches.
2. First identify how upstream changed the accelerator paths.
3. Rebuild the accelerator touchpoint checklist.
4. Recreate the API migration decision table from official documentation.
5. Reapply the NPU behavior by capability area, not by old line number.
6. Use old NPU patches as examples of intent, not as authoritative current code.
7. Keep upstream style and local helper APIs.
8. Preserve unrelated upstream changes.
9. Update or add tests for every adapted behavior.

At the end of a resync, produce a capability matrix listing each area as supported, rejected, fallback, or unverified.

## Testing

### Real Ascend NPU Tests

Add a dedicated NPU test module that skips at module level when `torch_npu` or an Ascend device is unavailable.

Real hardware tests should cover:

- `select_device("npu")`;
- `select_device("npu:0")`;
- tensor movement to NPU;
- minimal YOLO prediction;
- minimal one-epoch training;
- minimal validation;
- AMP request with success or safe downgrade;
- AutoBatch path;
- profiling path;
- at least one successful export path;
- at least one CUDA-only export rejection.

## Acceptance Criteria

Work is complete only when:

- existing CPU, CUDA, and MPS behavior is preserved;
- `device=npu`, `device=npu:0`, and supported multi-NPU spellings work on Ascend NPU;
- omitted-device auto-selection follows CUDA > NPU > MPS > CPU;
- train, val, and predict paths execute on real NPU when hardware is available;
- AMP either validates on NPU or downgrades safely with a clear warning;
- NPU memory, timing, profiling, cache clearing, and OOM paths avoid CUDA APIs;
- AutoBatch attempts real NPU behavior or falls back with a clear warning;
- export behavior is explicit for every relevant format;
- unsupported distributed NPU paths and CUDA-only backends fail clearly;
- real NPU tests pass on Ascend hardware and skip elsewhere;

## Reporting Back

When finished, summarize:

- changed files grouped by capability area;
- supported NPU user flows;
- unsupported or intentionally rejected flows;
- tests run;
- tests not run, especially real Ascend hardware tests;
- any remaining version-specific risks around PyTorch, `torch_npu`, CANN, drivers, or export support.
