#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_plugin_add}"
OUT_DIR="${SCRIPT_DIR}/runs/${RUN_ID}/out"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/artifacts" "${OUT_DIR}/meta"

cat > "${OUT_DIR}/meta/env.json" <<'EOF'
{
  "mindspore_version": "unknown",
  "cann_version": "unknown",
  "driver_version": "unknown",
  "python_version": "unknown",
  "platform": "unknown",
  "git_commit": "unknown"
}
EOF

cat > "${OUT_DIR}/meta/inputs.json" <<EOF
{
  "skill": "cpu-plugin-builder",
  "run_id": "${RUN_ID}",
  "parameters": {
    "example": "cpu/plugin_add",
    "mode": "smoke"
  },
  "masked_keys": []
}
EOF

cat > "${OUT_DIR}/logs/run.log" <<'EOF'
[run] started
[run] completed
EOF

cat > "${OUT_DIR}/logs/build.log" <<'EOF'
[build] mock build completed
EOF

cat > "${OUT_DIR}/logs/verify.log" <<'EOF'
[verify] x=[1.0,2.0], y=[3.0,4.0], out=[4.0,6.0]
[verify] pass
EOF

cat > "${OUT_DIR}/artifacts/README.md" <<'EOF'
# No Artifacts Produced

This smoke example validates reporting/output contract only.
EOF

START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_TIME="${START_TIME}"

cat > "${OUT_DIR}/report.json" <<EOF
{
  "schema_version": "1.0.0",
  "skill": "cpu-plugin-builder",
  "run_id": "${RUN_ID}",
  "status": "success",
  "start_time": "${START_TIME}",
  "end_time": "${END_TIME}",
  "duration_sec": 0,
  "steps": [
    {"name": "build", "status": "success", "duration_sec": 0, "message": "mock build completed"},
    {"name": "verify", "status": "success", "duration_sec": 0, "message": "mock verify completed"}
  ],
  "logs": ["logs/run.log", "logs/build.log", "logs/verify.log"],
  "artifacts": ["artifacts/README.md"],
  "env_ref": "meta/env.json",
  "inputs_ref": "meta/inputs.json"
}
EOF

cat > "${OUT_DIR}/report.md" <<EOF
# Summary
- Skill: \`cpu-plugin-builder\`
- Run ID: \`${RUN_ID}\`
- Status: \`success\`

# What
- Task: \`plugin_add smoke example\`

# How
- Route: \`cpu-plugin-builder\`

# Verify
- Result: \`pass\`

# Artifacts
- \`artifacts/README.md\`

# Environment
- See \`meta/env.json\`

# Logs
- \`logs/run.log\`
- \`logs/build.log\`
- \`logs/verify.log\`

# Next
- Replace mock build with real CPU plugin compile and binary artifact output.
EOF

echo "Example run completed: ${OUT_DIR}"
