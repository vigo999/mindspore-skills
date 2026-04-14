#!/usr/bin/env python3
"""
Source code build & verify — build torch_npu from source and run verification.

Mode is determined by --remote flag and --container-name:
  --remote + --container-name  : patch → remote container build → remote install/verify
  --remote (no container)      : patch → remote host build → remote install/verify
  --container-name (no remote) : local container build → local install/verify
  (neither)                    : local host build → local install/verify

Usage
-----
    # Local: build in container (on the NPU server itself)
    python src_code_build_verify.py /home/zxl/pytorch \
        --container-name zxl_build \
        --verify-cmd "python test_api.py"

    # Local: build on host directly
    python src_code_build_verify.py /home/zxl/pytorch \
        --verify-cmd "python test_api.py"

    # Remote: from a CPU laptop, build & verify on NPU server
    python src_code_build_verify.py D:/open_source/pytorch_npu \
        --remote \
        --remote-pta-path /home/zxl/pytorch \
        --container-name zxl_build \
        --servers-json servers.json \
        --verify-cmd "python /home/zxl/test.py"
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import getpass
from pathlib import Path

# ════════════════════════════════════════════════════════════════════
#  Logging
# ════════════════════════════════════════════════════════════════════

def log(tag, msg):
    print(f"[{tag}] {msg}")


def _print_tail(text, n=80, prefix="  "):
    for ln in text.strip().split("\n")[-n:]:
        print(f"{prefix}{ln}")


# ════════════════════════════════════════════════════════════════════
#  Local shell execution
# ════════════════════════════════════════════════════════════════════

def _local_run(cmd, timeout=600, cwd=None):
    kwargs = dict(
        shell=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if cwd:
        kwargs["cwd"] = cwd
    if os.name != "nt":
        kwargs["executable"] = "/bin/bash"
    r = subprocess.run(cmd, **kwargs)
    return r.stdout, r.stderr, r.returncode


# ════════════════════════════════════════════════════════════════════
#  SSH helpers  (paramiko, imported lazily)
# ════════════════════════════════════════════════════════════════════

def _ssh_connect(server_cfg):
    try:
        import paramiko
    except ImportError:
        sys.exit("Error: paramiko is required for remote mode.  pip install paramiko")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        server_cfg["host"],
        username=server_cfg["user"],
        password=server_cfg["password"],
        timeout=15,
    )
    transport = ssh.get_transport()
    if transport:
        transport.set_keepalive(30)
    return ssh


def _remote_run(ssh, cmd, timeout=600):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def _upload_bytes(ssh, data: bytes, remote_path: str):
    sftp = ssh.open_sftp()
    with sftp.open(remote_path, "wb") as f:
        f.write(data)
    sftp.close()


# ════════════════════════════════════════════════════════════════════
#  Config / detection helpers
# ════════════════════════════════════════════════════════════════════

def load_server_config(servers_json_path):
    with open(servers_json_path) as f:
        cfg = json.load(f)
    server_name = cfg.get("default", "npu")
    servers = cfg.get("servers", {})
    if server_name not in servers:
        sys.exit(f"Error: server '{server_name}' not found in {servers_json_path}")
    return servers[server_name]


def _detect_py_version(version_str):
    m = re.search(r"Python\s+(\d+\.\d+)", version_str)
    return m.group(1) if m else "3.10"


def detect_python_version_local(container=None, user=None):
    if container:
        uflag = f"-u {user} " if user else ""
        cmd = f'docker exec {uflag}{container} bash -lc "python3 --version" 2>&1'
    else:
        cmd = "python3 --version 2>&1 || python --version 2>&1"
    out, _, _ = _local_run(cmd, timeout=30)
    return _detect_py_version(out)


def detect_python_version_remote(ssh, container=None, user=None):
    if container:
        uflag = f"-u {user} " if user else ""
        cmd = f'docker exec {uflag}{container} bash -lc "python3 --version" 2>&1'
    else:
        cmd = "python3 --version 2>&1 || python --version 2>&1"
    out, _, _ = _remote_run(ssh, cmd, timeout=30)
    return _detect_py_version(out)


# ════════════════════════════════════════════════════════════════════
#  Patch generation (runs on the *local* machine)
# ════════════════════════════════════════════════════════════════════

def generate_patch(local_pta_path):
    """Create a unified diff of uncommitted changes (ignoring submodules)."""
    cmd = "git diff --no-color --no-ext-diff --ignore-submodules HEAD"
    out, err, rc = _local_run(cmd, timeout=30, cwd=local_pta_path)
    if rc != 0:
        log("PATCH", f"git diff failed: {err}")
        return None
    if not out.strip():
        log("PATCH", "No local changes detected (staged + unstaged)")
        return None
    return out


# ════════════════════════════════════════════════════════════════════
#  LOCAL mode helpers
# ════════════════════════════════════════════════════════════════════

def build_in_container_local(pta_path, container, py_ver, cur_user):
    cmd = (
        f"docker exec -w {pta_path} -u {cur_user} {container} "
        f'bash -lc "bash ci/build.sh --python={py_ver}" 2>&1'
    )
    log("BUILD", f"Building in container {container}...")
    return _local_run(cmd, timeout=3600)


def build_on_host_local(pta_path, py_ver):
    cmd = f"cd {pta_path} && python{py_ver} setup.py build bdist_wheel 2>&1"
    log("BUILD", "Building locally on host...")
    return _local_run(cmd, timeout=3600)


def install_wheel_local(pta_path):
    cmd = f"pip install {pta_path}/dist/torch_npu*.whl --force-reinstall --no-deps 2>&1"
    log("INSTALL", "Installing wheel...")
    return _local_run(cmd, timeout=300)


def verify_local(verify_cmd):
    log("VERIFY", "Running verification...")
    return _local_run(verify_cmd, timeout=600)


# ════════════════════════════════════════════════════════════════════
#  REMOTE mode helpers
# ════════════════════════════════════════════════════════════════════

def remote_apply_patch(ssh, remote_pta_path, patch_data):
    tag = int(time.time())
    patch_remote = f"/tmp/torch_npu_patch_{tag}.patch"
    _upload_bytes(ssh, patch_data.encode("utf-8"), patch_remote)
    log("PATCH", f"Uploaded patch → {patch_remote}")

    # Try git apply first (strict)
    cmd = f"cd {remote_pta_path} && git apply --stat {patch_remote} && git apply {patch_remote}"
    out, err, rc = _remote_run(ssh, cmd, timeout=60)
    if rc == 0:
        log("PATCH", "Applied via git apply")
        if out.strip():
            for ln in out.strip().split("\n"):
                log("PATCH", f"  {ln}")
        return True, patch_remote

    # Fallback: patch -p1 (tolerant of index/commit differences)
    log("PATCH", "git apply failed, trying patch -p1...")
    cmd2 = f"cd {remote_pta_path} && patch -p1 --fuzz=3 < {patch_remote}"
    out2, err2, rc2 = _remote_run(ssh, cmd2, timeout=60)
    if rc2 == 0:
        log("PATCH", "Applied via patch -p1")
        if out2.strip():
            for ln in out2.strip().split("\n"):
                log("PATCH", f"  {ln}")
        return True, patch_remote

    log("PATCH", f"Apply FAILED:\n  git apply: {err.strip()}\n  patch -p1: {(out2 + err2).strip()}")
    return False, patch_remote


def remote_revert_patch(ssh, remote_pta_path, patch_remote):
    cmd = f"cd {remote_pta_path} && git checkout -- . 2>&1 && rm -f {patch_remote}"
    out, err, rc = _remote_run(ssh, cmd, timeout=60)
    if rc == 0:
        log("PATCH", "Reverted changes on remote (git checkout -- .)")
    else:
        log("PATCH", f"Warning: revert failed: {out}{err}")


def remote_clean_wheels(ssh, remote_pta_path, container=None, user=None):
    if container:
        uflag = f"-u {user} " if user else ""
        cmd = f'docker exec -w {remote_pta_path} {uflag}{container} bash -c "rm -f dist/torch_npu*.whl" 2>/dev/null'
    else:
        cmd = f"rm -f {remote_pta_path}/dist/torch_npu*.whl"
    _remote_run(ssh, cmd, timeout=30)


def remote_build_in_container(ssh, container, user, remote_pta_path, py_ver):
    cmd = (
        f"docker exec -w {remote_pta_path} -u {user} {container} "
        f'bash -lc "bash ci/build.sh --python={py_ver}" 2>&1'
    )
    log("BUILD", f"Building in container {container} on remote (may take 20+ min)...")
    log("BUILD", f"  docker exec -w {remote_pta_path} -u {user} {container} "
                  f'bash -lc "bash ci/build.sh --python={py_ver}"')
    return _remote_run(ssh, cmd, timeout=3600)


def remote_build_on_host(ssh, remote_pta_path, py_ver, env_script):
    cmd = (
        f"source {env_script} && "
        f"cd {remote_pta_path} && "
        f"python{py_ver} setup.py build bdist_wheel 2>&1"
    )
    log("BUILD", "Building on remote host (may take 20+ min)...")
    return _remote_run(ssh, cmd, timeout=3600)


def remote_install_wheel(ssh, remote_pta_path, env_script):
    cmd = (
        f"source {env_script} && "
        f"pip install {remote_pta_path}/dist/torch_npu*.whl "
        f"--force-reinstall --no-deps 2>&1"
    )
    log("INSTALL", "Installing wheel on remote host...")
    return _remote_run(ssh, cmd, timeout=300)


def remote_verify(ssh, verify_cmd, env_script):
    cmd = f"source {env_script} && {verify_cmd}"
    log("VERIFY", "Running verification on remote host...")
    return _remote_run(ssh, cmd, timeout=600)


# ════════════════════════════════════════════════════════════════════
#  Shared output / reporting
# ════════════════════════════════════════════════════════════════════

def report_build_failure(out, err, tag="BUILD"):
    log(tag, "--- stdout (last 80 lines) ---")
    _print_tail(out, 80)
    if err.strip():
        log(tag, "--- stderr (last 40 lines) ---")
        _print_tail(err, 40)


def report_verification(out, err, rc, elapsed):
    print("\n" + "=" * 72)
    print("  VERIFICATION OUTPUT")
    print("=" * 72)
    print(out)
    if err.strip():
        print("--- stderr ---")
        print(err)
    print("=" * 72)
    log("BUILD", f"Verify exit_code={rc}")
    log("BUILD", f"Total elapsed : {elapsed:.1f}s")


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build torch_npu from source and verify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Local: build in container
              python src_code_build_verify.py --local-pta-path /home/zxl/pytorch \\
                  --container-name zxl_build --verify-cmd "python test.py"

              # Remote: build on NPU server from CPU laptop
              python src_code_build_verify.py --local-pta-path D:/open_source/pytorch_npu \\
                  --remote --remote-pta-path /home/zxl/pytorch \\
                  --container-name zxl_build \\
                  --servers-json servers.json \\
                  --verify-cmd "python /home/zxl/test.py"
        """),
    )
    ap.add_argument("--local-pta-path", required=True,
                    help="Local torch_npu source root")
    ap.add_argument("--remote-pta-path", default="",
                    help="torch_npu path on remote server (required for --remote)")
    ap.add_argument("--container-name", "--container", default="",
                    help="Docker container name (with: build in container; without: build on host)")
    ap.add_argument("--remote", action="store_true",
                    help="Remote build via SSH (requires --servers-json)")
    ap.add_argument("--servers-json", default="",
                    help="Path to servers.json (uses 'default' server entry)")
    ap.add_argument("--patch", default="",
                    help="Pre-made patch file (skip auto-generation)")
    ap.add_argument("--verify-cmd", required=True,
                    help="Verification command to run after install")
    args = ap.parse_args()

    local_pta = os.path.abspath(args.local_pta_path) if args.local_pta_path else ""

    is_remote = args.remote
    use_container = bool(args.container_name)
    mode_label = ("remote" if is_remote else "local") + ("+container" if use_container else "+host")

    T = "BUILD"
    t0 = time.time()
    log(T, f"Mode          : {mode_label}")

    # ╔════════════════════════════════════════════════════════════╗
    # ║  REMOTE MODE                                              ║
    # ╚════════════════════════════════════════════════════════════╝
    if is_remote:
        if not args.servers_json:
            sys.exit("Error: --servers-json is required for --remote")
        if not args.remote_pta_path:
            sys.exit("Error: --remote-pta-path is required for --remote")
        if not local_pta or not os.path.isdir(local_pta):
            sys.exit(f"Error: local PTA path not found: {local_pta}")

        server_cfg = load_server_config(args.servers_json)
        env_script = server_cfg.get("env_script", "~/.bashrc")
        ssh_user = server_cfg["user"]

        log(T, f"Local PTA     : {local_pta}")
        log(T, f"Remote PTA    : {args.remote_pta_path}")
        log(T, f"Server        : {server_cfg['host']} (user={ssh_user})")
        log(T, f"Container     : {args.container_name or '(none, build on host)'}")
        log(T, f"Env script    : {env_script}")
        log(T, f"Verify cmd    : {args.verify_cmd}")

        # 1 ── Patch ─────────────────────────────────────────────
        if args.patch and os.path.isfile(args.patch):
            with open(args.patch, encoding="utf-8") as f:
                patch_data = f.read()
            log(T, f"Using provided patch: {args.patch}")
        else:
            log(T, "Generating patch from local changes...")
            patch_data = generate_patch(local_pta)
            if not patch_data:
                sys.exit("Error: no local changes to patch (nothing to build)")
            nlines = patch_data.count("\n")
            log(T, f"Patch generated: {nlines} lines")

        # 2 ── SSH connect ───────────────────────────────────────
        log(T, f"Connecting to {server_cfg['host']}...")
        ssh = _ssh_connect(server_cfg)
        log(T, "SSH connected")

        patch_remote = None
        try:
            # 3 ── Apply patch ───────────────────────────────────
            ok, patch_remote = remote_apply_patch(ssh, args.remote_pta_path, patch_data)
            if not ok:
                sys.exit("Error: failed to apply patch on remote server")

            # 4 ── Detect Python version ─────────────────────────
            py_ver = detect_python_version_remote(
                ssh,
                container=args.container_name or None,
                user=ssh_user,
            )
            log(T, f"Python version: {py_ver}")

            # 5 ── Clean old wheels ──────────────────────────────
            remote_clean_wheels(
                ssh, args.remote_pta_path,
                container=args.container_name or None,
                user=ssh_user,
            )
            log(T, "Cleaned old wheels on remote")

            # 6 ── Build ─────────────────────────────────────────
            if args.container_name:
                out, err, rc = remote_build_in_container(
                    ssh, args.container_name, ssh_user,
                    args.remote_pta_path, py_ver,
                )
            else:
                out, err, rc = remote_build_on_host(
                    ssh, args.remote_pta_path, py_ver, env_script,
                )

            if rc != 0:
                log(T, f"Build FAILED (exit={rc})")
                report_build_failure(out, err, T)
                sys.exit(1)

            log(T, "Build succeeded")
            _print_tail(out, 5, prefix=f"[{T}]   ")

            # 7 ── Install ───────────────────────────────────────
            out, err, rc = remote_install_wheel(ssh, args.remote_pta_path, env_script)
            if rc != 0:
                log(T, f"Install FAILED (exit={rc})")
                _print_tail(out + "\n" + err, 20)
                sys.exit(1)
            log(T, "Install succeeded")

            # 8 ── Verify ────────────────────────────────────────
            out, err, rc = remote_verify(ssh, args.verify_cmd, env_script)
            report_verification(out, err, rc, time.time() - t0)

        finally:
            if patch_remote:
                remote_revert_patch(ssh, args.remote_pta_path, patch_remote)
            ssh.close()

        return rc

    # ╔════════════════════════════════════════════════════════════╗
    # ║  LOCAL / CONTAINER MODE                                   ║
    # ╚════════════════════════════════════════════════════════════╝
    if not local_pta or not os.path.isdir(local_pta):
        sys.exit(f"Error: PTA path not found: {local_pta}")

    cur_user = getpass.getuser()

    log(T, f"PTA path      : {local_pta}")
    log(T, f"Verify cmd    : {args.verify_cmd}")
    log(T, f"Current user  : {cur_user}")

    py_ver = detect_python_version_local(args.container_name or None, cur_user)
    log(T, f"Python version: {py_ver}")

    # Clean old wheels
    clean_cmd = f"rm -f {local_pta}/dist/torch_npu*.whl"
    if use_container:
        clean_cmd = (
            f"docker exec -w {local_pta} -u {cur_user} {args.container_name} "
            f"bash -c '{clean_cmd}' 2>/dev/null || true"
        )
    _local_run(clean_cmd, timeout=30)
    log(T, "Cleaned old wheels")

    # Build
    if use_container:
        out, err, rc = build_in_container_local(
            local_pta, args.container_name, py_ver, cur_user,
        )
    else:
        out, err, rc = build_on_host_local(local_pta, py_ver)

    if rc != 0:
        log(T, f"Build FAILED (exit={rc})")
        report_build_failure(out, err, T)
        sys.exit(1)

    log(T, "Build succeeded")
    _print_tail(out, 5, prefix=f"[{T}]   ")

    # Install
    out, err, rc = install_wheel_local(local_pta)
    if rc != 0:
        log(T, f"Install FAILED (exit={rc})")
        _print_tail(out + "\n" + err, 20)
        sys.exit(1)
    log(T, "Install succeeded")

    # Verify
    out, err, rc = verify_local(args.verify_cmd)
    report_verification(out, err, rc, time.time() - t0)

    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
