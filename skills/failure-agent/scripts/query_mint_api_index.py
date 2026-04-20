#!/usr/bin/env python3
"""Read-only query helper for the mint API SQLite index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import sqlite3
except ImportError:  # pragma: no cover - tested via helper branch
    sqlite3 = None


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_DB = SKILL_ROOT / "reference" / "index" / "mint_api_index.db"


def json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=False)


def unavailable_payload(kind: str, db_path: Path) -> dict:
    return {"ok": False, "error": kind, "db_path": str(db_path)}


def check_index_available(db_path: Path) -> dict | None:
    if sqlite3 is None:
        return unavailable_payload("sqlite unavailable", db_path)
    if not db_path.exists():
        return unavailable_payload("index unavailable", db_path)
    return None


def scalar_list(conn: sqlite3.Connection, query: str, params: tuple) -> list[str]:
    return [str(row[0]) for row in conn.execute(query, params).fetchall()]


def path_list(conn: sqlite3.Connection, query: str, params: tuple) -> list:
    """Read path+anchor rows.  Return ``str`` when anchor is empty, ``dict`` otherwise."""
    result: list = []
    for row in conn.execute(query, params).fetchall():
        path = str(row[0])
        anchor = str(row[1]) if len(row) > 1 and row[1] else ""
        if anchor:
            result.append({"path": path, "anchor": anchor})
        else:
            result.append(path)
    return result


def path_hints(conn: sqlite3.Connection, api_id: int) -> dict:
    result = {
        "api_def_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'api_def_path' ORDER BY ordinal",
            (api_id,),
        ),
        "dispatch_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'dispatch_path' ORDER BY ordinal",
            (api_id,),
        ),
        "implementation_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'implementation_path' ORDER BY ordinal",
            (api_id,),
        ),
        "op_def_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'op_def_path' ORDER BY ordinal",
            (api_id,),
        ),
        "kernel_paths": {
            "pynative": {"ascend": [], "cpu": [], "gpu": []},
            "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
        },
        "infer_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'infer_path' ORDER BY ordinal",
            (api_id,),
        ),
    }
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            result["kernel_paths"][exec_mode][backend] = path_list(
                conn,
                f"SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'kernel_path_{exec_mode}_{backend}' ORDER BY ordinal",
                (api_id,),
            )
    return result


def support_matrix(conn: sqlite3.Connection, api_id: int) -> dict[str, dict[str, str]]:
    result = {
        "pynative": {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"},
        "graph_kbk_o0": {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"},
    }
    for row in conn.execute(
        "SELECT exec_mode, backend, support_state FROM api_support WHERE api_id = ?",
        (api_id,),
    ).fetchall():
        result[str(row["exec_mode"])][str(row["backend"])] = str(row["support_state"])
    return result


def evidence_digest(conn: sqlite3.Connection, api_id: int, limit: int) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT domain, exec_mode, backend, primitive_name, path, kind, anchor, summary
        FROM api_evidence
        WHERE api_id = ?
        ORDER BY domain, exec_mode, backend, ordinal
        LIMIT ?
        """,
        (api_id, limit),
    ).fetchall()
    return [
        {
            "domain": str(row["domain"]),
            "exec_mode": "" if row["exec_mode"] is None else str(row["exec_mode"]),
            "backend": "" if row["backend"] is None else str(row["backend"]),
            "primitive": str(row["primitive_name"]),
            "path": str(row["path"]),
            "kind": str(row["kind"]),
            "anchor": str(row["anchor"]),
            "summary": str(row["summary"]),
        }
        for row in rows
    ]


def fetch_api_row(conn: sqlite3.Connection, api_name: str):
    return conn.execute("SELECT * FROM api WHERE api_name = ?", (api_name,)).fetchone()


def build_meta_payload(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
    if row is None:
        return {"ok": False, "error": "schema_meta missing"}
    return {
        "ok": True,
        "meta": {
            "schema_version": str(row["schema_version"]),
            "generator_name": str(row["generator_name"]),
            "generator_version": str(row["generator_version"]),
            "generated_at": str(row["generated_at"]),
            "source_mode": str(row["source_mode"]),
            "source_repo_url": str(row["source_repo_url"]),
            "source_branch": str(row["source_branch"]),
            "source_commit": str(row["source_commit"]),
            "mindspore_version_hint": str(row["mindspore_version_hint"]),
            "generated_after_gen_ops": bool(row["generated_after_gen_ops"]),
            "repo_root_hint": str(row["repo_root_hint"]),
            "api_count": int(row["api_count"]),
            "source_repositories": [
                {
                    "name": str(item["name"]),
                    "repo_url": str(item["repo_url"]),
                    "branch": str(item["branch"]),
                    "commit": str(item["commit_hash"]),
                    "source_type": str(item["source_type"]),
                }
                for item in conn.execute(
                    """
                    SELECT name, repo_url, branch, commit_hash, source_type
                    FROM source_repository
                    ORDER BY repo_id
                    """
                ).fetchall()
            ],
        },
    }


def build_api_payload(conn: sqlite3.Connection, api_name: str) -> dict:
    row = fetch_api_row(conn, api_name)
    if row is None:
        return {"ok": False, "error": "api not found", "api_name": api_name}
    api_id = int(row["api_id"])
    return {
        "ok": True,
        "api": {
            "api": str(row["api_name"]),
            "category": str(row["category"]),
            "api_level": str(row["api_level"]),
            "trust_level": str(row["trust_level"]),
            "fact_origin": str(row["fact_origin"]),
            "call_chain_kind": str(row["call_chain_kind"]),
            "resolution_kind": str(row["resolution_kind"]),
            "implementation_type": str(row["implementation_type"]),
            "support_reason_kind": str(row["support_reason_kind"]),
            "alias_of": str(row["alias_of"]),
            "unknown_reason": str(row["unknown_reason"]),
            "summary": str(row["summary"]),
            "primitive": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'support_terminal' ORDER BY ordinal",
                (api_id,),
            ),
            "possible_primitives": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'possible' ORDER BY ordinal",
                (api_id,),
            ),
            "func_op_expands_to": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'func_expanded' ORDER BY ordinal",
                (api_id,),
            ),
            "support_matrix": support_matrix(conn, api_id),
            "flags": scalar_list(conn, "SELECT flag FROM api_flag WHERE api_id = ? ORDER BY ordinal", (api_id,)),
            "path_hints": path_hints(conn, api_id),
            "grad": {
                "mode": str(row["grad_mode"]),
                "differentiable": str(row["grad_differentiable"]),
                "backward_primitives": scalar_list(
                    conn,
                    "SELECT primitive_name FROM api_grad_primitive WHERE api_id = ? ORDER BY ordinal",
                    (api_id,),
                ),
                "impl": [
                    {
                        "primitive": str(item["primitive_name"]),
                        "kind": str(item["kind"]),
                        "path": str(item["path"]),
                        "anchor": str(item["anchor"]),
                        "scope_kind": str(item["scope_kind"]),
                    }
                    for item in conn.execute(
                        "SELECT primitive_name, kind, path, anchor, scope_kind FROM api_grad_impl WHERE api_id = ? ORDER BY ordinal",
                        (api_id,),
                    ).fetchall()
                ],
            },
        },
    }


def build_explain_payload(conn: sqlite3.Connection, api_name: str, evidence_limit: int) -> dict:
    row = fetch_api_row(conn, api_name)
    if row is None:
        return {"ok": False, "error": "api not found", "api_name": api_name}
    api_id = int(row["api_id"])
    return {
        "ok": True,
        "api_core": {
            "api_name": str(row["api_name"]),
            "implementation_type": str(row["implementation_type"]),
            "trust_level": str(row["trust_level"]),
            "fact_origin": str(row["fact_origin"]),
            "call_chain_kind": str(row["call_chain_kind"]),
            "resolution_kind": str(row["resolution_kind"]),
            "support_reason_kind": str(row["support_reason_kind"]),
        },
        "support_matrix": support_matrix(conn, api_id),
        "primitive": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'support_terminal' ORDER BY ordinal",
            (api_id,),
        ),
        "possible_primitives": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'possible' ORDER BY ordinal",
            (api_id,),
        ),
        "func_op_expands_to": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'func_expanded' ORDER BY ordinal",
            (api_id,),
        ),
        "support_targets": [
            {
                "primitive": str(item["primitive_name"]),
                "api_def": str(item["api_def_name"]),
                "op_yaml": str(item["op_yaml"]),
                "op_def_path": str(item["op_def_path"]),
                "origin_kind": str(item["origin_kind"]),
            }
            for item in conn.execute(
                """
                SELECT primitive_name, api_def_name, op_yaml, op_def_path, origin_kind
                FROM api_support_target
                WHERE api_id = ?
                ORDER BY ordinal
                """,
                (api_id,),
            ).fetchall()
        ],
        "grad": {
            "mode": str(row["grad_mode"]),
            "differentiable": str(row["grad_differentiable"]),
            "backward_primitives": scalar_list(
                conn,
                "SELECT primitive_name FROM api_grad_primitive WHERE api_id = ? ORDER BY ordinal",
                (api_id,),
            ),
            "impl": [
                {
                    "primitive": str(item["primitive_name"]),
                    "kind": str(item["kind"]),
                    "path": str(item["path"]),
                    "anchor": str(item["anchor"]),
                    "scope_kind": str(item["scope_kind"]),
                }
                for item in conn.execute(
                    """
                    SELECT primitive_name, kind, path, anchor, scope_kind
                    FROM api_grad_impl
                    WHERE api_id = ?
                    ORDER BY ordinal
                    """,
                    (api_id,),
                ).fetchall()
            ],
        },
        "flags": scalar_list(conn, "SELECT flag FROM api_flag WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "path_hints": path_hints(conn, api_id),
        "warnings": {"summary": str(row["summary"])},
        "evidence_digest": evidence_digest(conn, api_id, evidence_limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the mint API SQLite index.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to mint_api_index.db.")
    parser.add_argument("--evidence-limit", type=int, default=12, help="Max evidence rows for `explain`.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("meta", help="Fetch index build metadata.")

    api_parser = subparsers.add_parser("api", help="Fetch one API record.")
    api_parser.add_argument("--name", required=True, help="Public API path.")

    explain_parser = subparsers.add_parser("explain", help="Fetch one API explanation bundle.")
    explain_parser.add_argument("--name", required=True, help="Public API path.")

    args = parser.parse_args()
    db_path = Path(args.db).resolve()
    unavailable = check_index_available(db_path)
    if unavailable is not None:
        print(json_text(unavailable))
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.command == "meta":
            payload = build_meta_payload(conn)
        elif args.command == "api":
            payload = build_api_payload(conn, args.name)
        else:
            payload = build_explain_payload(conn, args.name, args.evidence_limit)
    finally:
        conn.close()
    print(json_text(payload))


if __name__ == "__main__":
    main()
