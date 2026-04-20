# MindSpore mint API Diagnosis Notes

Use this file when stack is `ms` and the failure clearly sits in
`mindspore.mint`, `mindspore.mint.nn`, or `mindspore.mint.nn.functional`
usage, wrapper semantics, or `mint`-specific mode behavior.

Default structured inputs live in:

- `reference/index/mint_api_index.db`
- `reference/index/mint_api_methodology.md`
- `scripts/query_mint_api_index.py`

Optional maintenance-side outputs such as `mint_api_evidence.yaml`,
`mint_api_review_queue.yaml`, `mint_api_index_review.md`,
`mint_api_index_rulebook.md`, and optional `mint_api_index.yaml` exports are
not part of the default runtime read path.

Use this file as the lightweight route into the `mindspore.mint` index. Read
the SQLite index only for `mint`-scoped failures through
`scripts/query_mint_api_index.py`, and use `mint_api_methodology.md` to decide
how much trust a record deserves
before concluding `mint` wrapper semantics or support hints.

For non-`mint` failures, this index is usually not the first thing to read.
Prefer the lightweight MindSpore routing references and source investigation
path first when the failure is outside `mindspore.mint`.

## mint API Layer Hierarchy

Read the issue from top to bottom:

- `mindspore.mint`
- `mindspore.mint.nn`
- `mindspore.mint.nn.functional`
- the wrapped lower-layer call reached by the `mint` entrypoint

If the failure only appears at one layer, do not immediately assume the lower
layer is broken.

Practical read order:

1. confirm whether the symptom is Graph vs PyNative, Ascend vs CPU, or forward vs backward specific
2. confirm that the failing public API really lives under `mindspore.mint`, `mindspore.mint.nn`, or `mindspore.mint.nn.functional`
3. query `reference/index/mint_api_index.db` through `scripts/query_mint_api_index.py meta`
   and compare `mindspore_version_hint`, `source_branch`, and `source_commit`
   with the user's active MindSpore environment when that information is available
4. if the user provides a local MindSpore repo, or enough version or commit
   information to get the matching source tree, regenerate a fresh
   `mint_api_index.db` for that source before trusting the built-in static index
5. query `reference/index/mint_api_index.db` through `scripts/query_mint_api_index.py`
   for the specific `mint` API record
6. read `reference/index/mint_api_methodology.md` when the record is inherited, indirect, or scenario-dependent

If the failing symbol is `mindspore.ops.*`, `mindspore.nn.*`, or a backend
Primitive without a `mint` entrypoint, prefer the general MindSpore route
first instead of starting from the `mint` index.

## mint-Focused Routing Hints

Use this file to decide whether the failure is really `mint`-scoped before
going deeper.

High-value checks:

- compare `GRAPH_MODE` vs `PYNATIVE_MODE`
  - Graph-only failure usually points to wrapper semantics, infer, or graph-safe API usage before backend code
- check view vs copy behavior in `mint` wrappers
  - `view`, `flatten`, `squeeze`, and similar APIs can be mode-sensitive
- check strict validation in `mint.nn`
  - parameter validation often fails before the first real kernel launch
- check wrapper drift first when imports or call signatures changed
  - import-path or callable-shape errors are often API drift, not backend instability

If these checks suggest a broader MindSpore route instead of a `mint`-specific
wrapper issue, move to [mindspore-diagnosis](mindspore-diagnosis.md) rather than
expanding the `mint` index read.

## When to Go Deeper

If the issue still looks like a MindSpore operator or backend implementation
problem after these checks, move to [mindspore-diagnosis](mindspore-diagnosis.md).

Prefer the SQLite index first only for `mindspore.mint`-scoped failures. If
`mint_api_index.db` is missing or `sqlite3` is unavailable, skip the mint
index query instead of falling back to optional YAML exports. In other words,
skip the mint index query and move on to the general MindSpore route. Use the
methodology note to understand whether the record reflects a direct fact,
inherited wrapper behavior, or a scenario-dependent `mint` path.

When the user's active MindSpore version, branch, or commit is known, query
`scripts/query_mint_api_index.py meta` first and treat a clear mismatch as a
reason to lower trust in the static mint index.

If the user provides a local MindSpore codebase, or enough version or commit
information to regenerate against the matching code, prefer rebuilding the mint
index with `scripts/index_builders/generate_mindspore_failure_index.py` over
reasoning from a mismatched static DB snapshot.
