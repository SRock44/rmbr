# Releasing rmbr

## One-time setup (done)

`.github/workflows/publish.yml` publishes to PyPI using [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API
token lives in this repo.

1. **GitHub environment `pypi`** — created (`gh api -X PUT
   repos/SRock44/rmbr/environments/pypi`). The workflow's `publish` job
   runs under this environment; you can add required reviewers to it
   later in Settings → Environments → pypi if you want a manual approval
   gate before every publish.

2. **PyPI trusted publisher — done**, registered on pypi.org for the
   existing `rmbr` project (Manage → Publishing → GitHub publisher:
   owner `SRock44`, repo `rmbr`, workflow `publish.yml`, environment
   `pypi`). Nothing else to do here before cutting a release.

## Cutting a release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/rmbr/__init__.py` (kept in sync manually — no version-sync
   tooling yet, see the gaps list).
2. Merge that to `main` via PR (branch protection requires it).
3. Tag and create a GitHub Release from `main`:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   gh release create v0.1.0 --title "v0.1.0" --generate-notes
   ```
4. Publishing the release triggers `publish.yml` automatically — build,
   `twine check`, then upload via trusted publishing. No manual `twine
   upload` step, and no token to manage or leak.

## Before the v0.1.0 tag specifically

Done: `bench/run.py` (bulk ingest/recall) and `bench/latency.py`
(single-call latency, real embedder) have both been run on the pinned
Ubuntu benchmark machine (3 runs each, 4 cores isolated via `taskset`),
and README.md's Performance section reflects the real numbers, honestly
scoped — single-call latency is the headline, bulk-ingest throughput is
disclosed but not led with, since it's not what rmbr optimizes for. Raw
output for every run is checked in at `bench/pinned/`. Two real bugs
were caught and fixed by these runs: a recall regression in `AnnIndex`'s
default HNSW parameters, and a per-row-commit performance bug in
`store.py` (see `src/rmbr/ann.py`, `src/rmbr/store.py`, and README.md's
Performance section). Re-run both scripts before any future release if
`ann.py`, `search.py`, `store.py`, or `bench/corpus.py` change — the
numbers are load-bearing, not decorative.
