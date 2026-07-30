# Releasing rmbr

## One-time setup (already partly done)

`.github/workflows/publish.yml` publishes to PyPI using [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API
token lives in this repo. Two pieces are required before the first real
release:

1. **GitHub environment `pypi`** — created (`gh api -X PUT
   repos/SRock44/rmbr/environments/pypi`). The workflow's `publish` job
   runs under this environment; you can add required reviewers to it
   later in Settings → Environments → pypi if you want a manual approval
   gate before every publish.

2. **PyPI trusted publisher — not yet done, requires the pypi.org UI:**
   `rmbr` already exists on PyPI (the `0.0.1` stub), so this is the
   *existing project* flow, not the "pending publisher for a new
   project" flow:
   - Log in at pypi.org → your projects → **rmbr** → Manage → **Publishing**
   - Add a new publisher → GitHub
     - Owner: `SRock44`
     - Repository: `rmbr`
     - Workflow name: `publish.yml`
     - Environment name: `pypi`
   - Save.

   Nobody but a maintainer with PyPI access to the `rmbr` project can do
   this step — it can't be done via API or by me.

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

Per the claims policy (README.md, docs/PLAN.md): don't add performance
numbers to the README until `bench/run.py` has actually been run on the
project's pinned Linux benchmark machine and produced them. The harness
is built and validated locally, but nobody's run it on real target
hardware yet. v0.1.0 can ship without those numbers (design commitments
only, as README.md currently states) if that's preferred over waiting.
