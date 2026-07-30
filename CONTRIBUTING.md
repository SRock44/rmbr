# Contributing to rmbr

Thanks for considering it. rmbr is currently maintained solo, so response
times may vary — but PRs and issues are genuinely welcome.

## Setup

```bash
git clone https://github.com/SRock44/rmbr.git
cd rmbr
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install --only-binary :all: -e ".[bench]"
pytest tests/
```

`--only-binary :all:` is intentional, not optional: rmbr promises no
build toolchain is required to install it, and this flag is how we catch
a dependency that's quietly stopped shipping a wheel for some platform
before a user does. If it fails for you locally, that's a real bug worth
reporting, not something to work around with a source build.

The test suite runs fully offline — every test uses
`rmbr.embed.FakeEmbedder`, a deterministic embedder with no model
download and no network access. If you add a test that needs real
semantic behavior, prefer constructing vectors by hand (see
`tests/test_query_cache.py`'s `MappedEmbedder` for an example) over
pulling in the real ONNX model — it keeps CI fast and CI-runner-agnostic.

## Before opening a PR

- `pytest tests/` passes locally.
- New behavior has a test. A bug fix should include a test that fails
  without the fix.
- Docstrings on anything public explain what a *caller* needs to know
  (arguments, return shape, gotchas) — not what the code obviously does.
  Inline comments are for the non-obvious *why*, not a restatement of the
  next line.
- No performance claims in README.md that `bench/run.py` didn't produce
  — see the "Performance claims policy" section there. This is a hard
  rule for this project, not a style preference.

## Workflow

`main` is protected: changes land via pull request, and CI
(`.github/workflows/ci.yml`) must pass before merging. Fork or branch,
push, open a PR. Keep PRs focused — several small PRs beat one that
touches storage, search, and the MCP server at once, both for review and
for `git bisect` later.

## Reporting bugs / proposing features

Open a GitHub issue. For bugs, a minimal reproduction (a failing test is
ideal) is worth more than a long description.

## Scope

Things that are explicitly *not* the direction of this project, so you
don't spend time on a PR that won't land:

- A CLI. `python -m rmbr` exists solely as an MCP launch shim.
- A hosted/cloud service or consumer app. rmbr is an embedded library.
- Calling an LLM from inside rmbr. It returns text; the caller decides
  what model sees it.
- Dozens of embedding providers. A handful, deliberately.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design
rationale.
