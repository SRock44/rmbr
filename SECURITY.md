# Security

rmbr is maintained solo, same as noted in [CONTRIBUTING.md](CONTRIBUTING.md) — response times may vary, but reports are taken seriously and acknowledged as fast as possible.

## Reporting a vulnerability

**Don't open a public GitHub issue for a security vulnerability.** Use [GitHub's private security advisory form](https://github.com/SRock44/rmbr/security/advisories/new) instead — it reaches the maintainer directly without disclosing the issue publicly before a fix is out.

Include what you'd include in a good bug report: what's affected, how to reproduce it, and what you think the impact is (a memory-namespace isolation bypass and a docstring typo are both "security" in the broadest sense, but not the same urgency).

## Scope

What's actually worth reporting here, given what rmbr is:

- **Namespace isolation bypasses** — `Policy` is deny-by-default and MCP/HTTP tool schemas expose no `namespace` parameter on purpose (see README's [Multi-agent isolation](README.md#multi-agent-isolation-honestly-stated) section). A way to read or write another namespace's data without an explicit grant is a real vulnerability.
- **Injection in the search/storage path** — anything that lets untrusted input (a document being indexed, a memory being remembered, a search query) execute code, corrupt the SQLite file, or escape its intended scope.
- **Auth bypass in `serve_http()`** — a way to reach a token-protected route without the token.

What's explicitly **not** a namespace-isolation bug, because it's already documented as out of scope: `Policy` is an organizational boundary enforced by rmbr's own code, not a cryptographic one — anyone with direct access to the `.db` file (or a policy-free `Memory`/`Index` handle on it) can already read everything in it. That's true of every embedded database, and it's disclosed, not a bug to report.

## Supported versions

Only the latest published release on [PyPI](https://pypi.org/project/rmbr/) is supported. There's no long-term-support branch — given the pace and size of this project, backporting fixes to old versions isn't realistic; upgrading is the fix.
