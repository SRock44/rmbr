"""`python -m rmbr` — a launch shim for MCP clients, not a CLI.

rmbr has no command-line interface: you don't "run rmbr" the way you'd
run a server. This module exists only so an MCP client (Claude Desktop,
another agent's tool-launcher) can start `serve_mcp()` as a subprocess by
pointing at `python -m rmbr`, the same way any other stdio MCP server is
launched.

    python -m rmbr agents.db --namespace coder --read-only
"""

from __future__ import annotations

import argparse

from .mcp_server import serve_mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rmbr",
        description="Launch an rmbr MCP (stdio) server for one namespace of one .db file.",
    )
    parser.add_argument("path", help="Path to the .db file (created if it doesn't exist)")
    parser.add_argument("--namespace", default="default", help="Namespace to expose (default: 'default')")
    parser.add_argument("--read-only", action="store_true", help="Disable the remember tool")
    args = parser.parse_args(argv)

    serve_mcp(args.path, namespace=args.namespace, read_only=args.read_only)


if __name__ == "__main__":
    main()
