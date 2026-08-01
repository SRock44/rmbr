"""Real MCP protocol round-trip latency: an actual `python -m rmbr`
subprocess, spoken to over real stdio by the real `mcp` client SDK - not
`build_mcp_server()` called in-process. Measures what a real MCP client
(Claude Desktop, Claude Code, Cursor) actually experiences per tool call:
subprocess IPC + JSON-RPC framing/parsing on top of the same underlying
`Memory`/`Index` call latency.py already measures in-process.

    python bench/mcp_latency.py

Uses the real default embedder (same reasoning as latency.py): this
measures what a real MCP tool call costs end to end, not an
apples-to-apples engine comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[max(0, int(len(values) * p) - 1)]


def summarize(label: str, samples: list[float]) -> dict:
    result = {
        "label": label,
        "n": len(samples),
        "mean_ms": statistics.mean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99) if len(samples) >= 20 else None,
    }
    p99 = f"{result['p99_ms']:.2f}ms" if result["p99_ms"] is not None else "n/a"
    print(
        f"{label:<32} mean={result['mean_ms']:>8.2f}ms  p50={result['p50_ms']:>8.2f}ms  "
        f"p95={result['p95_ms']:>8.2f}ms  p99={p99}  (n={result['n']})"
    )
    return result


async def run(db_path: Path, corpus_size: int, n_calls: int) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rmbr", str(db_path), "--namespace", "agent"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            t0 = time.perf_counter()
            await session.initialize()
            init_ms = (time.perf_counter() - t0) * 1000
            print(f"session.initialize() (subprocess spawn + handshake): {init_ms:.2f}ms")

            # Pre-load a realistic corpus via remember() itself (real MCP
            # calls, not a backdoor) before measuring steady-state cost.
            for i in range(corpus_size):
                await session.call_tool(
                    "remember", {"text": f"seed memory {i} about topic {i % 50}"}
                )

            remember_samples = []
            for i in range(n_calls):
                t0 = time.perf_counter()
                await session.call_tool("remember", {"text": f"incremental memory {i} about deployment"})
                remember_samples.append((time.perf_counter() - t0) * 1000)

            recall_samples = []
            for i in range(n_calls):
                t0 = time.perf_counter()
                await session.call_tool("recall", {"query": f"topic {i % 50}", "k": 5})
                recall_samples.append((time.perf_counter() - t0) * 1000)

            search_samples = []
            for i in range(n_calls):
                t0 = time.perf_counter()
                await session.call_tool("search", {"query": f"topic {i % 50}", "k": 5})
                search_samples.append((time.perf_counter() - t0) * 1000)

    return {
        "corpus_size": corpus_size,
        "session_initialize_ms": init_ms,
        "remember": summarize("MCP remember tool call", remember_samples),
        "recall": summarize("MCP recall tool call", recall_samples),
        "search": summarize("MCP search tool call", search_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-size", type=int, default=500, help="memories pre-loaded before measuring")
    parser.add_argument("--n-calls", type=int, default=30, help="samples per tool")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    db_path = args.out / "mcp_latency.db"
    db_path.unlink(missing_ok=True)

    print("Real MCP protocol round-trip latency (subprocess + stdio + real mcp client SDK):\n")
    result = asyncio.run(run(db_path, args.corpus_size, args.n_calls))

    out_path = args.out / f"mcp_latency_{int(time.time())}.json"
    out_path.write_text(
        json.dumps({"config": {"corpus_size": args.corpus_size, "n_calls": args.n_calls}, "result": result}, indent=2)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
