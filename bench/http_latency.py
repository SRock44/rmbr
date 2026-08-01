"""Real HTTP round-trip latency: an actual uvicorn server bound to a real
OS socket, hit with a real `httpx` client - not Starlette's in-process
`TestClient`. Measures what a real caller (a serverless function,
another process, `curl`) actually experiences per request: real socket
I/O + HTTP framing on top of the same underlying `Memory`/`Index` call
latency.py already measures in-process.

    python bench/http_latency.py

Uses the real default embedder (same reasoning as latency.py / mcp_latency.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from rmbr.server import build_app  # noqa: E402


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


def start_server(db_path: Path) -> tuple[uvicorn.Server, threading.Thread, str]:
    app = build_app(str(db_path), namespace="agent")
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn server did not start within 15s")

    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def run(db_path: Path, corpus_size: int, n_calls: int) -> dict:
    db_path.unlink(missing_ok=True)
    server, thread, base_url = start_server(db_path)
    try:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            for i in range(corpus_size):
                r = client.post("/memories", json={"text": f"seed memory {i} about topic {i % 50}"})
                r.raise_for_status()

            remember_samples = []
            for i in range(n_calls):
                t0 = time.perf_counter()
                r = client.post("/memories", json={"text": f"incremental memory {i} about deployment"})
                r.raise_for_status()
                remember_samples.append((time.perf_counter() - t0) * 1000)

            recall_samples = []
            for i in range(n_calls):
                t0 = time.perf_counter()
                r = client.post("/memories/search", json={"query": f"topic {i % 50}", "k": 5})
                r.raise_for_status()
                recall_samples.append((time.perf_counter() - t0) * 1000)

            get_samples = []
            memory_id = client.post("/memories", json={"text": "for GET benchmark"}).json()["id"]
            for _ in range(n_calls):
                t0 = time.perf_counter()
                r = client.get(f"/memories/{memory_id}")
                r.raise_for_status()
                get_samples.append((time.perf_counter() - t0) * 1000)
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    return {
        "corpus_size": corpus_size,
        "post_memories": summarize("POST /memories", remember_samples),
        "post_memories_search": summarize("POST /memories/search", recall_samples),
        "get_memories_id": summarize("GET /memories/{id}", get_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-size", type=int, default=500, help="memories pre-loaded before measuring")
    parser.add_argument("--n-calls", type=int, default=30, help="samples per endpoint")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    db_path = args.out / "http_latency.db"

    print("Real HTTP round-trip latency (real uvicorn socket + real httpx client):\n")
    result = run(db_path, args.corpus_size, args.n_calls)

    out_path = args.out / f"http_latency_{int(time.time())}.json"
    out_path.write_text(
        json.dumps({"config": {"corpus_size": args.corpus_size, "n_calls": args.n_calls}, "result": result}, indent=2)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
