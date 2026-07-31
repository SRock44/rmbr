"""Real end-to-end HTTP test: an actual uvicorn server bound to a real OS
socket, hit with a real `httpx` client over the network stack - not
Starlette's `TestClient` (tests/test_server.py), which talks to the ASGI
app directly in-process and never touches a socket, a port, or uvicorn's
own request-handling code at all.

Skipped entirely if `rmbr.server` doesn't exist yet on whatever branch
this runs against (it ships alongside the optional HTTP server mode) -
this file is written to start running the moment that lands, not to
block until it does. Uses `FakeEmbedder`, so unlike
test_mcp_subprocess.py this needs no network and stays fast. Run
explicitly:

    pytest tests_integration/ -v
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
import uvicorn

pytest.importorskip("rmbr.server")

from rmbr.embed import FakeEmbedder  # noqa: E402
from rmbr.server import build_app  # noqa: E402


@pytest.fixture
def live_server_url(tmp_path):
    app = build_app(str(tmp_path / "agents.db"), namespace="coder", embedder=FakeEmbedder(dimension=16))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn server did not start within 10s"

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_remember_then_recall_over_a_real_socket(live_server_url):
    with httpx.Client(base_url=live_server_url, timeout=10) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["namespace"] == "coder"

        remember = client.post("/memories", json={"text": "user prefers dark mode"})
        assert remember.status_code == 201

        recall = client.post("/memories/search", json={"query": "dark mode"})
        assert recall.status_code == 200
        hits = recall.json()["results"]
        assert len(hits) == 1
        assert "dark mode" in hits[0]["text"]
