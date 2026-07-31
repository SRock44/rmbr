"""Expose a namespace-pinned Memory + Index over HTTP (REST/JSON) — for
callers that can't hold a persistent stdio connection the way an MCP
client does: serverless functions, a backend that wants `curl`, anything
that just wants to POST some JSON.

    from rmbr import serve_http
    serve_http("agents.db", namespace="coder", read_only=True)

Built on Starlette + uvicorn — not a new dependency: `mcp` (a hard rmbr
dependency, for its own HTTP/SSE transport) already pulls both in, so this
adds zero new packages. `build_app()` returns a plain `Starlette` instance
you can mount into a larger app, wrap in your own middleware (CORS isn't
handled here — add `starlette.middleware.cors.CORSMiddleware` yourself if
you need it), or run directly via `serve_http()`.

**Namespace-pinned**, same principle as `serve_mcp()`: every route operates
on the namespace this server was built for — there's no `namespace` field
in any request body or query string for a caller to fill in, so a client
structurally can't reach another agent's data by asking nicely.

**Auth is opt-in, not automatic.** Pass `token=` (or set `RMBR_TOKEN`) and
every route except `/health` requires `Authorization: Bearer <token>`; leave
both unset and the server has no auth at all — matching rmbr's "you own the
network boundary" posture elsewhere, but worth being deliberate about.
`serve_http()` also binds to `127.0.0.1` by default, not `0.0.0.0` — "it
just works" locally shouldn't also mean "reachable from your whole network"
without you asking for that explicitly.

**Endpoints:**

    GET    /health
    POST   /memories                 remember()   -> {"id": ...}
    GET    /memories?limit=&where=   list()       -> {"results": [...]}
    GET    /memories/{id}            get()        -> record, or 404
    PATCH  /memories/{id}            update()     -> {"status": "updated"}
    DELETE /memories/{id}            forget()     -> 204
    POST   /memories/search          recall()     -> {"results": [...], "timings": {...}}
    GET    /memories/stats           stats()      -> {namespace: {...}}
    POST   /documents                add_text()   -> {"id": ...}
    DELETE /documents/{id}           delete()     -> 204
    GET    /documents/stats          stats()      -> {namespace: {...}}
    POST   /search                   search()     -> {"results": [...], "timings": {...}}

`add_files()` isn't exposed — it reads from this *process's* local
filesystem, which is meaningless for a remote caller with no access to it.
Ingest via `POST /documents` (raw text) instead, or use `Index` directly
in-process if you're already local.
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import __version__
from .embed import Embedder
from .index import Index
from .memory import Memory
from .policy import Policy
from .tools import hit_to_dict

_TOKEN_ENV_VAR = "RMBR_TOKEN"


def build_app(
    path: str,
    *,
    namespace: str = "default",
    policy: Policy | None = None,
    embedder: Embedder | None = None,
    read_only: bool = False,
    token: str | None = None,
) -> Any:
    """Build the configured Starlette app without running it.

    Split out from `serve_http()` so it can be tested in-process (via
    `starlette.testclient.TestClient`) or mounted into a larger ASGI app,
    same reasoning as `build_mcp_server()` vs `serve_mcp()`.
    """
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    mem = Memory(path, namespace, policy=policy, embedder=embedder)
    idx = Index(path, namespace=namespace, policy=policy, embedder=embedder)
    effective_token = token if token is not None else os.environ.get(_TOKEN_ENV_VAR)

    def error_response(exc: Exception) -> JSONResponse:
        if isinstance(exc, PermissionError):
            return JSONResponse({"error": str(exc)}, status_code=403)
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return JSONResponse({"error": str(exc)}, status_code=400)
        raise exc

    async def body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
        except Exception:
            raise ValueError("request body must be valid JSON") from None
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def read_only_guard() -> JSONResponse | None:
        if read_only:
            return JSONResponse({"error": "this server is read-only"}, status_code=405)
        return None

    # -- memories ----------------------------------------------------------

    async def remember(request: Request) -> Response:
        if (guard := read_only_guard()) is not None:
            return guard
        try:
            data = await body(request)
            kwargs: dict[str, Any] = {}
            if "metadata" in data:
                kwargs["metadata"] = data["metadata"]
            if "pinned" in data:
                kwargs["pinned"] = data["pinned"]
            if "dedupe_threshold" in data:
                kwargs["dedupe_threshold"] = data["dedupe_threshold"]
            memory_id = await mem.aremember(data["text"], **kwargs)
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"id": memory_id}, status_code=201)

    async def list_memories(request: Request) -> Response:
        try:
            limit = request.query_params.get("limit")
            where_raw = request.query_params.get("where")
            where = json.loads(where_raw) if where_raw else None
            records = mem.list(limit=int(limit) if limit else None, where=where)
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"results": [_record_to_dict(r) for r in records]})

    async def get_memory(request: Request) -> Response:
        record = mem.get(request.path_params["memory_id"])
        if record is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(_record_to_dict(record))

    async def update_memory(request: Request) -> Response:
        if (guard := read_only_guard()) is not None:
            return guard
        try:
            data = await body(request)
            kwargs = {k: data[k] for k in ("text", "metadata") if k in data}
            await mem.aupdate(request.path_params["memory_id"], **kwargs)
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"status": "updated"})

    async def forget_memory(request: Request) -> Response:
        if (guard := read_only_guard()) is not None:
            return guard
        try:
            await mem.aforget(request.path_params["memory_id"])
        except Exception as exc:
            return error_response(exc)
        return Response(status_code=204)

    async def recall(request: Request) -> Response:
        try:
            data = await body(request)
            hits = await mem.arecall(
                data["query"],
                k=data.get("k", 5),
                where=data.get("where"),
                min_similarity=data.get("min_similarity"),
                recency_weight=data.get("recency_weight", 0.0),
                rerank=data.get("rerank", False),
            )
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"results": [hit_to_dict(h) for h in hits], "timings": hits.timings})

    async def memory_stats(request: Request) -> Response:
        return JSONResponse(mem.stats())

    # -- documents -----------------------------------------------------------

    async def add_document(request: Request) -> Response:
        if (guard := read_only_guard()) is not None:
            return guard
        try:
            data = await body(request)
            document_id = await idx.aadd_text(
                data["text"], source=data.get("source"), metadata=data.get("metadata")
            )
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"id": document_id}, status_code=201)

    async def delete_document(request: Request) -> Response:
        if (guard := read_only_guard()) is not None:
            return guard
        try:
            idx.delete(request.path_params["document_id"])
        except Exception as exc:
            return error_response(exc)
        return Response(status_code=204)

    async def search_documents(request: Request) -> Response:
        try:
            data = await body(request)
            hits = await idx.asearch(
                data["query"],
                k=data.get("k", 5),
                where=data.get("where"),
                min_similarity=data.get("min_similarity"),
                rerank=data.get("rerank", False),
            )
        except Exception as exc:
            return error_response(exc)
        return JSONResponse({"results": [hit_to_dict(h) for h in hits], "timings": hits.timings})

    async def document_stats(request: Request) -> Response:
        return JSONResponse(idx.stats())

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "namespace": namespace, "version": __version__})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/memories", remember, methods=["POST"]),
        Route("/memories", list_memories, methods=["GET"]),
        Route("/memories/search", recall, methods=["POST"]),
        Route("/memories/stats", memory_stats, methods=["GET"]),
        Route("/memories/{memory_id:int}", get_memory, methods=["GET"]),
        Route("/memories/{memory_id:int}", update_memory, methods=["PATCH"]),
        Route("/memories/{memory_id:int}", forget_memory, methods=["DELETE"]),
        Route("/documents", add_document, methods=["POST"]),
        Route("/documents/stats", document_stats, methods=["GET"]),
        Route("/documents/{document_id:int}", delete_document, methods=["DELETE"]),
        Route("/search", search_documents, methods=["POST"]),
    ]

    class _BearerTokenMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Response:
            if request.url.path == "/health":
                return await call_next(request)
            if request.headers.get("authorization") != f"Bearer {effective_token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    middleware = [Middleware(_BearerTokenMiddleware)] if effective_token else []
    return Starlette(routes=routes, middleware=middleware)


def serve_http(
    path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    namespace: str = "default",
    policy: Policy | None = None,
    embedder: Embedder | None = None,
    read_only: bool = False,
    token: str | None = None,
) -> None:
    """Build the app and run it with uvicorn. Blocks until stopped — this is
    meant to be your process's entire job, same as `serve_mcp()`.

    Binds to `127.0.0.1` by default, not `0.0.0.0` — pass that explicitly
    once you've actually decided this should be reachable from outside
    this machine (and set a `token`, or put a real auth layer in front).
    """
    import uvicorn

    app = build_app(
        path, namespace=namespace, policy=policy, embedder=embedder, read_only=read_only, token=token
    )
    uvicorn.run(app, host=host, port=port)


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "namespace": record.namespace,
        "text": record.text,
        "metadata": record.metadata,
        "created_at": record.created_at,
    }
