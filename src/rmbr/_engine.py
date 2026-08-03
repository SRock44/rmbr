"""Plumbing shared by Memory and Index.

Both are "a namespace-scoped view over one Store, one embedder, and one
AnnIndex per collection" — this module holds that shared logic (loading/
persisting the vector index, resolving which namespaces a caller may
touch) so memory.py and index.py stay focused on their own public API
instead of duplicating it. Nothing here is part of rmbr's public surface.
"""

from __future__ import annotations

from collections.abc import Callable

from .ann import AnnIndex
from .embed import CachingEmbedder, Embedder, get_shared_fastembed_embedder
from .policy import Policy
from .store import Store


def make_embedder(embedder: Embedder | None, store: Store) -> CachingEmbedder:
    return CachingEmbedder(embedder or get_shared_fastembed_embedder(), store)


def load_ann_index(store: Store, collection: str) -> AnnIndex | None:
    """Restore a collection's vector index from the store's file, if one was ever saved."""
    row = store.get_ann_blob(collection)
    if row is None:
        return None
    blob, dim, metric = row
    return AnnIndex.from_bytes(blob, dim=dim, metric=metric)


def save_ann_index(store: Store, collection: str, ann_index: AnnIndex) -> None:
    store.set_ann_blob(collection, ann_index.to_bytes(), ann_index.dim, ann_index.metric)


def resolve_readable_namespaces(
    policy: Policy,
    who: str,
    requested: str | list[str] | None,
    list_all_namespaces: Callable[[], list[str]],
) -> list[str]:
    """Turn a caller's `namespaces=` argument into a concrete, permission-checked list.

    - ``None`` (the default): just the caller's own namespace.
    - ``"*"``: every namespace present in this collection that the policy
      allows the caller to read (silently narrowed — a wildcard request
      that this policy doesn't fully grant isn't an error, just filtered).
    - an explicit namespace or list: checked against the policy, and a
      disallowed one raises ``PermissionError`` rather than being dropped
      silently, since the caller asked for it by name.
    """
    if requested is None:
        return [who]
    if requested == "*":
        return [ns for ns in list_all_namespaces() if ns == who or policy.can_read(who, ns)]

    candidates = [requested] if isinstance(requested, str) else list(requested)
    for ns in candidates:
        if ns != who and not policy.can_read(who, ns):
            raise PermissionError(f"{who!r} is not allowed to read namespace {ns!r}")
    return candidates


def resolve_writable_namespace(policy: Policy, who: str, target: str | None) -> str:
    """Return the namespace to write into, checking the policy if it differs from `who`."""
    if target is None or target == who:
        return who
    if not policy.can_write(who, target):
        raise PermissionError(f"{who!r} is not allowed to write to namespace {target!r}")
    return target


def check_ann_consistency(ann: AnnIndex | None, expected_ids: list[int]) -> list[str]:
    """Compare a vector index's ids against the SQLite table's ids for the
    same collection. Returns human-readable problems, empty if healthy.

    The two are kept in sync by every write path (`remember`/`add_text`/
    `forget`/`delete`, all inside one `Store.transaction()`), so a mismatch
    here means something touched the file outside rmbr's own code - manual
    SQL, a crash mid-write, a bug - not something that happens in normal
    use. This is a diagnostic over ids only; it never reads memory/chunk
    content, so it isn't policy-gated the way `recall()`/`search()` are.
    """
    ann_ids = set(ann.ids()) if ann is not None else set()
    expected = set(expected_ids)
    problems = []
    missing_vectors = expected - ann_ids
    orphaned_vectors = ann_ids - expected
    if missing_vectors:
        sample = sorted(missing_vectors)[:10]
        problems.append(f"{len(missing_vectors)} row(s) have no vector in the index: {sample}")
    if orphaned_vectors:
        sample = sorted(orphaned_vectors)[:10]
        problems.append(f"{len(orphaned_vectors)} vector(s) in the index have no matching row: {sample}")
    return problems
