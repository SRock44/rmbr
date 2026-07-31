"""Deny-by-default access control between namespaces — the "master harness".

By default, a namespace can only read and write its own data. Reaching
into another namespace requires an explicit grant::

    policy = Policy()                          # deny-by-default (the default)
    policy.allow("supervisor", read="*")       # supervisor can read every namespace
    policy.allow("coder", read="researcher")   # coder can read researcher's namespace only

    mem = Memory("agents.db", namespace="coder", policy=policy)

Checks are a plain dict lookup — deterministic, no LLM call, no network —
because gating memory access is exactly the kind of decision that must not
depend on a model's mood that day.

**Honesty note:** this is an organizational boundary enforced by rmbr's
own code, not a cryptographic one. Anyone with the `.db` file and no
policy attached (or direct SQLite access) can read everything in it. For
hard isolation between agents that don't trust each other, use separate
`.db` files with OS-level file permissions — namespaces inside one file
are for keeping cooperating agents out of each other's way, not for
security against an adversarial one.
"""

from __future__ import annotations

from collections.abc import Callable

ALL_NAMESPACES = "*"

OnAccessCallback = Callable[[str, str, str, bool], bool]


class Policy:
    """A grant table over namespaces, plus an optional custom override callback."""

    def __init__(self) -> None:
        self._reads: dict[str, set[str]] = {}
        self._writes: dict[str, set[str]] = {}
        self._allow_all = False
        self._on_access: OnAccessCallback | None = None

    @classmethod
    def strict(cls) -> Policy:
        """Deny-by-default: a namespace can only read/write itself. This is also the default."""
        return cls()

    @classmethod
    def open(cls) -> Policy:
        """Allow every namespace to read/write every other namespace. Useful for single-agent setups."""
        policy = cls()
        policy._allow_all = True
        return policy

    def allow(
        self,
        namespace: str,
        *,
        read: str | list[str] | None = None,
        write: str | list[str] | None = None,
    ) -> Policy:
        """Grant ``namespace`` read and/or write access to other namespace(s).

        Pass ``"*"`` (or include it in a list) to grant access to every
        namespace. Returns self, so grants can be chained.
        """
        if read is not None:
            self._reads.setdefault(namespace, set()).update(_as_set(read))
        if write is not None:
            self._writes.setdefault(namespace, set()).update(_as_set(write))
        return self

    def on_access(self, callback: OnAccessCallback) -> Policy:
        """Install a custom override: ``callback(who, verb, namespace, default) -> bool``.

        ``default`` is what the grant table would have decided on its
        own — return it unchanged to keep default behavior for cases your
        callback doesn't care about, or return your own bool to override.
        """
        self._on_access = callback
        return self

    def can_read(self, who: str, namespace: str) -> bool:
        return self._check(who, "read", namespace)

    def can_write(self, who: str, namespace: str) -> bool:
        return self._check(who, "write", namespace)

    def _check(self, who: str, verb: str, namespace: str) -> bool:
        if who == namespace:
            default = True
        elif self._allow_all:
            default = True
        else:
            grants = self._reads if verb == "read" else self._writes
            allowed = grants.get(who, set())
            default = namespace in allowed or ALL_NAMESPACES in allowed
        if self._on_access is not None:
            return bool(self._on_access(who, verb, namespace, default))
        return default


def _as_set(value: str | list[str]) -> set[str]:
    return {value} if isinstance(value, str) else set(value)
