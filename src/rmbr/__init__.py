"""rmbr — embedded, local-first memory + retrieval for AI agents.

    from rmbr import Memory, Index, Policy

    mem = Memory("agents.db", namespace="researcher")
    mem.remember("user prefers dark mode and short answers")
    mem.recall("user preferences")

    idx = Index("agents.db")
    idx.add_files("docs/")
    idx.search("how do I deploy?", k=5)

Source and design docs: https://github.com/SRock44/rmbr
"""

__version__ = "0.2.3"

from .index import Index
from .mcp_server import serve_mcp
from .memory import Memory
from .policy import Policy

__all__ = ["Memory", "Index", "Policy", "serve_mcp"]
