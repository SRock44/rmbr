"""Optional adapters for other retrieval/agent frameworks.

Nothing in here is imported by `import rmbr` — each submodule (`langchain`,
`llamaindex`) lazily imports its target framework only when actually used,
so wrapping an `Index` for LangChain never requires LlamaIndex to be
installed, or vice versa, and neither is a hard rmbr dependency. Prefer
`Index.as_langchain_retriever()` / `Index.as_llamaindex_retriever()` over
importing these modules directly — same lazy-import contract, more
discoverable.
"""
