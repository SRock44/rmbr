"""Text splitters for turning documents into search-sized chunks.

Two entry points: ``split_text`` for plain text, ``split_markdown`` for
markdown (splits on headers first so a chunk never silently straddles two
unrelated sections). Both return a plain list of strings — chunking has no
opinion on storage or embeddings, it just cuts text into pieces of a
reasonable size.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Tried in order, coarsest first: paragraph breaks, then lines, then
# sentences, then words, then "give up and cut anywhere". This mirrors how
# a human would shorten a passage — keep whole paragraphs if they fit,
# only fall back to cutting mid-sentence as a last resort.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Recursively split ``text`` into chunks of roughly ``chunk_size`` characters.

    Splits on the largest available boundary (paragraph > line > sentence
    > word) so chunks stay coherent, then stitches pieces back together
    with ``chunk_overlap`` characters of shared context between
    consecutive chunks so a fact near a chunk boundary isn't lost.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    pieces = _split_recursive(text, chunk_size)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def split_markdown(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown on headers first, then apply ``split_text`` within each section.

    Each chunk is prefixed with its header breadcrumb (e.g. ``"# Setup >
    ## Install"``) so a chunk retrieved on its own still carries the
    context of where it came from in the document.
    """
    chunks = []
    for breadcrumb, section in _split_by_headers(text):
        for piece in split_text(section, chunk_size, chunk_overlap):
            chunks.append(f"{breadcrumb}\n{piece}" if breadcrumb else piece)
    return chunks


def _split_recursive(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for sep in _SEPARATORS:
        if sep and sep in text:
            parts = text.split(sep)
            pieces = []
            for part in parts:
                if len(part) > chunk_size:
                    pieces.extend(_split_recursive(part, chunk_size))
                elif part:
                    pieces.append(part)
            return pieces
    # No separator matched (single huge word): hard-cut at chunk_size.
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack small pieces into chunks near ``chunk_size``, carrying overlap forward."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        overlap_tail = _tail(current, chunk_overlap)
        current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece
        # A hard-cut piece plus overlap can still exceed chunk_size. Slide a
        # chunk_size window across it, stepping by (chunk_size - overlap) so
        # every character survives into some chunk rather than being
        # truncated away.
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :]
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, overlap: int) -> str:
    return text[-overlap:] if overlap > 0 else ""


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown into (breadcrumb, section_text) pairs at header lines.

    Breadcrumb tracks the current heading path (e.g. "# A > ## B") so
    nested sections keep their ancestry even though we don't build a full
    tree — just a running stack of (level, title).
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []

    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))

    for i, match in enumerate(matches):
        level, title = len(match.group(1)), match.group(2).strip()
        stack = [h for h in stack if h[0] < level] + [(level, title)]
        breadcrumb = " > ".join(f"{'#' * lvl} {t}" for lvl, t in stack)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((breadcrumb, body))

    return sections
