"""Text splitters for turning documents into search-sized chunks.

Five entry points, all returning a plain list of strings (chunking has no
opinion on storage or embeddings, it just cuts text into pieces of a
reasonable size): ``split_text`` for plain text, ``split_markdown`` for
markdown (splits on headers first), ``split_python`` for Python source
(splits on top-level function/class boundaries via the standard library's
``ast`` module), ``split_json`` for JSON (splits on top-level object keys
or array elements via the standard library's ``json`` module), and
``split_rst`` for reStructuredText (splits on section headings, detected
by the underline-punctuation heuristic — see its docstring for what that
does and doesn't cover). Every structure-aware splitter falls back to
``split_text`` on input it can't parse rather than raising — chunking
should never be the thing that breaks ingestion.
"""

from __future__ import annotations

import ast
import json
import re
import string

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


def split_python(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split Python source on top-level function/class boundaries.

    Parses with the standard library's ``ast`` module — no parser
    dependency. Each function or class becomes its own chunk (further
    split by ``split_text`` only if it alone exceeds ``chunk_size``),
    prefixed with a breadcrumb (``"# def handler"`` / ``"# class
    Server"``) so a chunk retrieved on its own still says what it's part
    of. Decorators and any comments immediately above a definition are
    kept attached to it, not silently dropped or split off — everything
    between one definition and the next (imports, module docstring,
    stray comments) rides along with whichever definition follows it.

    Falls back to ``split_text`` on unparseable input (a syntax error, or
    text that just isn't Python) rather than raising — chunking should
    never be the thing that breaks ingestion.
    """
    text = text.strip()
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_text(text, chunk_size, chunk_overlap)
    if not tree.body:
        return split_text(text, chunk_size, chunk_overlap)

    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    pending_start = 0

    def flush(end_line: int, breadcrumb: str) -> None:
        nonlocal pending_start
        source = "\n".join(lines[pending_start:end_line]).strip()
        if source:
            segments.append((breadcrumb, source))
        pending_start = end_line

    for node in tree.body:
        node_end = getattr(node, "end_lineno", None) or node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush(node_end, f"# def {node.name}")
        elif isinstance(node, ast.ClassDef):
            flush(node_end, f"# class {node.name}")
        # Anything else (imports, module docstring, top-level assignments)
        # just accumulates unflushed until the next def/class carries it
        # along, or the final flush below picks up whatever's left.

    if pending_start < len(lines):
        flush(len(lines), "")

    chunks = []
    for breadcrumb, source in segments:
        for piece in split_text(source, chunk_size, chunk_overlap):
            chunks.append(f"{breadcrumb}\n{piece}" if breadcrumb else piece)
    return chunks


def split_json(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split JSON on top-level object keys or array elements.

    Parses with the standard library's ``json`` module — no parser
    dependency. A top-level object becomes one chunk per key (prefixed
    with a ``"# 'key'"`` breadcrumb); a top-level array becomes one chunk
    per element (prefixed ``"# [i]"``); either is further split by
    ``split_text`` only if a single element alone exceeds ``chunk_size``.

    Each chunk is the element re-serialized (``json.dumps(..., indent=2,
    ensure_ascii=False)``), not a literal slice of the original text — so
    whitespace/formatting can differ from the source file, but the actual
    data doesn't. That tradeoff is deliberate: finding exact byte offsets
    for arbitrary nested JSON would need a custom parser, and re-indented
    JSON is more readable in a chunk returned to an LLM anyway.

    Falls back to ``split_text`` on invalid JSON, or JSON whose top level
    is a bare scalar/empty container with nothing structural to split at.
    """
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return split_text(text, chunk_size, chunk_overlap)

    if isinstance(data, dict) and data:
        segments = [(f"# {key!r}", json.dumps({key: value}, indent=2, ensure_ascii=False)) for key, value in data.items()]
    elif isinstance(data, list) and data:
        segments = [(f"# [{i}]", json.dumps(item, indent=2, ensure_ascii=False)) for i, item in enumerate(data)]
    else:
        return split_text(text, chunk_size, chunk_overlap)

    chunks = []
    for breadcrumb, source in segments:
        for piece in split_text(source, chunk_size, chunk_overlap):
            chunks.append(f"{breadcrumb}\n{piece}")
    return chunks


_RST_UNDERLINE_CHARS = set(string.punctuation)


def split_rst(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split reStructuredText on section headings.

    Detects a heading as a non-blank line immediately followed by a line
    of a single repeated punctuation character at least as long as the
    heading text — the common underline-only heading style (``Title`` /
    ``=====``). This is a heuristic, not a full RST parser: it doesn't
    handle overline+underline headings (a punctuation line both above and
    below the title, RST's other valid style) or transitions (a lone
    punctuation-line divider not attached to any heading, which this
    would ignore correctly anyway since there's no heading text above it
    to pair with). Falls back to ``split_text`` whenever no heading is
    detected — including on any text that just isn't RST.

    Each section (heading line plus everything until the next heading) is
    further split by ``split_text`` only if it alone exceeds
    ``chunk_size``; unlike the markdown/Python splitters, sections aren't
    given a synthetic breadcrumb prefix since the heading line itself is
    already the first line of its chunk.
    """
    text = text.strip()
    if not text:
        return []

    lines = text.splitlines()
    heading_starts = [i for i in range(len(lines) - 1) if _is_rst_heading(lines[i], lines[i + 1])]
    if not heading_starts:
        return split_text(text, chunk_size, chunk_overlap)

    boundaries = heading_starts + [len(lines)]
    chunks: list[str] = []
    if boundaries[0] > 0:
        preamble = "\n".join(lines[: boundaries[0]]).strip()
        if preamble:
            chunks.extend(split_text(preamble, chunk_size, chunk_overlap))

    for start, end in zip(boundaries, boundaries[1:]):
        section = "\n".join(lines[start:end]).strip()
        if section:
            chunks.extend(split_text(section, chunk_size, chunk_overlap))

    return chunks


def _is_rst_heading(heading_line: str, underline_line: str) -> bool:
    heading = heading_line.strip()
    underline = underline_line.strip()
    if not heading or not underline or len(underline) < len(heading):
        return False
    chars = set(underline)
    return len(chars) == 1 and chars <= _RST_UNDERLINE_CHARS


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
