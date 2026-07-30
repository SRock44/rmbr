from rmbr.chunk import split_markdown, split_text


def test_split_text_short_text_is_one_chunk():
    assert split_text("hello world", chunk_size=800) == ["hello world"]


def test_split_text_empty_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_respects_paragraph_boundaries():
    text = "Paragraph one is short.\n\nParagraph two is also short."
    chunks = split_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
    assert all(len(c) <= 40 for c in chunks)  # a little slack for overlap merges


def test_split_text_overlap_carries_context_between_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    # the tail of one chunk should reappear at the head of the next
    for a, b in zip(chunks, chunks[1:]):
        assert a[-5:] in b or a.split()[-1] in b.split()[0:3]


def test_split_text_rejects_overlap_bigger_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, chunk_overlap=10)


def test_split_text_hard_cuts_unsplittable_word():
    chunks = split_text("x" * 50, chunk_size=10, chunk_overlap=2)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks).count("x") >= 50  # no characters silently dropped


def test_split_markdown_no_headers_falls_back_to_split_text():
    text = "just plain text, no headers here"
    assert split_markdown(text, chunk_size=800) == [text]


def test_split_markdown_adds_header_breadcrumb():
    text = "# Setup\n\nRun the installer.\n\n## Install\n\nUse pip."
    chunks = split_markdown(text, chunk_size=800)
    assert any("# Setup" in c and "Run the installer." in c for c in chunks)
    assert any("# Setup > ## Install" in c and "Use pip." in c for c in chunks)


def test_split_markdown_preamble_before_first_header_kept():
    text = "intro line\n\n# Section\n\nbody text"
    chunks = split_markdown(text, chunk_size=800)
    assert any("intro line" in c for c in chunks)
