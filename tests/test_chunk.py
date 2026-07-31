import json

from rmbr.chunk import split_json, split_markdown, split_python, split_rst, split_text


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


def test_split_python_separates_top_level_functions():
    text = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
    chunks = split_python(text, chunk_size=800)
    assert any("# def foo" in c and "return 1" in c for c in chunks)
    assert any("# def bar" in c and "return 2" in c for c in chunks)
    # each function is its own chunk, not merged together
    assert not any("return 1" in c and "return 2" in c for c in chunks)


def test_split_python_keeps_class_and_its_methods_together():
    text = "class Server:\n    def start(self):\n        pass\n\n    def stop(self):\n        pass\n"
    chunks = split_python(text, chunk_size=800)
    assert any("# class Server" in c and "def start" in c and "def stop" in c for c in chunks)


def test_split_python_keeps_decorator_attached_to_function():
    text = "@app.route('/health')\ndef health():\n    return 'ok'\n"
    chunks = split_python(text, chunk_size=800)
    assert any("@app.route" in c and "def health" in c for c in chunks)


def test_split_python_keeps_comment_above_function_attached():
    text = "# explains what bar does\ndef bar():\n    pass\n"
    chunks = split_python(text, chunk_size=800)
    assert any("explains what bar does" in c and "def bar" in c for c in chunks)


def test_split_python_no_content_silently_dropped():
    text = (
        '"""Module docstring."""\n'
        "import os\n\n"
        "def foo():\n    return os.getcwd()\n\n"
        "TRAILING = 1\n"
    )
    chunks = split_python(text, chunk_size=800)
    combined = "\n".join(chunks)
    assert "Module docstring" in combined
    assert "import os" in combined
    assert "def foo" in combined
    assert "TRAILING = 1" in combined


def test_split_python_falls_back_to_split_text_on_syntax_error():
    text = "this is not valid python syntax :::: def ("
    chunks = split_python(text, chunk_size=800)
    assert chunks == split_text(text, chunk_size=800)


def test_split_python_falls_back_on_empty_input():
    assert split_python("") == []
    assert split_python("   ") == []


def test_split_python_splits_oversized_function_further():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    text = f"def big():\n{body}\n"
    chunks = split_python(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.startswith("# def big") for c in chunks)


def test_split_json_object_splits_by_top_level_key():
    text = '{"name": "rmbr", "version": "0.2.0"}'
    chunks = split_json(text, chunk_size=800)
    assert any("# 'name'" in c and "rmbr" in c for c in chunks)
    assert any("# 'version'" in c and "0.2.0" in c for c in chunks)
    assert not any("rmbr" in c and "0.2.0" in c for c in chunks)  # separate chunks, not merged


def test_split_json_array_splits_by_element():
    text = '[{"id": 1}, {"id": 2}]'
    chunks = split_json(text, chunk_size=800)
    assert any("# [0]" in c and '"id": 1' in c for c in chunks)
    assert any("# [1]" in c and '"id": 2' in c for c in chunks)


def test_split_json_preserves_unicode():
    text = '{"greeting": "héllo wörld"}'
    chunks = split_json(text, chunk_size=800)
    assert any("héllo wörld" in c for c in chunks)


def test_split_json_falls_back_to_split_text_on_invalid_json():
    text = "this is not { valid json"
    chunks = split_json(text, chunk_size=800)
    assert chunks == split_text(text, chunk_size=800)


def test_split_json_falls_back_on_bare_scalar():
    text = '"just a string"'
    chunks = split_json(text, chunk_size=800)
    assert chunks == split_text(text, chunk_size=800)


def test_split_json_falls_back_on_empty_input():
    assert split_json("") == []
    assert split_json("   ") == []


def test_split_json_splits_oversized_value_further():
    big_value = "x" * 2000
    text = json.dumps({"big": big_value})
    chunks = split_json(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.startswith("# 'big'") for c in chunks)


def test_split_rst_separates_sections_by_underline_heading():
    text = "Introduction\n============\n\nSome intro text.\n\nUsage\n=====\n\nSome usage text.\n"
    chunks = split_rst(text, chunk_size=800)
    assert any("Introduction" in c and "Some intro text" in c for c in chunks)
    assert any("Usage" in c and "Some usage text" in c for c in chunks)
    assert not any("intro text" in c and "usage text" in c for c in chunks)


def test_split_rst_keeps_preamble_before_first_heading():
    text = "A short preamble line.\n\nTitle\n=====\n\nBody text.\n"
    chunks = split_rst(text, chunk_size=800)
    assert any("A short preamble line" in c for c in chunks)


def test_split_rst_falls_back_to_split_text_when_no_heading_detected():
    text = "Just some plain prose with no headings at all, spanning a couple sentences."
    chunks = split_rst(text, chunk_size=800)
    assert chunks == split_text(text, chunk_size=800)


def test_split_rst_does_not_treat_short_underline_as_heading():
    # Underline shorter than the heading text above it doesn't count -
    # avoids false positives on a coincidental short punctuation line.
    text = "A Longer Heading Line\n===\n\nBody text.\n"
    chunks = split_rst(text, chunk_size=800)
    assert chunks == split_text(text, chunk_size=800)


def test_split_rst_falls_back_on_empty_input():
    assert split_rst("") == []
    assert split_rst("   ") == []
