import sys
from unittest.mock import patch

import pytest

from rmbr.extract import EXTRACTORS, extract_docx, extract_pdf


def _make_minimal_pdf(text: str) -> bytes:
    """Hand-rolled, minimal-but-valid single-page PDF with one text line -
    no reportlab dependency needed just to generate a test fixture. Byte
    offsets are computed from the actual bytes written, not guessed.
    """
    stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


def test_extract_pdf_returns_page_text(tmp_path):
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.write_bytes(_make_minimal_pdf("the deployment guide covers docker"))

    text = extract_pdf(pdf_path)
    assert "the deployment guide covers docker" in text


def test_extract_pdf_missing_extra_raises_clearly(tmp_path):
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.write_bytes(_make_minimal_pdf("x"))

    with patch.dict(sys.modules, {"pypdf": None}):
        with pytest.raises(ImportError, match="rmbr\\[pdf\\]"):
            extract_pdf(pdf_path)


def test_extract_docx_returns_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    docx_path = tmp_path / "notes.docx"

    document = docx.Document()
    document.add_paragraph("the deployment guide covers docker")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "key"
    table.rows[0].cells[1].text = "value"
    document.save(str(docx_path))

    text = extract_docx(docx_path)
    assert "the deployment guide covers docker" in text
    assert "key\tvalue" in text


def test_extract_docx_missing_extra_raises_clearly(tmp_path):
    docx_path = tmp_path / "notes.docx"
    docx_path.write_bytes(b"not a real docx")  # never opened - ImportError fires first

    with patch.dict(sys.modules, {"docx": None}):
        with pytest.raises(ImportError, match="rmbr\\[docx\\]"):
            extract_docx(docx_path)


def test_extractors_registry_covers_pdf_and_docx():
    assert set(EXTRACTORS) == {".pdf", ".docx"}
