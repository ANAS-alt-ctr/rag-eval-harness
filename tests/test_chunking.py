"""Chunking: headings become pages, chunks are bounded and overlapping,
ids are stable, and fixture files starting with '_' never load."""

from pathlib import Path

from rag_eval_harness.chunking import (
    Chunk,
    chunk_corpus,
    chunk_document,
    extract_text_file,
    load_corpus,
    load_document,
)


def test_text_file_headings_become_pages(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text(
        "# Title\nSome intro text here for the first section.\n"
        "## 1. First section\nContent of the first section goes here.\n"
        "## 2. Second section\nContent of the second section goes here.\n",
        encoding="utf-8",
    )
    pages = extract_text_file(path)
    assert len(pages) >= 2
    assert all(text.strip() for _, text in pages)


def test_chunks_bounded_and_overlapping(chunks: list[Chunk]):
    assert len(chunks) == 35  # sample corpus, chunk_size 800 / overlap 150
    for chunk in chunks:
        assert len(chunk.text) <= 1000  # 800 + one sentence of slack
        assert len(chunk.text) >= 40
        assert chunk.chunk_id.count("#") == 1


def test_chunk_ids_stable_and_unique(chunks: list[Chunk]):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "EU-AI-ACT#8" in ids  # the fines chunk the golden set cites
    assert "OWASP-TOP10-LLM#1" in ids


def test_corpus_skips_underscore_files(tmp_path: Path):
    (tmp_path / "_hidden").mkdir()
    (tmp_path / "_hidden" / "secret.txt").write_text(
        "This must never load into the corpus.", encoding="utf-8"
    )
    (tmp_path / "real.txt").write_text("Real content here.", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["REAL"]


def test_pdf_extraction(tmp_path: Path):
    pytest = __import__("pytest")
    try:
        from pypdf import PdfWriter
    except ImportError:
        pytest.skip("pypdf not installed")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)
    doc = load_document(path)
    assert doc.doc_id == "BLANK"
    assert doc.pages == []  # blank page has no text


def test_long_section_splits_at_sentences():
    from rag_eval_harness.chunking import Document

    text = " ".join(f"Sentence number {i} about governance and risk." for i in range(60))
    doc = Document(doc_id="TEST", source="test.txt", pages=[(1, text)])
    chunks = chunk_document(doc, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 400
