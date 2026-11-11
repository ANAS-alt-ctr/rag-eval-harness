"""Document ingestion and chunking.

Sources are plain text (``.txt`` / ``.md``) or PDF (``.pdf``, read with
pypdf). Every chunk carries its source document id, the page it came from
(PDF page number; for text files the section number is used as the page),
and a stable id of the form ``<doc-id>#<n>``.

Files or directories whose name starts with ``_`` are skipped: that is
where test fixtures (e.g. the poisoned chunk used by the eval) live, so
they can never leak into the real corpus by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """One source document, split into numbered pages of text."""

    doc_id: str
    source: str  # file name as shown in citations, e.g. "eu-ai-act.pdf"
    pages: list[tuple[int, str]]  # (page number, text)
    trusted: bool = True


@dataclass
class Chunk:
    """One retrievable unit of text."""

    chunk_id: str  # "<doc-id>#<n>"
    doc_id: str
    source: str
    page: int
    text: str
    trusted: bool = True

    def citation(self) -> str:
        """How this chunk is cited in answers: [DOC-ID p.N]."""
        return f"[{self.doc_id} p.{self.page}]"


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Extract text per page from a PDF. Raises on unreadable files."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def extract_text_file(path: Path) -> list[tuple[int, str]]:
    """Split a text/markdown file into pages at its headings.

    A heading line (``# ...`` or an all-caps / numbered section line) starts
    a new page, so citations point at a recognisable section rather than an
    arbitrary character offset. Files without headings become one page.
    """
    raw = path.read_text(encoding="utf-8")
    sections: list[tuple[int, str]] = []
    current: list[str] = []
    page_no = 1
    for line in raw.splitlines():
        is_heading = line.startswith("#") or re.match(
            r"^\d+(\.\d+)*\s+[A-Z]", line.strip()
        )
        if is_heading and current and "".join(current).strip():
            sections.append((page_no, "\n".join(current).strip()))
            page_no += 1
            current = [line]
        else:
            current.append(line)
    if "".join(current).strip():
        sections.append((page_no, "\n".join(current).strip()))
    return sections or [(1, raw.strip())]


def load_document(path: Path) -> Document:
    """Load one file into a Document. ``doc_id`` is the file stem,
    upper-cased with underscores replaced by hyphens (e.g.
    ``eu-ai-act.pdf`` -> ``EU-AI-ACT``)."""
    doc_id = path.stem.upper().replace("_", "-")
    if path.suffix.lower() == ".pdf":
        pages = extract_pdf(path)
    else:
        pages = extract_text_file(path)
    return Document(doc_id=doc_id, source=path.name, pages=pages)


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every supported file under ``corpus_dir`` (sorted, so chunk ids
    are deterministic). Files/dirs starting with ``_`` are skipped."""
    docs: list[Document] = []
    if not corpus_dir.exists():
        return docs
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith("_") for part in path.relative_to(corpus_dir).parts):
            continue
        if path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        docs.append(load_document(path))
    return docs


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    doc: Document, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    Chunks are cut at sentence boundaries where possible, never exceed
    ``chunk_size`` characters by much, and overlap by ``overlap``
    characters so a fact straddling a boundary is still retrievable.
    Chunks shorter than 40 characters are dropped (headings alone).
    """
    chunks: list[Chunk] = []
    n = 0
    for page_no, page_text in doc.pages:
        text = " ".join(page_text.split())
        if len(text) < 40:
            continue
        sentences = _SENTENCE_END.split(text)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > chunk_size and current:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{n}",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        page=page_no,
                        text=current,
                        trusted=doc.trusted,
                    )
                )
                n += 1
                # Overlap: carry the tail of the finished chunk forward.
                tail = current[-overlap:] if overlap > 0 else ""
                tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
                current = f"{tail} {sentence}".strip() if tail else sentence
            else:
                current = candidate
        if current.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{n}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    page=page_no,
                    text=current,
                    trusted=doc.trusted,
                )
            )
            n += 1
    return chunks


def chunk_corpus(
    docs: list[Document], chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Chunk every document in a corpus, in order."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return out
