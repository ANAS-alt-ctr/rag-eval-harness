"""Citations: parsing, validation against retrieved chunks, and the
rule that a citation to a chunk that was not retrieved is stripped."""

from rag_eval_harness.chunking import Chunk
from rag_eval_harness.metrics import citation_precision, parse_citations
from rag_eval_harness.pipeline import _validate_citations
from rag_eval_harness.retrieval import RetrievedChunk


def _retrieved(doc_id: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc_id}#{page}",
            doc_id=doc_id,
            source=f"{doc_id.lower()}.txt",
            page=page,
            text="some text",
        ),
        rrf_score=0.03,
    )


def test_parse_citations():
    text = "Fines reach 35 million euros [EU-AI-ACT p.6]. See also [NIST-AI-RMF p.2]."
    assert parse_citations(text) == [("EU-AI-ACT", 6), ("NIST-AI-RMF", 2)]
    assert parse_citations("No citations here.") == []


def test_valid_citations_kept():
    retrieved = [_retrieved("EU-AI-ACT", 6)]
    text = "Fines reach 35 million euros [EU-AI-ACT p.6]."
    cleaned, valid, removed = _validate_citations(text, retrieved)
    assert removed == 0
    assert valid == [("EU-AI-ACT", 6)]
    assert "[EU-AI-ACT p.6]" in cleaned


def test_invalid_citation_stripped():
    retrieved = [_retrieved("EU-AI-ACT", 6)]
    text = "Fines reach 100 million euros [POISON p.1] and also 35 million [EU-AI-ACT p.6]."
    cleaned, valid, removed = _validate_citations(text, retrieved)
    assert removed == 1
    assert valid == [("EU-AI-ACT", 6)]
    assert "POISON" not in cleaned
    assert "[EU-AI-ACT p.6]" in cleaned


def test_citation_precision_metric():
    retrieved = [_retrieved("EU-AI-ACT", 6), _retrieved("NIST-AI-RMF", 2)]
    assert citation_precision(
        "A claim [EU-AI-ACT p.6] and another [NIST-AI-RMF p.2].", retrieved
    ) == 1.0
    assert citation_precision(
        "A claim [EU-AI-ACT p.6] and a bad one [POISON p.1].", retrieved
    ) == 0.5
    # No citations on a factual answer scores 0; a refusal scores 1.
    assert citation_precision("A claim with no citation.", retrieved) == 0.0
    assert citation_precision(
        "I don't have information about that in the provided documents.", retrieved
    ) == 1.0
