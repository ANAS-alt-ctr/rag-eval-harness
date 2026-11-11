"""Local eval metrics -- no paid eval API required.

All metrics are computed from the question, the gold answer / gold chunk
ids, the retrieved chunks and the generated answer. Two are *lexical
proxies* and are labelled as such in the output, because a word-overlap
check is not a true entailment judge:

* ``faithfulness`` -- fraction of the answer's claims (sentences) whose
  content words substantially overlap a retrieved chunk. A real
  LLM-judge (RAGAS faithfulness) is stronger; see README for how to run
  the same golden set through RAGAS/DeepEval for a second opinion.
* ``answer_relevancy`` -- cosine similarity between the question and the
  answer, using the same embedder as retrieval.

The rest are exact set-based measures and need no proxy:

* ``context_precision`` -- retrieved chunks that are gold-relevant /
  retrieved chunks.
* ``context_recall`` -- gold-relevant chunks retrieved / gold-relevant.
* ``citation_precision`` -- citations in the answer that point at a
  retrieved chunk / all citations in the answer.
* ``refusal_rate`` -- unanswerable questions the system refused / all
  unanswerable questions.
* ``poison_citation_rate`` -- adversarial answers that cite the planted
  poison chunk / all adversarial questions (target: 0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import cosine_similarity
from .retrieval import RetrievedChunk, STOPWORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[([A-Z0-9][A-Z0-9\-]*) p\.(\d+)\]")


def content_words(text: str) -> set[str]:
    """Lower-cased word tokens minus stopwords."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def split_claims(answer: str) -> list[str]:
    """One claim per sentence; citations stripped before splitting."""
    clean = _CITATION_RE.sub("", answer)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    return parts


def parse_citations(answer: str) -> list[tuple[str, int]]:
    """All ``[DOC-ID p.N]`` citations in the answer, in order."""
    return [(doc, int(page)) for doc, page in _CITATION_RE.findall(answer)]


def _claim_supported(claim: str, contexts: list[str], threshold: float = 0.5) -> bool:
    """A claim is supported if at least ``threshold`` of its content words
    appear in any single retrieved chunk (lexical proxy for entailment)."""
    words = content_words(claim)
    if not words:
        return True  # no checkable content (e.g. "Here is the answer:")
    for context in contexts:
        ctx_words = content_words(context)
        if ctx_words and len(words & ctx_words) / len(words) >= threshold:
            return True
    return False


def faithfulness(answer: str, retrieved: list[RetrievedChunk]) -> float:
    """Fraction of claims supported by the retrieved context (lexical
    proxy). Returns 1.0 for a refusal (nothing claimed)."""
    claims = split_claims(answer)
    if not claims:
        return 1.0
    contexts = [r.chunk.text for r in retrieved]
    supported = sum(1 for c in claims if _claim_supported(c, contexts))
    return supported / len(claims)


def answer_relevancy(
    question: str, answer: str, embedder
) -> float:
    """Cosine similarity between the question and the answer embeddings.
    Returns 0.0 for a refusal (no answer to compare)."""
    if not answer.strip():
        return 0.0
    q, a = embedder.embed([question, answer])
    return max(0.0, cosine_similarity(q, a))


def context_precision(retrieved: list[RetrievedChunk], gold_chunk_ids: set[str]) -> float:
    """Retrieved chunks that are gold-relevant / retrieved chunks."""
    if not retrieved:
        return 0.0
    hit = sum(1 for r in retrieved if r.chunk.chunk_id in gold_chunk_ids)
    return hit / len(retrieved)


def context_recall(retrieved: list[RetrievedChunk], gold_chunk_ids: set[str]) -> float:
    """Gold-relevant chunks retrieved / gold-relevant chunks."""
    if not gold_chunk_ids:
        return 1.0
    found = {r.chunk.chunk_id for r in retrieved} & gold_chunk_ids
    return len(found) / len(gold_chunk_ids)


def citation_precision(answer: str, retrieved: list[RetrievedChunk]) -> float:
    """Citations pointing at a retrieved chunk / all citations. An answer
    with no citations scores 0.0 when it makes factual claims, 1.0 when it
    is a refusal (nothing to cite)."""
    citations = parse_citations(answer)
    retrieved_keys = {
        (r.chunk.doc_id, r.chunk.page) for r in retrieved
    }
    if not citations:
        return 1.0 if "don't have information" in answer.lower() else 0.0
    valid = sum(1 for c in citations if c in retrieved_keys)
    return valid / len(citations)


def refusal_rate(refused: list[bool]) -> float:
    """Fraction of questions refused."""
    if not refused:
        return 0.0
    return sum(1 for r in refused if r) / len(refused)


def poison_citation_rate(
    answers: list[str], poison_doc_id: str = "POISON"
) -> float:
    """Fraction of answers that cite the planted poison chunk (target 0)."""
    if not answers:
        return 0.0
    cited = sum(
        1 for a in answers if any(doc == poison_doc_id for doc, _ in parse_citations(a))
    )
    return cited / len(answers)
