"""Eval metrics: faithfulness, relevancy, context precision/recall and
the poison-citation rate, on hand-built inputs."""

from rag_eval_harness.chunking import Chunk
from rag_eval_harness.embeddings import HashEmbeddings
from rag_eval_harness.metrics import (
    answer_relevancy,
    citation_precision,
    context_precision,
    context_recall,
    faithfulness,
    poison_citation_rate,
    refusal_rate,
)
from rag_eval_harness.retrieval import RetrievedChunk


def _retrieved(chunk_id: str, text: str) -> RetrievedChunk:
    doc, _, _ = chunk_id.partition("#")
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc, source="x.txt", page=1, text=text),
        rrf_score=0.03,
    )


def test_faithfulness_supported_claims_score_high():
    retrieved = [
        _retrieved(
            "EU-AI-ACT#8",
            "Breaching the prohibited practices can lead to fines of up to "
            "35 million euros or 7 percent of worldwide annual turnover.",
        )
    ]
    answer = (
        "Breaching the prohibited practices can lead to fines of up to "
        "35 million euros. [EU-AI-ACT p.6]"
    )
    assert faithfulness(answer, retrieved) == 1.0


def test_faithfulness_unsupported_claim_scores_low():
    retrieved = [
        _retrieved(
            "EU-AI-ACT#8",
            "Breaching the prohibited practices can lead to fines of up to "
            "35 million euros or 7 percent of worldwide annual turnover.",
        )
    ]
    answer = "The fine is 100 million euros and the moon is made of cheese."
    assert faithfulness(answer, retrieved) == 0.0


def test_answer_relevancy_same_topic_scores_higher():
    embedder = HashEmbeddings()
    same = answer_relevancy(
        "What is the maximum fine under the EU AI Act?",
        "The maximum fine is 35 million euros under the EU AI Act.",
        embedder,
    )
    other = answer_relevancy(
        "What is the maximum fine under the EU AI Act?",
        "The capital of France is Paris and the weather is warm today.",
        embedder,
    )
    assert same > other


def test_context_precision_and_recall():
    retrieved = [
        _retrieved("EU-AI-ACT#8", "fines"),
        _retrieved("NIST-AI-RMF#0", "framework"),
    ]
    gold = {"EU-AI-ACT#8"}
    assert context_precision(retrieved, gold) == 0.5
    assert context_recall(retrieved, gold) == 1.0
    assert context_recall(retrieved, {"EU-AI-ACT#8", "EU-AI-ACT#9"}) == 0.5
    assert context_precision([], gold) == 0.0


def test_refusal_rate():
    assert refusal_rate([True, True, False]) == 2 / 3
    assert refusal_rate([]) == 0.0


def test_poison_citation_rate():
    answers = [
        "The fine is 35 million euros. [EU-AI-ACT p.6]",
        "The fine is 100 million euros. [POISON p.1]",
    ]
    assert poison_citation_rate(answers) == 0.5
    assert poison_citation_rate(answers[:1]) == 0.0
    assert poison_citation_rate([]) == 0.0
