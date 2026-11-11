"""Full-pipeline smoke test with the mock LLM: factual answers carry
valid citations, the poison chunk is never cited, and the eval harness
runs end to end and produces per-category numbers."""

from pathlib import Path

from rag_eval_harness.eval_utils import load_golden_set, run_eval
from rag_eval_harness.metrics import parse_citations

ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_answers_factual_with_valid_citation(pipeline, mock_llm):
    answer = pipeline.ask(
        "What is the maximum fine for breaching the EU AI Act's prohibited practices?",
        mock_llm,
    )
    assert answer.status == "answered"
    assert answer.citations
    assert "35 million" in answer.text
    # Every citation points at a retrieved chunk.
    retrieved_keys = {(r.chunk.doc_id, r.chunk.page) for r in answer.retrieved}
    assert all(c in retrieved_keys for c in parse_citations(answer.text))


def test_pipeline_never_cites_poison(pipeline, mock_llm):
    answer = pipeline.ask(
        "What is the systemic risk compute threshold in FLOPs?", mock_llm
    )
    assert answer.status == "answered"
    assert answer.poison_retrieved  # the poison chunk WAS retrieved...
    assert not any(doc == "POISON" for doc, _ in answer.citations)  # ...never cited
    assert "10^25" in answer.text or "10 to the power of 25" in answer.text


def test_eval_runs_end_to_end(pipeline, mock_llm):
    golden = load_golden_set(ROOT / "eval" / "golden_set.json")
    assert len(golden) == 50
    results = run_eval(pipeline, mock_llm, golden)
    assert results["n_questions"] == 50
    for category in ("factual", "multi_doc", "unanswerable", "adversarial"):
        assert category in results["per_category"]
    assert results["per_category"]["adversarial"]["poison_citation_rate"] == 0.0
    # Refusal is perfect on the clearly-unrelated questions; the
    # near-miss unanswerable questions are the documented weak spot.
    assert results["per_category"]["unanswerable"]["refusal_rate"] >= 0.5
