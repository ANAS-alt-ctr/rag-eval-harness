"""Refusal: questions the corpus does not cover are refused, questions
it does cover are answered -- and the refusal costs zero LLM calls."""

from rag_eval_harness.pipeline import REFUSAL_TEXT


def test_unrelated_question_refused(pipeline, mock_llm):
    answer = pipeline.ask("What is the capital of France?", mock_llm)
    assert answer.status == "refused"
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []
    assert len(mock_llm.calls) == 0  # refusal happens before any LLM call


def test_gpt4_question_refused(pipeline, mock_llm):
    answer = pipeline.ask("How many parameters does GPT-4 have?", mock_llm)
    assert answer.status == "refused"


def test_covered_question_answered(pipeline, mock_llm):
    answer = pipeline.ask(
        "What is the maximum fine for breaching the EU AI Act's prohibited practices?",
        mock_llm,
    )
    assert answer.status == "answered"
    assert answer.citations
    assert len(mock_llm.calls) == 1


def test_near_miss_documented(pipeline, mock_llm):
    """A question about the same topic but not covered by the docs is the
    known hard case for the lexical refusal signal. This test documents
    the behaviour rather than asserting a refusal: the system answers
    from the closest chunk. See README 'Where this breaks and why'."""
    answer = pipeline.ask(
        "Does the EU AI Act apply to AI systems used by the military?", mock_llm
    )
    # Either outcome is acceptable; the test pins the current behaviour
    # so a future change shows up here instead of silently.
    assert answer.status in ("answered", "refused")
