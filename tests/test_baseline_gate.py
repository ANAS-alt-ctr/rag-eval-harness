"""CI gate: the baseline check fails on a real regression and passes
when the numbers match."""

from rag_eval_harness.eval_utils import check_baseline


def _results(**overrides):
    base = {
        "factual": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.8,
            "citation_precision": 1.0,
        },
        "unanswerable": {"refusal_rate": 0.9},
        "adversarial": {"poison_citation_rate": 0.0},
    }
    base.update(overrides)
    return {"per_category": base}


def _baseline():
    return {
        "per_category": {
            "factual": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.8,
                "citation_precision": 1.0,
            },
            "unanswerable": {"refusal_rate": 0.9},
            "adversarial": {"poison_citation_rate": 0.0},
        }
    }


def test_baseline_check_passes_on_match(tmp_path):
    baseline = tmp_path / "baseline.json"
    import json

    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    assert check_baseline(_results(), baseline) == []


def test_baseline_check_fails_on_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    import json

    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    bad = _results(
        factual={
            "faithfulness": 0.5,  # dropped 0.4, far beyond tolerance
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.8,
            "citation_precision": 1.0,
        }
    )
    problems = check_baseline(bad, baseline)
    assert len(problems) == 1
    assert "factual.faithfulness" in problems[0]


def test_baseline_check_fails_on_poison_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    import json

    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    bad = _results(adversarial={"poison_citation_rate": 0.5})
    problems = check_baseline(bad, baseline)
    assert any("poison_citation_rate" in p for p in problems)


def test_baseline_check_tolerates_small_noise(tmp_path):
    baseline = tmp_path / "baseline.json"
    import json

    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    noisy = _results(
        factual={
            "faithfulness": 0.87,  # within the 0.05 tolerance
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.8,
            "citation_precision": 1.0,
        }
    )
    assert check_baseline(noisy, baseline) == []
