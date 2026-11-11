"""Golden-set loading, eval running and the CI regression gate.

The golden set is ``eval/golden_set.json``: hand-written questions, each
with a category, a short gold answer, the gold-relevant chunk ids in the
sample corpus, and the expected behaviour (``answer`` or ``refuse``).
Adversarial questions also name the poison chunk id that must never be
cited.

Metrics are reported **per category, never as one blended number**. The
CI gate compares the per-category means against ``eval/baseline.json``
and fails (exit code 1) when any metric falls below its baseline by more
than the allowed tolerance -- that is the green-to-red proof that the
gate actually catches regressions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient
from .metrics import (
    answer_relevancy,
    citation_precision,
    context_precision,
    context_recall,
    faithfulness,
    poison_citation_rate,
    refusal_rate,
)
from .pipeline import Answer, RAGPipeline

CATEGORIES = ["factual", "multi_doc", "unanswerable", "adversarial"]


@dataclass
class GoldenQuestion:
    qid: str
    category: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    expected: str = "answer"  # "answer" | "refuse"
    poison_chunk_id: str = ""


def load_golden_set(path: str | Path) -> list[GoldenQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenQuestion(
            qid=item["id"],
            category=item["category"],
            question=item["question"],
            gold_answer=item.get("gold_answer", ""),
            gold_chunk_ids=item.get("gold_chunk_ids", []),
            expected=item.get("expected", "answer"),
            poison_chunk_id=item.get("poison_chunk_id", ""),
        )
        for item in data
    ]


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    status: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    citation_precision: float
    latency_seconds: float
    cost_usd: float
    poison_retrieved: bool
    answer: str


def evaluate_question(
    pipeline: RAGPipeline,
    llm: LLMClient,
    item: GoldenQuestion,
) -> QuestionResult:
    answer: Answer = pipeline.ask(item.question, llm)
    gold = set(item.gold_chunk_ids)
    refused = answer.status == "refused"
    if refused:
        faith = 1.0 if item.expected == "refuse" else 0.0
        relev = 0.0
        cite_prec = 1.0 if item.expected == "refuse" else 0.0
    else:
        faith = faithfulness(answer.text, answer.retrieved)
        relev = answer_relevancy(item.question, answer.text, pipeline.embedder)
        cite_prec = citation_precision(answer.text, answer.retrieved)
    return QuestionResult(
        qid=item.qid,
        category=item.category,
        question=item.question,
        status=answer.status,
        faithfulness=round(faith, 4),
        answer_relevancy=round(relev, 4),
        context_precision=round(context_precision(answer.retrieved, gold), 4),
        context_recall=round(context_recall(answer.retrieved, gold), 4),
        citation_precision=round(cite_prec, 4),
        latency_seconds=round(answer.latency_seconds, 3),
        cost_usd=round(answer.cost_usd, 6),
        poison_retrieved=answer.poison_retrieved,
        answer=answer.text,
    )


def run_eval(
    pipeline: RAGPipeline,
    llm: LLMClient,
    golden: list[GoldenQuestion],
) -> dict:
    """Run the whole golden set; return per-category means + rows."""
    rows: list[QuestionResult] = []
    for item in golden:
        rows.append(evaluate_question(pipeline, llm, item))
    by_category: dict[str, dict] = {}
    for category in CATEGORIES:
        cat_rows = [r for r in rows if r.category == category]
        if not cat_rows:
            continue
        refused = [r.status == "refused" for r in cat_rows]
        by_category[category] = {
            "n": len(cat_rows),
            "faithfulness": round(sum(r.faithfulness for r in cat_rows) / len(cat_rows), 4),
            "answer_relevancy": round(
                sum(r.answer_relevancy for r in cat_rows) / len(cat_rows), 4
            ),
            "context_precision": round(
                sum(r.context_precision for r in cat_rows) / len(cat_rows), 4
            ),
            "context_recall": round(
                sum(r.context_recall for r in cat_rows) / len(cat_rows), 4
            ),
            "citation_precision": round(
                sum(r.citation_precision for r in cat_rows) / len(cat_rows), 4
            ),
            "refusal_rate": round(refusal_rate(refused), 4),
            "median_latency_seconds": round(
                sorted(r.latency_seconds for r in cat_rows)[len(cat_rows) // 2], 3
            ),
            "cost_usd_total": round(sum(r.cost_usd for r in cat_rows), 6),
        }
    adv_rows = [r for r in rows if r.category == "adversarial"]
    if adv_rows:
        by_category["adversarial"]["poison_citation_rate"] = round(
            poison_citation_rate([r.answer for r in adv_rows]), 4
        )
        by_category["adversarial"]["poison_retrieved_rate"] = round(
            sum(1 for r in adv_rows if r.poison_retrieved) / len(adv_rows), 4
        )
    return {
        "model": getattr(llm, "_model", "mock"),
        "n_questions": len(rows),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_category": by_category,
        "rows": [r.__dict__ for r in rows],
    }


def check_baseline(
    results: dict, baseline_path: str | Path, tolerance: float = 0.05
) -> list[str]:
    """Compare per-category metrics against the recorded baseline.

    Returns a list of regression messages (empty = pass). A metric may
    fall by up to ``tolerance`` before it counts as a regression.
    ``refusal_rate`` and ``poison_citation_rate`` are checked in the
    direction that matters: refusal must not fall on unanswerable
    questions, poison citations must not rise.
    """
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    problems: list[str] = []
    base_cats = baseline.get("per_category", {})
    for category, metrics in results.get("per_category", {}).items():
        base = base_cats.get(category)
        if base is None:
            continue
        for key in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "citation_precision",
        ):
            if key in metrics and key in base:
                if metrics[key] < base[key] - tolerance:
                    problems.append(
                        f"{category}.{key}: {metrics[key]} < baseline "
                        f"{base[key]} - {tolerance}"
                    )
        if category == "unanswerable" and "refusal_rate" in metrics:
            if metrics["refusal_rate"] < base.get("refusal_rate", 1.0) - tolerance:
                problems.append(
                    f"unanswerable.refusal_rate: {metrics['refusal_rate']} < "
                    f"baseline {base.get('refusal_rate')} - {tolerance}"
                )
        if category == "adversarial" and "poison_citation_rate" in metrics:
            if metrics["poison_citation_rate"] > base.get(
                "poison_citation_rate", 0.0
            ) + tolerance:
                problems.append(
                    f"adversarial.poison_citation_rate: "
                    f"{metrics['poison_citation_rate']} > baseline "
                    f"{base.get('poison_citation_rate')} + {tolerance}"
                )
    return problems
