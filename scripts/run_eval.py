"""Run the golden-set eval and write per-category metrics.

Usage:
    python scripts/run_eval.py --llm mock                      # no key
    python scripts/run_eval.py --llm openai                    # real model
    python scripts/run_eval.py --llm mock --poison             # inject the
                    # planted contradictory chunk (adversarial test)
    python scripts/run_eval.py --llm openai --check-baseline   # CI gate:
                    # exit 1 when any metric regresses vs eval/baseline.json
    python scripts/run_eval.py --llm openai --out eval/results-real.json

The mock LLM builds its answers from the retrieved context, so mock
numbers measure the plumbing (retrieval, refusal, citations) -- not a
real model's accuracy. Never copy mock numbers into a CV; run with
--llm openai for the numbers that matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.chunking import Chunk
from rag_eval_harness.config import Settings
from rag_eval_harness.embeddings import make_embedder
from rag_eval_harness.eval_utils import check_baseline, load_golden_set, run_eval
from rag_eval_harness.llm import MockLLM, OpenAICompatibleClient
from rag_eval_harness.logging import StepLog
from rag_eval_harness.pipeline import RAGPipeline
from rag_eval_harness.store import make_store

ROOT = Path(__file__).resolve().parent.parent


def _load_poison() -> Chunk:
    data = json.loads((ROOT / "eval" / "poison_chunk.json").read_text(encoding="utf-8"))
    return Chunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        source=data["source"],
        page=data["page"],
        text=data["text"],
        trusted=data["trusted"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"])
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--embedder", default=None, choices=["st", "hash"])
    parser.add_argument("--store", default="memory", choices=["auto", "chroma", "memory"])
    parser.add_argument("--reranker", default="off", choices=["on", "off"])
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.json"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--baseline", default=str(ROOT / "eval" / "baseline.json"))
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    settings = Settings()
    if args.corpus:
        settings.corpus_dir = args.corpus
    if args.embedder:
        settings.embedder = args.embedder
    settings.vector_store = args.store
    settings.reranker = args.reranker
    settings.log_file = str(ROOT / "logs" / "eval-queries.jsonl")

    llm = MockLLM() if args.llm == "mock" else OpenAICompatibleClient(settings)
    pipeline = RAGPipeline(
        settings,
        embedder=make_embedder(settings.embedder, settings.embedding_model),
        store=make_store(settings.vector_store, settings.chroma_dir),
        log=StepLog(log_file=None),
    )
    extra = [_load_poison()] if args.poison else []
    chunks = pipeline.ingest(extra_chunks=extra)
    print(
        f"Indexed {len(chunks)} chunks "
        f"({sum(1 for c in chunks if not c.trusted)} untrusted poison) | "
        f"LLM: {args.llm} | embedder: {settings.embedder} | "
        f"reranker: {settings.reranker}"
    )

    golden = load_golden_set(args.golden)
    results = run_eval(pipeline, llm, golden)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'category':<14}{'n':>3}  {'faith':>6}  {'rel':>6}  "
          f"{'ctxP':>6}  {'ctxR':>6}  {'citP':>6}  {'refuse':>6}")
    for category, m in results["per_category"].items():
        extra_cols = ""
        if category == "adversarial":
            extra_cols = (f"  poison_cited={m.get('poison_citation_rate')} "
                          f"poison_seen={m.get('poison_retrieved_rate')}")
        print(
            f"{category:<14}{m['n']:>3}  {m['faithfulness']:>6.2f}  "
            f"{m['answer_relevancy']:>6.2f}  {m['context_precision']:>6.2f}  "
            f"{m['context_recall']:>6.2f}  {m['citation_precision']:>6.2f}  "
            f"{m['refusal_rate']:>6.2f}{extra_cols}"
        )
    print(f"\nResults written to {out_path}")

    if args.check_baseline:
        problems = check_baseline(results, args.baseline, tolerance=args.tolerance)
        if problems:
            print("\nBASELINE CHECK FAILED -- regressions:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\nBaseline check passed: no metric regressed beyond tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
