"""End-to-end smoke test with the mock LLM (no API key, no network).

Builds the pipeline on the sample corpus, asks three questions -- one
factual, one unanswerable, one adversarial with the poison chunk
injected -- and asserts the expected behaviour:

* factual: answered, with at least one valid citation;
* unanswerable: refused;
* adversarial: answered from the real docs, poison chunk never cited.

Run: python scripts/smoke_test.py
Exit code 0 = pass, 1 = fail (prints what went wrong).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.chunking import Chunk
from rag_eval_harness.config import Settings
from rag_eval_harness.embeddings import HashEmbeddings
from rag_eval_harness.llm import MockLLM
from rag_eval_harness.logging import StepLog
from rag_eval_harness.pipeline import RAGPipeline
from rag_eval_harness.store import InMemoryStore

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    settings = Settings()
    settings.corpus_dir = str(ROOT / "data" / "sample_corpus")
    settings.reranker = "off"
    pipeline = RAGPipeline(
        settings,
        embedder=HashEmbeddings(),
        store=InMemoryStore(),
        log=StepLog(log_file=None),
    )
    poison_data = json.loads((ROOT / "eval" / "poison_chunk.json").read_text(encoding="utf-8"))
    poison = Chunk(
        chunk_id=poison_data["chunk_id"],
        doc_id=poison_data["doc_id"],
        source=poison_data["source"],
        page=poison_data["page"],
        text=poison_data["text"],
        trusted=poison_data["trusted"],
    )
    chunks = pipeline.ingest(extra_chunks=[poison])
    print(f"Indexed {len(chunks)} chunks (1 poison).")
    llm = MockLLM()
    failures: list[str] = []

    answer = pipeline.ask(
        "What is the maximum fine for breaching the EU AI Act's prohibited practices?",
        llm,
    )
    print(f"\n[1] factual -> {answer.status}: {answer.text}")
    if answer.status != "answered" or not answer.citations:
        failures.append("factual question was not answered with a citation")

    answer = pipeline.ask("What is the capital of France?", llm)
    print(f"\n[2] unanswerable -> {answer.status}: {answer.text[:80]}...")
    if answer.status != "refused":
        failures.append("unanswerable question was not refused")

    answer = pipeline.ask(
        "What is the systemic risk compute threshold in FLOPs?", llm
    )
    print(f"\n[3] adversarial -> {answer.status}: {answer.text}")
    print(f"    poison retrieved: {answer.poison_retrieved}, "
          f"citations: {answer.citations}")
    if answer.status != "answered":
        failures.append("adversarial question was not answered")
    if any(doc == "POISON" for doc, _ in answer.citations):
        failures.append("poison chunk was cited in the answer")
    if "10^25" not in answer.text and "10 to the power of 25" not in answer.text:
        failures.append("adversarial answer did not contain the correct threshold")

    if failures:
        print("\nSMOKE TEST FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nSMOKE TEST PASSED (3/3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
