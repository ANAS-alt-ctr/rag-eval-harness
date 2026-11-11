"""Build (or rebuild) the vector index from a corpus directory.

Reads every .txt/.md/.pdf under --corpus, chunks it, embeds it and stores
it in the configured vector store (ChromaDB by default, in-memory with
--store memory). Prints the document and chunk counts.

Run:
    python scripts/build_index.py --corpus data/sample_corpus
    python scripts/build_index.py --corpus data/corpus --embedder st --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.config import Settings
from rag_eval_harness.embeddings import make_embedder
from rag_eval_harness.logging import StepLog
from rag_eval_harness.pipeline import RAGPipeline
from rag_eval_harness.store import make_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--embedder", default=None, choices=["st", "hash"])
    parser.add_argument("--store", default=None, choices=["auto", "chroma", "memory"])
    parser.add_argument("--reset", action="store_true", default=True)
    parser.add_argument("--no-reset", dest="reset", action="store_false")
    args = parser.parse_args()

    settings = Settings()
    if args.corpus:
        settings.corpus_dir = args.corpus
    if args.embedder:
        settings.embedder = args.embedder
    if args.store:
        settings.vector_store = args.store

    pipeline = RAGPipeline(
        settings,
        embedder=make_embedder(settings.embedder, settings.embedding_model),
        store=make_store(settings.vector_store, settings.chroma_dir),
        log=StepLog(log_file=None),
    )
    chunks = pipeline.ingest(reset=args.reset)
    docs = sorted({c.doc_id for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents: {docs}")
    print(f"Store: {type(pipeline.store).__name__} ({pipeline.store.count()} vectors)")
    print(f"Embedder: {settings.embedder} ({settings.embedding_model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
