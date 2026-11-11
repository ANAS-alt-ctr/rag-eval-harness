"""Shared fixtures: sample corpus, hash embedder, in-memory store, mock
LLM and a ready pipeline. No API key, no network, no model downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.chunking import Chunk, chunk_corpus, load_corpus
from rag_eval_harness.config import Settings
from rag_eval_harness.embeddings import HashEmbeddings
from rag_eval_harness.llm import MockLLM
from rag_eval_harness.logging import StepLog
from rag_eval_harness.pipeline import RAGPipeline
from rag_eval_harness.store import InMemoryStore

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return ROOT / "data" / "sample_corpus"


@pytest.fixture(scope="session")
def chunks(corpus_dir: Path) -> list[Chunk]:
    docs = load_corpus(corpus_dir)
    return chunk_corpus(docs, chunk_size=800, overlap=150)


@pytest.fixture(scope="session")
def embedder() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture(scope="session")
def poison_chunk() -> Chunk:
    data = json.loads((ROOT / "eval" / "poison_chunk.json").read_text(encoding="utf-8"))
    return Chunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        source=data["source"],
        page=data["page"],
        text=data["text"],
        trusted=data["trusted"],
    )


@pytest.fixture()
def pipeline(chunks: list[Chunk], embedder: HashEmbeddings, poison_chunk: Chunk) -> RAGPipeline:
    """A fresh pipeline per test (in-memory store, hash embeddings, no
    reranker, no log file). The poison chunk is included so adversarial
    behaviour is tested by default."""
    settings = Settings()
    settings.corpus_dir = str(ROOT / "data" / "sample_corpus")
    settings.reranker = "off"
    pipe = RAGPipeline(
        settings,
        embedder=embedder,
        store=InMemoryStore(),
        log=StepLog(log_file=None),
    )
    pipe.ingest(extra_chunks=[poison_chunk])
    return pipe


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()
