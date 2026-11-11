"""FastAPI backend for the RAG eval harness.

Run:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health   -- liveness + corpus/model status
    GET  /corpus   -- documents and chunk count in the index
    POST /ask      -- {"question": "..."} -> full Answer JSON
    POST /rebuild  -- rebuild the index from the configured corpus
    GET  /metrics  -- session totals (questions, tokens, cost, refusals)

The LLM is configured ONLY via environment variables (see .env.example).
No key is ever read from the request body. Without a key the API runs
with the mock LLM and says so in /health -- it answers from the sample
corpus, which is enough to demo the retrieval, refusal and citation
paths.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.config import Settings
from rag_eval_harness.llm import MockLLM, OpenAICompatibleClient
from rag_eval_harness.pipeline import RAGPipeline

settings = Settings()
_started_at = time.time()
_stats = {"questions": 0, "answered": 0, "refused": 0, "failed": 0,
          "tokens": 0, "cost_usd": 0.0}


def _build_llm():
    """Real client when a key is configured, mock otherwise (clearly labelled)."""
    if settings.llm_api_key:
        return OpenAICompatibleClient(settings)
    return MockLLM()


def _build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(settings)
    pipeline.ingest()
    return pipeline


_pipeline = _build_pipeline()
_llm = _build_llm()

app = FastAPI(
    title="RAG System with Failure Analysis + LLM Eval Harness",
    description="Cited answers over AI-governance documents, with a refusal "
    "path and a golden-set eval harness.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _started_at, 1),
        "chunks_indexed": _pipeline.store.count(),
        "documents": sorted({c.doc_id for c in _pipeline.chunks}),
        "llm": "real" if settings.llm_api_key else "mock (no LLM_API_KEY set)",
        "model": settings.llm_model,
        "embedder": type(_pipeline.embedder).__name__,
        "reranker": settings.reranker,
    }


@app.get("/corpus")
def corpus() -> dict:
    docs: dict[str, dict] = {}
    for chunk in _pipeline.chunks:
        docs.setdefault(chunk.doc_id, {"source": chunk.source, "chunks": 0})
        docs[chunk.doc_id]["chunks"] += 1
    return {"documents": docs, "total_chunks": len(_pipeline.chunks)}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    answer = _pipeline.ask(request.question, _llm)
    _stats["questions"] += 1
    _stats[answer.status] = _stats.get(answer.status, 0) + 1
    _stats["tokens"] += answer.tokens_used
    _stats["cost_usd"] += answer.cost_usd
    return {
        "question": answer.question,
        "answer": answer.text,
        "status": answer.status,
        "citations": [f"{doc} p.{page}" for doc, page in answer.citations],
        "retrieved": [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "page": r.chunk.page,
                "trusted": r.chunk.trusted,
                "rrf_score": round(r.rrf_score, 4),
                "rerank_score": r.rerank_score,
                "text": r.chunk.text[:400],
            }
            for r in answer.retrieved
        ],
        "poison_retrieved": answer.poison_retrieved,
        "reranked": answer.reranked,
        "tokens_used": answer.tokens_used,
        "cost_usd": round(answer.cost_usd, 6),
        "latency_seconds": round(answer.latency_seconds, 3),
    }


@app.post("/rebuild")
def rebuild() -> dict:
    chunks = _pipeline.ingest(reset=True)
    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.get("/metrics")
def metrics() -> dict:
    return dict(_stats, uptime_seconds=round(time.time() - _started_at, 1))
