"""The RAG pipeline: ingest once, then answer questions.

Flow per question:

1. Hybrid retrieval (BM25 + dense, RRF fusion) -> top candidates.
2. Source allow-list: only chunks from the trusted corpus may be used to
   generate an answer. An untrusted chunk that retrieval surfaced (the
   eval's planted poison chunk) is logged as ``poison_retrieved`` and
   excluded from the generation context -- it can never be cited.
3. Refusal: if the question's rare words (appearing in at most 3 chunks
   or in none) are not covered by the retrieved chunks, or both the
   fused RRF score and the dense cosine are weak, or no trusted chunk
   survived the allow-list, the system refuses ("not in my docs")
   instead of hallucinating.
4. Optional cross-encoder reranking of the trusted candidates.
5. Cited answer generation: the LLM gets the trusted chunks with their
   citation markers and is instructed to answer ONLY from them, citing
   every factual claim as ``[DOC-ID p.N]``.
6. Citation validation: citations in the answer are checked against the
   retrieved chunks. A citation to a chunk that was not retrieved (or to
   an untrusted chunk) is stripped from the answer and counted.

Every query is logged as one JSON object (question, retrieved chunks +
scores, answer, citations, tokens, latency, cost) to the JSONL log file
and, when enabled, to Langfuse.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_corpus, load_corpus
from .config import Settings
from .embeddings import EmbeddingModel, make_embedder
from .llm import LLMClient, LLMResponse, MockLLM
from .logging import StepLog
from .metrics import parse_citations
from .retrieval import HybridRetriever, Reranker, RetrievedChunk
from .store import VectorStore, make_store
from .tracing import LangfuseTracer

REFUSAL_TEXT = (
    "I don't have information about that in the provided documents. "
    "The documents I can answer from are the EU AI Act, the NIST AI Risk "
    "Management Framework, and the OWASP Top 10 for LLM Applications. "
    "Try rephrasing your question around one of those."
)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using ONLY the context chunks below. "
    "Every factual claim in your answer MUST end with a citation in the "
    "exact form [DOC-ID p.N], using the citation shown with each chunk. "
    "If the context does not contain the answer, say you don't have "
    "information about that in the provided documents. Do not use any "
    "knowledge outside the context. Keep the answer to 2-4 sentences."
)


@dataclass
class Answer:
    """The result of one question."""

    question: str
    text: str
    status: str  # "answered" | "refused" | "failed"
    citations: list[tuple[str, int]] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    poison_retrieved: bool = False
    reranked: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    invalid_citations_removed: int = 0

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens


def _format_context(candidates: list[RetrievedChunk]) -> str:
    blocks = []
    for i, cand in enumerate(candidates, start=1):
        c = cand.chunk
        blocks.append(
            f"--- Chunk {i} | {c.doc_id} p.{c.page} ---\n{c.text}"
        )
    return "\n\n".join(blocks)


def _validate_citations(
    text: str, retrieved: list[RetrievedChunk]
) -> tuple[str, list[tuple[str, int]], int]:
    """Keep only citations that point at a retrieved chunk. Returns
    (cleaned_text, valid_citations, removed_count)."""
    valid_keys = {(r.chunk.doc_id, r.chunk.page) for r in retrieved}
    citations = parse_citations(text)
    removed = 0
    cleaned = text
    for doc_id, page in citations:
        if (doc_id, page) not in valid_keys:
            cleaned = cleaned.replace(f"[{doc_id} p.{page}]", "", 1)
            removed += 1
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    valid = [c for c in citations if c in valid_keys]
    return cleaned, valid, removed


class RAGPipeline:
    """Ingest the corpus once, then answer questions against it."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
        log: StepLog | None = None,
    ) -> None:
        self.settings = settings
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = make_embedder(
                    settings.embedder, settings.embedding_model
                )
            except RuntimeError as exc:
                # The configured embedder (sentence-transformers) is not
                # installed: fall back to the hash test embedder so the
                # app still starts, and say so loudly. Retrieval quality
                # will be lower -- install sentence-transformers for the
                # real embeddings.
                print(f"WARNING: {exc} Falling back to EMBEDDER=hash.")
                from .embeddings import HashEmbeddings

                self.embedder = HashEmbeddings()
        self.store = store or make_store(settings.vector_store, settings.chroma_dir)
        self.log = log or StepLog(log_file=settings.log_file)
        self.tracer = LangfuseTracer(settings)
        self.chunks: list[Chunk] = []
        self.retriever: HybridRetriever | None = None
        self.reranker = Reranker(settings.reranker_model, enabled=settings.reranker != "off")
        self._tokens_spent = 0

    # --- Ingestion -----------------------------------------------------

    def ingest(
        self,
        corpus_dir: str | Path | None = None,
        extra_chunks: list[Chunk] | None = None,
        reset: bool = True,
    ) -> list[Chunk]:
        """Load, chunk, embed and store the corpus. ``extra_chunks`` (e.g.
        the eval's poison chunk) are added to the store but never trusted.
        Returns the chunk list."""
        corpus_dir = Path(corpus_dir or self.settings.corpus_dir)
        docs = load_corpus(corpus_dir)
        chunks = chunk_corpus(
            docs,
            chunk_size=self.settings.chunk_size_chars,
            overlap=self.settings.chunk_overlap_chars,
        )
        if extra_chunks:
            chunks = chunks + extra_chunks
        if reset:
            self.store.reset()
        if chunks:
            embeddings = self.embedder.embed([c.text for c in chunks])
            self.store.add(chunks, embeddings)
        self.chunks = chunks
        self.retriever = HybridRetriever(
            chunks,
            self.store,
            self.embedder,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        self.log.record(
            "ingest",
            f"{len(docs)} documents, {len(chunks)} chunks "
            f"({sum(1 for c in chunks if not c.trusted)} untrusted)",
            corpus_dir=str(corpus_dir),
            chunk_count=len(chunks),
        )
        return chunks

    # --- Answering -----------------------------------------------------

    def ask(self, question: str, llm: LLMClient) -> Answer:
        """Answer one question, with refusal and citation validation."""
        started = time.time()
        if self.retriever is None:
            self.ingest()
        assert self.retriever is not None

        candidates, pool = self.retriever.retrieve_with_pool(
            question,
            top_k_retrieval=self.settings.top_k_retrieval,
            top_k_final=self.settings.top_k_final,
        )
        poison_retrieved = any(not c.chunk.trusted for c in candidates)
        trusted = [c for c in candidates if c.chunk.trusted]
        top_rrf = candidates[0].rrf_score if candidates else 0.0
        top_cosine = self.retriever.dense_top_score(question)
        # Refusal is judged on the wide retrieval pool, not the final
        # top-5: a question is covered if its rare words appear anywhere
        # in the retrieved pool, even when the perfect chunk ranks 7th.
        top_overlap = self.retriever.rare_word_coverage(question, pool)

        self.log.record(
            "retrieve",
            f"top RRF {top_rrf:.4f}, top cosine {top_cosine:.4f}, "
            f"rare-word coverage {top_overlap:.2f}, "
            f"{len(candidates)} candidates, {len(trusted)} trusted",
            question=question,
            top_rrf=top_rrf,
            top_cosine=top_cosine,
            top_overlap=top_overlap,
            poison_retrieved=poison_retrieved,
        )

        # Refusal: the docs do not cover this question. Three signals:
        # rare-word coverage (the question's discriminative words do not
        # appear in any retrieved chunk), RRF and dense cosine (both
        # weak). Also refuse when no trusted chunk survived the allow-list.
        if (
            top_overlap < self.settings.refusal_min_overlap
            or (
                top_rrf < self.settings.refusal_min_rrf
                and top_cosine < self.settings.refusal_min_cosine
            )
            or not trusted
        ):
            answer = Answer(
                question=question,
                text=REFUSAL_TEXT,
                status="refused",
                retrieved=candidates,
                poison_retrieved=poison_retrieved,
                latency_seconds=time.time() - started,
            )
            self._log_answer(answer)
            return answer

        reranked = False
        if trusted:
            reranked_list = self.reranker.rerank(question, trusted)
            reranked = reranked_list is not trusted or any(
                c.rerank_score is not None for c in reranked_list
            )
            trusted = reranked_list

        context = _format_context(trusted)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "Answer using only the context above, citing every factual "
                "claim as [DOC-ID p.N].",
            },
        ]
        if self._tokens_spent >= self.settings.llm_session_token_budget:
            answer = Answer(
                question=question,
                text="Session token budget reached; I cannot make another "
                "LLM call in this session.",
                status="failed",
                retrieved=candidates,
                latency_seconds=time.time() - started,
            )
            self._log_answer(answer)
            return answer

        response: LLMResponse = llm.complete(messages)
        self._tokens_spent += response.total_tokens
        cost = response.cost_usd(
            self.settings.llm_price_input_per_1k,
            self.settings.llm_price_output_per_1k,
        )
        cleaned, valid_citations, removed = _validate_citations(
            response.text, trusted
        )
        answer = Answer(
            question=question,
            text=cleaned,
            status="answered",
            citations=valid_citations,
            retrieved=candidates,
            poison_retrieved=poison_retrieved,
            reranked=reranked,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            latency_seconds=time.time() - started,
            invalid_citations_removed=removed,
        )
        self._log_answer(answer)
        return answer

    def _log_answer(self, answer: Answer) -> None:
        trace = {
            "question": answer.question,
            "status": answer.status,
            "answer": answer.text,
            "citations": [f"{d} p.{p}" for d, p in answer.citations],
            "retrieved": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "page": r.chunk.page,
                    "trusted": r.chunk.trusted,
                    "rrf_score": round(r.rrf_score, 4),
                    "dense_score": r.dense_score,
                    "bm25_score": r.bm25_score,
                    "rerank_score": r.rerank_score,
                }
                for r in answer.retrieved
            ],
            "poison_retrieved": answer.poison_retrieved,
            "tokens": answer.tokens_used,
            "cost_usd": round(answer.cost_usd, 6),
            "latency_seconds": round(answer.latency_seconds, 3),
        }
        self.log.record("answer", answer.status, **trace)
        self.tracer.record_query(trace)


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Build a pipeline and ingest the configured corpus."""
    pipeline = RAGPipeline(settings)
    pipeline.ingest()
    return pipeline
