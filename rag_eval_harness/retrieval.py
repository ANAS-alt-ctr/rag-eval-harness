"""Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion,
then an optional cross-encoder reranker.

Why hybrid: BM25 catches exact terms (article numbers, "LLM01", "GOVERN")
that dense embeddings blur, while dense retrieval catches paraphrases
("what fines apply" vs "administrative fines"). RRF merges the two
rankings without needing calibrated scores. The reranker then re-scores
the fused top-k with a cross-encoder, which reads the query and the chunk
together and is the strongest single ranking signal -- but it is optional:
if the model cannot load, the pipeline keeps the RRF order and logs it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import EmbeddingModel
from .store import ScoredChunk, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against
    between into through during before after above below to from up down
    in out on off over under again further once here there all any both
    each few more most other some such no nor not only own same so than
    too very can will just don should now is are was were be been being
    have has had having do does did doing would could shall may might
    must i you he she it we they them his her its our their this that
    these those am of as are what which who whom how why when where many
    much several various
    """.split()
)

# Words about the question itself rather than its topic ("discuss",
# "compare", "relate"). They never appear in the documents, so counting
# them as discriminative would make every comparison question look
# uncovered. Excluded from the rare-word refusal signal.
META_WORDS = frozenset(
    """
    discuss discusses discussed compare compares compared differ differs
    difference different relate relates related relationship contrast
    explain explains describe describes list tell tells say says said
    mention mentions according document documents section chapter article
    page pages
    """.split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(word: str) -> str:
    """Light plural stemming, for matching only (never shown to users).

    Maps plurals to their singular ("practices" -> "practice",
    "companies" -> "company", "fines" -> "fine") so a question word
    matches the form used in the documents. Deliberately minimal: verb
    forms ("fined" vs "fines") are NOT conflated, because aggressive
    stemming creates false matches on a small corpus.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stem_tokens(text: str) -> list[str]:
    return [stem(t) for t in _tokens(text)]


def _matches(qword: str, chunk_words: set[str]) -> bool:
    """A question word matches a chunk word on exact equality, or when one
    is a prefix of the other (min 5 chars) -- so "prohibition" matches
    "prohibited" and "fines" matches "fine"."""
    if qword in chunk_words:
        return True
    if len(qword) < 5:
        return False
    return any(
        (w.startswith(qword) or qword.startswith(w)) and len(w) >= 5
        for w in chunk_words
    )


@dataclass
class RetrievedChunk:
    """One candidate after fusion/reranking, with the scores that explain
    why it was chosen."""

    chunk: Chunk
    rrf_score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    dense: list[ScoredChunk],
    bm25: list[tuple[Chunk, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse a dense ranking and a BM25 ranking with RRF.

    ``score = sum(weight / (k + rank))`` over the rankings a chunk appears
    in. A chunk ranked #1 in both lists scores ~0.033 (weights 1.0); a
    chunk ranked #20 in one list scores ~0.0125. The result is ordered by
    descending score. Weights let a deployment favour the more reliable
    signal on its corpus: on this small, terminology-dense corpus BM25
    is weighted 2.0 by default (see Settings), because exact terms
    ("LLM01", "Article 5") discriminate better than small-model dense
    scores here; on a large corpus, 1.0/1.0 is the standard choice.
    """
    fused: dict[str, RetrievedChunk] = {}

    def _add(chunk: Chunk, rank: int, *, dense_score=None, bm25_score=None):
        entry = fused.get(chunk.chunk_id)
        if entry is None:
            entry = RetrievedChunk(
                chunk=chunk,
                rrf_score=0.0,
                dense_score=dense_score,
                bm25_score=bm25_score,
            )
            fused[chunk.chunk_id] = entry
        weight = dense_weight if dense_score is not None else bm25_weight
        entry.rrf_score += weight / (k + rank)
        if dense_score is not None:
            entry.dense_score = dense_score
            entry.dense_rank = rank
        if bm25_score is not None:
            entry.bm25_score = bm25_score
            entry.bm25_rank = rank

    for rank, scored in enumerate(dense, start=1):
        _add(scored.chunk, rank, dense_score=scored.score)
    for rank, (chunk, score) in enumerate(bm25, start=1):
        _add(chunk, rank, bm25_score=score)
    return sorted(fused.values(), key=lambda r: r.rrf_score, reverse=True)


class HybridRetriever:
    """BM25 + dense retrieval over a fixed chunk set."""

    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore,
        embedder: EmbeddingModel,
        dense_weight: float = 1.0,
        bm25_weight: float = 2.0,
    ) -> None:
        self.chunks = chunks
        self.store = store
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._bm25 = BM25Okapi([_stem_tokens(c.text) for c in chunks])
        self._by_id = {c.chunk_id: c for c in chunks}
        # Document frequency over stemmed tokens, for the rare-word
        # refusal signal: words that appear in almost every chunk ("ai",
        # "act") carry no information about whether a question is
        # covered, so only rare words are checked.
        n = max(1, len(chunks))
        df: dict[str, int] = {}
        for chunk in chunks:
            for token in set(_stem_tokens(chunk.text)):
                df[token] = df.get(token, 0) + 1
        self._df = df
        self._idf = {
            token: math.log((n + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }
        # Words absent from the corpus entirely are maximally
        # discriminative -- and unmatched, so they pull overlap down.
        self._oov_idf = math.log(n + 1) + 1.0

    def idf_overlap(self, query: str, chunk: Chunk) -> float:
        """IDF-weighted share of the question's content words that match
        the chunk (stemmed exact or prefix match). 0.0 = nothing in common
        beyond stopwords; 1.0 = every discriminative word matched."""
        qwords = {stem(t) for t in _tokens(query) if t not in STOPWORDS}
        if not qwords:
            return 0.0
        chunk_words = set(_stem_tokens(chunk.text))
        total = sum(self._idf.get(w, self._oov_idf) for w in qwords)
        matched = sum(
            self._idf.get(w, self._oov_idf) for w in qwords if _matches(w, chunk_words)
        )
        return matched / total if total > 0 else 0.0

    def top_idf_overlap(self, query: str, candidates: list[RetrievedChunk]) -> float:
        """Best IDF overlap over the candidate list."""
        if not candidates:
            return 0.0
        return max(self.idf_overlap(query, c.chunk) for c in candidates)

    def rare_word_coverage(self, query: str, candidates: list[RetrievedChunk]) -> float:
        """Share of the question's *rare* words covered by the union of
        the given chunks (stemmed exact match).

        Rare = a content word appearing in at most 4 chunks, or in none
        at all. Corpus-common words ("ai", "act", "system") are excluded:
        they match everywhere and say nothing about whether the question
        is actually covered. Meta-words about the question itself
        ("discuss", "compare", "relate") and pure numbers are excluded
        too -- a question can be answerable while its rarest word is
        "discuss". A chunk also contributes its document id as words, so
        "gpai" matches the EU AI Act GPAI document and "owasp" matches
        the OWASP document even when the acronym never appears in the
        body text. Returns 1.0 when the question has no rare words
        (nothing discriminative to check).

        Note the deliberate trade-off: document-id words make "which
        document is this about?" questions answerable, but they also let
        a question that merely *names* a document ("the NIST AI RMF's
        guidance on quantum computing") look covered. That case is a
        documented near-miss in the README.
        """
        qwords = {
            stem(t)
            for t in _tokens(query)
            if t not in STOPWORDS and t not in META_WORDS and not t.isdigit()
        }
        rare = {w for w in qwords if self._df.get(w, 0) <= 4}
        if not rare:
            return 1.0
        if not candidates:
            return 0.0
        union_words: set[str] = set()
        for candidate in candidates:
            union_words |= set(_stem_tokens(candidate.chunk.text))
            union_words |= {
                stem(part)
                for part in candidate.chunk.doc_id.lower().split("-")
            }
        return len(rare & union_words) / len(rare)

    def retrieve_with_pool(
        self, query: str, top_k_retrieval: int = 20, top_k_final: int = 5
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Return ``(final, pool)``: the fused top-``top_k_final``
        candidates for generation, and the wider fused top-
        ``top_k_retrieval`` pool the refusal signal is judged on. The
        refusal question ("is this topic in the corpus at all?") is
        answered over the pool; answer quality ("did we retrieve the
        right chunks?") is measured on the final list."""
        if not self.chunks:
            return [], []
        qvec = self.embedder.embed([query])[0]
        dense = self.store.query(qvec, top_k_retrieval)
        bm25_scores = self._bm25.get_scores(_stem_tokens(query))
        bm25_order = sorted(
            range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k_retrieval]
        bm25_ranked = [
            (self.chunks[i], float(bm25_scores[i]))
            for i in bm25_order
            if bm25_scores[i] > 0
        ]
        fused = reciprocal_rank_fusion(
            dense,
            bm25_ranked,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
        return fused[:top_k_final], fused[:top_k_retrieval]

    def retrieve(
        self, query: str, top_k_retrieval: int = 20, top_k_final: int = 5
    ) -> list[RetrievedChunk]:
        """Return the fused top-``top_k_final`` candidates."""
        final, _ = self.retrieve_with_pool(query, top_k_retrieval, top_k_final)
        return final

    def dense_top_score(self, query: str) -> float:
        """Best dense cosine similarity for the query (refusal signal)."""
        if not self.chunks:
            return 0.0
        qvec = self.embedder.embed([query])[0]
        results = self.store.query(qvec, 1)
        return results[0].score if results else 0.0


class Reranker:
    """Cross-encoder reranker (lazy). Falls back to a no-op when the model
    cannot load, so retrieval still works offline."""

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._load_error: str | None = None
        self.available = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.available = True
            return True
        except Exception as exc:  # missing lib, no network, bad model name
            self._load_error = str(exc)
            return False

    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Re-score candidates with the cross-encoder, highest first.
        Returns the input unchanged (with ``reranked=False`` behaviour)
        when disabled or unavailable."""
        if not self.enabled or not candidates:
            return candidates
        if not self._load():
            return candidates
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(
            candidates,
            key=lambda c: c.rerank_score if c.rerank_score is not None else -1e9,
            reverse=True,
        )
