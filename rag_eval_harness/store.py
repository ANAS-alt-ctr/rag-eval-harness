"""Vector stores.

Two implementations share one interface:

* :class:`ChromaStore` -- persistent ChromaDB collection (the real store).
  Imported lazily so the package works without ChromaDB installed.
* :class:`InMemoryStore` -- exact cosine search over numpy arrays. Used by
  the tests and whenever ChromaDB is unavailable. Same ranking semantics,
  no persistence.

Both store the chunk id, text and metadata (doc id, source, page,
trusted flag) alongside the embedding, and return ``(chunk, score)``
pairs ordered by descending score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity, 0..1


class VectorStore(Protocol):
    """Minimal interface every vector store implements."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[ScoredChunk]:
        ...

    def count(self) -> int:
        ...

    def reset(self) -> None:
        ...


class InMemoryStore:
    """Exact cosine search in memory. No persistence, no dependencies."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._chunks.extend(chunks)
        mat = np.asarray(embeddings, dtype=np.float64)
        self._matrix = mat if self._matrix is None else np.vstack([self._matrix, mat])

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[ScoredChunk]:
        if not self._chunks or self._matrix is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0.0] = 1.0
        sims = (self._matrix / norms[:, None]) @ q
        order = np.argsort(-sims)[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(sims[i])) for i in order
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks = []
        self._matrix = None


class ChromaStore:
    """Persistent ChromaDB collection (lazy import)."""

    def __init__(self, persist_dir: str, collection_name: str = "gov-rag") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - exercised in CI only
            raise RuntimeError(
                "chromadb is not installed. Install it with "
                "`pip install chromadb`, or set VECTOR_STORE=memory to use "
                "the in-memory store."
            ) from exc
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page": c.page,
                    "trusted": c.trusted,
                }
                for c in chunks
            ],
        )

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out: list[ScoredChunk] = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        ids = result["ids"][0]
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance is 1 - cosine similarity.
            out.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=doc_id,
                        doc_id=meta["doc_id"],
                        source=meta["source"],
                        page=int(meta["page"]),
                        text=doc,
                        trusted=bool(meta["trusted"]),
                    ),
                    score=float(1.0 - dist),
                )
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": "cosine"}
        )


def make_store(kind: str, chroma_dir: str) -> VectorStore:
    """Build a store by name: ``"chroma"``, ``"memory"``, or ``"auto"``
    (ChromaDB if importable, else in-memory with a printed warning)."""
    if kind == "memory":
        return InMemoryStore()
    if kind == "chroma":
        return ChromaStore(chroma_dir)
    if kind == "auto":
        try:
            import chromadb  # noqa: F401
        except ImportError:
            print(
                "WARNING: chromadb not installed; using the in-memory store. "
                "Install it with `pip install chromadb` for persistence."
            )
            return InMemoryStore()
        return ChromaStore(chroma_dir)
    raise ValueError(f"Unknown vector store {kind!r}; use 'auto', 'chroma' or 'memory'.")
