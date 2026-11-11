"""Embeddings.

Two implementations share one interface:

* :class:`SentenceTransformerEmbeddings` -- the real model
  (``sentence-transformers/all-MiniLM-L6-v2``, 384 dimensions, CPU-friendly).
  Imported lazily so the package works without torch installed.
* :class:`HashEmbeddings` -- deterministic, dependency-free fallback used
  by the tests and the smoke test. It hashes word n-grams into a fixed
  384-dimensional vector, so identical or overlapping texts score high
  and unrelated texts score near zero. It is a *test double*, not a real
  embedding model: never report hash-embedding eval numbers as model
  results (see README).

Both return L2-normalised vectors, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity of two (already normalised) vectors."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class EmbeddingModel(Protocol):
    """Minimal interface every embedder implements."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddings:
    """Deterministic hash embeddings (tests / smoke test only).

    Each word unigram and bigram is hashed into one of ``DIM`` buckets
    with a +1/-1 sign (signed hashing, a.k.a. the hashing trick). The
    result is L2-normalised. Texts sharing words score high; unrelated
    texts score near zero. Fully deterministic across runs and machines.
    """

    dim = DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(DIM, dtype=np.float64)
            toks = _tokens(text)
            grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % DIM
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class SentenceTransformerEmbeddings:
    """Real embeddings via sentence-transformers (lazy import).

    The model is downloaded from Hugging Face on first use (~90 MB for
    all-MiniLM-L6-v2) and cached in ``~/.cache/huggingface``. Raises a
    clear error if the library is not installed.
    """

    dim = DIM

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in CI only
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers`, or set EMBEDDER=hash "
                "to use the deterministic test embedder."
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, row)) for row in np.asarray(vectors)]


def make_embedder(kind: str, model_name: str) -> EmbeddingModel:
    """Build an embedder by name: ``"st"`` (real) or ``"hash"`` (test)."""
    if kind == "hash":
        return HashEmbeddings()
    if kind == "st":
        return SentenceTransformerEmbeddings(model_name)
    raise ValueError(f"Unknown embedder {kind!r}; use 'st' or 'hash'.")
