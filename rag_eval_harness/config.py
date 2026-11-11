"""Configuration -- everything comes from environment variables.

No API key, token or secret is ever stored in this file or anywhere in the
codebase. Copy ``.env.example`` to ``.env`` and fill in your own values,
or export the variables in your shell.

The LLM is OpenAI-compatible: point LLM_BASE_URL at OpenAI
(https://api.openai.com/v1) or at any compatible gateway (e.g. a local
server, Azure OpenAI, or another provider that speaks the same API).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root if present; harmless if absent


def _getenv(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _getfloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _getint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _getbool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """All runtime settings. Read once at startup; tests can override."""

    # --- LLM (OpenAI-compatible chat completions) ---
    llm_base_url: str = field(
        default_factory=lambda: _getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_seconds: float = field(
        default_factory=lambda: _getfloat("LLM_TIMEOUT_SECONDS", 60.0)
    )
    llm_max_tokens: int = field(default_factory=lambda: _getint("LLM_MAX_TOKENS", 1024))
    # Hard ceiling on tokens spent in one user session (input + output).
    # The pipeline refuses new LLM calls once this is reached.
    llm_session_token_budget: int = field(
        default_factory=lambda: _getint("LLM_SESSION_TOKEN_BUDGET", 100_000)
    )
    # Prices in USD per 1,000 tokens, used only for cost reporting.
    llm_price_input_per_1k: float = field(
        default_factory=lambda: _getfloat("LLM_PRICE_INPUT_PER_1K", 0.00015)
    )
    llm_price_output_per_1k: float = field(
        default_factory=lambda: _getfloat("LLM_PRICE_OUTPUT_PER_1K", 0.0006)
    )

    # --- Corpus / ingestion ---
    corpus_dir: str = field(
        default_factory=lambda: _getenv("CORPUS_DIR", "data/sample_corpus")
    )
    chunk_size_chars: int = field(
        default_factory=lambda: _getint("CHUNK_SIZE_CHARS", 800)
    )
    chunk_overlap_chars: int = field(
        default_factory=lambda: _getint("CHUNK_OVERLAP_CHARS", 150)
    )

    # --- Embeddings / retrieval ---
    embedding_model: str = field(
        default_factory=lambda: _getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # "st" = sentence-transformers (real), "hash" = deterministic hash
    # embeddings (no model download; used by tests and the smoke test).
    embedder: str = field(default_factory=lambda: _getenv("EMBEDDER", "st"))
    reranker_model: str = field(
        default_factory=lambda: _getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    # "on" = use the cross-encoder when it loads; "off" = skip reranking.
    reranker: str = field(default_factory=lambda: _getenv("RERANKER", "on"))
    top_k_retrieval: int = field(
        default_factory=lambda: _getint("TOP_K_RETRIEVAL", 20)
    )
    top_k_final: int = field(default_factory=lambda: _getint("TOP_K_FINAL", 5))
    # RRF fusion weights. BM25 is weighted 2.0 on this small,
    # terminology-dense corpus because exact terms ("LLM01", "Article 5")
    # discriminate better than small-model dense scores here; on a large
    # corpus, 1.0/1.0 is the standard choice.
    bm25_weight: float = field(default_factory=lambda: _getfloat("BM25_WEIGHT", 2.0))
    dense_weight: float = field(default_factory=lambda: _getfloat("DENSE_WEIGHT", 1.0))
    # Minimum combined retrieval score (RRF) for the best chunk. Below this
    # the system refuses instead of answering ("not in my docs").
    refusal_min_rrf: float = field(
        default_factory=lambda: _getfloat("REFUSAL_MIN_RRF", 0.020)
    )
    # Minimum dense cosine similarity for the best chunk; a second,
    # independent refusal signal.
    refusal_min_cosine: float = field(
        default_factory=lambda: _getfloat("REFUSAL_MIN_COSINE", 0.05)
    )
    # Minimum IDF-weighted overlap between the question's discriminative
    # words and the best retrieved chunk. The main refusal signal is the
    # rare-word coverage in retrieval.py: a question whose rare words do
    # not appear in any retrieved chunk is not covered by the corpus.
    refusal_min_overlap: float = field(
        default_factory=lambda: _getfloat("REFUSAL_MIN_OVERLAP", 0.30)
    )

    # --- Vector store ---
    chroma_dir: str = field(
        default_factory=lambda: _getenv("CHROMA_DIR", "data/chroma")
    )
    # "auto" = ChromaDB if installed, else the in-memory store.
    # "chroma" / "memory" force one implementation.
    vector_store: str = field(default_factory=lambda: _getenv("VECTOR_STORE", "auto"))

    # --- Logging / tracing ---
    log_file: str = field(default_factory=lambda: _getenv("LOG_FILE", "logs/queries.jsonl"))
    # Langfuse is optional and OFF by default. Set LANGFUSE_ENABLED=1 and
    # the keys below to enable it; the pipeline works identically without it.
    langfuse_enabled: bool = field(
        default_factory=lambda: _getbool("LANGFUSE_ENABLED", False)
    )
    langfuse_public_key: str = field(
        default_factory=lambda: _getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: _getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_host: str = field(
        default_factory=lambda: _getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    def require_llm_key(self) -> None:
        """Raise a clear error if no key is configured (only when a real
        LLM call is about to be made -- importing/using the mock never
        needs a key)."""
        if not self.llm_api_key:
            raise RuntimeError(
                "No LLM API key configured. Set LLM_API_KEY in your environment "
                "or in a .env file (see .env.example). The mock LLM used in "
                "tests needs no key."
            )
