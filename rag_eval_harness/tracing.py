"""Optional Langfuse tracing. OFF by default.

When ``LANGFUSE_ENABLED=1`` and the keys are set in the environment, every
query is also sent to Langfuse as a trace (question, retrieved chunks,
answer, scores, latency, cost). When disabled -- the default -- this module
is a no-op and the pipeline works identically, logging only to the local
JSONL file. Langfuse is an *optional* dependency: it is only imported if
enabled.
"""

from __future__ import annotations

from .config import Settings


class LangfuseTracer:
    """Thin wrapper: no-op unless enabled and configured."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = False
        self._client = None
        if not settings.langfuse_enabled:
            return
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            print(
                "WARNING: LANGFUSE_ENABLED=1 but LANGFUSE_PUBLIC_KEY / "
                "LANGFUSE_SECRET_KEY are not set; Langfuse tracing stays off."
            )
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self.enabled = True
        except ImportError:
            print(
                "WARNING: langfuse package not installed; tracing stays off. "
                "Install it with `pip install langfuse`."
            )

    def record_query(self, trace: dict) -> None:
        """Send one query trace to Langfuse (no-op when disabled)."""
        if not self.enabled or self._client is None:
            return
        try:
            self._client.trace(
                name="rag-query",
                input=trace.get("question"),
                output=trace.get("answer"),
                metadata={
                    "status": trace.get("status"),
                    "citations": trace.get("citations"),
                    "latency_seconds": trace.get("latency_seconds"),
                    "tokens": trace.get("tokens"),
                    "cost_usd": trace.get("cost_usd"),
                    "retrieved": [
                        {
                            "chunk_id": r.get("chunk_id"),
                            "rrf_score": r.get("rrf_score"),
                            "rerank_score": r.get("rerank_score"),
                        }
                        for r in trace.get("retrieved", [])
                    ],
                },
            )
            self._client.flush()
        except Exception:
            # Tracing must never break answering.
            pass
