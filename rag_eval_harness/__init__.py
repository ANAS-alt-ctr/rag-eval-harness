"""RAG System with Failure Analysis + LLM Eval Harness.

Ask questions about AI-governance documents (EU AI Act, NIST AI RMF,
OWASP Top 10 for LLM) and get cited answers -- or a clear refusal when
the documents do not cover the question. Every answer is evaluated
against a golden set with locally computed metrics (no paid eval API).
"""

from .pipeline import Answer, RAGPipeline, build_pipeline

__all__ = ["RAGPipeline", "Answer", "build_pipeline"]
__version__ = "0.1.0"
