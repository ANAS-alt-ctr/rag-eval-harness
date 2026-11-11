"""Streamlit UI for the RAG eval harness.

Run: streamlit run app/streamlit_app.py   (http://localhost:8501)

Ask a question about the EU AI Act, the NIST AI RMF or the OWASP Top 10
for LLM Applications. The UI shows the answer with its citations, the
retrieved chunks with their scores, and whether the system refused.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval_harness.config import Settings
from rag_eval_harness.llm import MockLLM, OpenAICompatibleClient
from rag_eval_harness.pipeline import RAGPipeline

st.set_page_config(page_title="Gov RAG — cited answers", layout="wide")
st.title("RAG over AI-governance documents")
st.caption(
    "EU AI Act · NIST AI Risk Management Framework · OWASP Top 10 for LLM "
    "Applications — every factual claim cites its source, and questions the "
    "documents do not cover are refused, not guessed."
)


@st.cache_resource
def _pipeline() -> RAGPipeline:
    settings = Settings()
    pipeline = RAGPipeline(settings)
    pipeline.ingest()
    return pipeline


@st.cache_resource
def _llm():
    settings = Settings()
    if settings.llm_api_key:
        return OpenAICompatibleClient(settings)
    return MockLLM()


pipeline = _pipeline()
llm = _llm()
settings = Settings()

with st.sidebar:
    st.header("Corpus")
    docs = sorted({c.doc_id for c in pipeline.chunks})
    for doc_id in docs:
        n = sum(1 for c in pipeline.chunks if c.doc_id == doc_id)
        st.write(f"- **{doc_id}** — {n} chunks")
    st.write(f"Total chunks: **{len(pipeline.chunks)}**")
    st.header("Configuration")
    st.write(f"LLM: **{'real (' + settings.llm_model + ')' if settings.llm_api_key else 'mock (no key set)'}**")
    st.write(f"Embedder: **{settings.embedder}**")
    st.write(f"Reranker: **{settings.reranker}**")
    if st.button("Rebuild index"):
        pipeline.ingest(reset=True)
        st.success(f"Rebuilt: {len(pipeline.chunks)} chunks.")

examples = [
    "What is the maximum fine for breaching the EU AI Act's prohibited practices?",
    "What are the four functions of the NIST AI Risk Management Framework?",
    "What is LLM01 in the OWASP Top 10 for LLM Applications 2025?",
    "What is the capital of France?",
]
question = st.text_input("Ask a question", value="")
col1, col2, col3, col4 = st.columns(4)
for col, example in zip((col1, col2, col3, col4), examples):
    if col.button(example, key=example):
        question = example

if question:
    with st.spinner("Retrieving and answering..."):
        answer = pipeline.ask(question, llm)
    if answer.status == "refused":
        st.warning(answer.text)
    elif answer.status == "failed":
        st.error(answer.text)
    else:
        st.success(answer.text)
    st.write(
        f"Status: **{answer.status}** · Tokens: **{answer.tokens_used}** · "
        f"Cost: **${answer.cost_usd:.6f}** · Latency: "
        f"**{answer.latency_seconds:.2f}s**"
    )
    if answer.citations:
        st.write("Citations: " + ", ".join(
            f"[{doc} p.{page}]" for doc, page in answer.citations
        ))
    if answer.poison_retrieved:
        st.warning(
            "An untrusted (poisoned) chunk was retrieved for this question. "
            "It was excluded from the answer and never cited."
        )
    with st.expander("Retrieved chunks (with scores)"):
        for r in answer.retrieved:
            st.markdown(
                f"**{r.chunk.chunk_id}** ({r.chunk.source}, p.{r.chunk.page}) "
                f"— RRF {r.rrf_score:.4f}"
                + (f", rerank {r.rerank_score:.3f}" if r.rerank_score is not None else "")
                + (" — UNTRUSTED" if not r.chunk.trusted else "")
            )
            st.write(r.chunk.text)
    with st.expander("Latest trace (logs/queries.jsonl)"):
        log_path = Path(settings.log_file)
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            st.code("\n".join(lines[-3:]), language="json")
        else:
            st.write("No log file yet.")
