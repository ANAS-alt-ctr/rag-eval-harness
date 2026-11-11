"""LLM clients.

Two implementations share one small interface:

* :class:`OpenAICompatibleClient` -- talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI itself, Azure OpenAI, a local
  server, or another gateway). Configured ONLY via environment variables
  (see :mod:`rag_eval_harness.config`).
* :class:`MockLLM` -- deterministic, scripted responses for tests and the
  smoke test. Needs no key and makes no network calls. For RAG answer
  prompts it builds its answer from the context chunks embedded in the
  prompt (first sentence of the top chunks, with their citations), so the
  mock exercises the real citation and faithfulness paths instead of
  returning a canned string.

Token usage is taken from the API response when present, otherwise
estimated as ``len(text) // 4`` characters per token. Cost is computed
from the configured per-1k-token prices; a mock run reports $0.00.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Settings


@dataclass
class LLMResponse:
    """One chat completion."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, price_input_per_1k: float, price_output_per_1k: float) -> float:
        return (
            self.input_tokens / 1000.0 * price_input_per_1k
            + self.output_tokens / 1000.0 * price_output_per_1k
        )


class LLMClient(Protocol):
    """Minimal interface every LLM client in this project implements."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        ...


def _estimate_tokens(text: str) -> int:
    """Rough fallback when the API returns no usage block."""
    return max(1, len(text) // 4)


class OpenAICompatibleClient:
    """OpenAI-compatible chat completions client (httpx, sync)."""

    def __init__(self, settings: Settings) -> None:
        settings.require_llm_key()
        self._base_url = settings.llm_base_url.rstrip("/")
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds
        self._max_tokens = settings.llm_max_tokens
        self._client = httpx.Client(timeout=self._timeout)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        response = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens") or _estimate_tokens(
            json.dumps(messages)
        )
        output_tokens = usage.get("completion_tokens") or _estimate_tokens(text)
        return LLMResponse(
            text=text,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            model=data.get("model", self._model),
        )


# Context blocks in the answer prompt look like:
#   --- Chunk 1 | EU-AI-ACT p.2 ---
#   <chunk text>
_CHUNK_BLOCK_RE = re.compile(
    r"--- Chunk \d+ \| ([A-Z0-9][A-Z0-9\-]*) p\.(\d+) ---\n(.*?)(?=\n--- Chunk |\nQuestion:|\Z)",
    re.DOTALL,
)


def _first_sentence(text: str) -> str:
    """First sentence of a chunk, cleaned of newlines."""
    text = " ".join(text.split())
    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    return (match.group(0) if match else text[:200]).strip()


def _best_sentence(text: str, question: str) -> str:
    """The sentence in ``text`` sharing the most content words with the
    question (ties go to the earliest sentence). Falls back to the first
    sentence when nothing overlaps."""
    stopwords = {
        "what", "which", "who", "whom", "how", "why", "when", "where", "is",
        "are", "was", "were", "be", "been", "does", "do", "did", "the", "a",
        "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
        "under", "about", "from", "it", "its", "that", "this", "these",
        "those", "as", "at", "are",
    }
    qwords = {
        w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stopwords
    }
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]
    # Skip fragments (bare headings like "## 10.") -- a sentence needs at
    # least 3 alphabetic words to be an answer sentence.
    sentences = [
        s for s in sentences if len(re.findall(r"[a-zA-Z]+", s)) >= 3
    ]
    if not sentences:
        return ""
    best, best_score = sentences[0], -1
    for sentence in sentences:
        words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(qwords & words)
        if score > best_score:
            best, best_score = sentence, score
    return best


@dataclass
class MockLLM:
    """Scripted LLM for tests / smoke test. No network, no key.

    Behaviour is driven by ``script``: a list of ``(keyword, responses)``
    pairs. When the last user message contains ``keyword`` (case
    insensitive), the next response from that keyword's list is returned.
    ``responses`` may be plain strings or dicts (serialised to JSON).
    Anything unmatched gets an answer built from the prompt's context
    chunks (see :func:`_answer_from_context`), or ``default_response``
    when the prompt has no context blocks.

    This lets tests script realistic failure modes -- e.g. an answer that
    cites a poisoned chunk -- to exercise the citation validation
    deterministically.
    """

    script: list[tuple[str, list[Any]]] = field(default_factory=list)
    default_response: str = (
        "I don't have information about that in the provided documents."
    )
    calls: list[list[dict[str, str]]] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict, repr=False)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.calls.append(messages)
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
        for keyword, responses in self.script:
            if keyword.lower() in user_text.lower():
                idx = self._counters.get(keyword, 0)
                chosen = responses[min(idx, len(responses) - 1)]
                self._counters[keyword] = idx + 1
                text = chosen if isinstance(chosen, str) else json.dumps(chosen)
                return LLMResponse(
                    text=text,
                    input_tokens=_estimate_tokens(json.dumps(messages)),
                    output_tokens=_estimate_tokens(text),
                    model="mock",
                )
        text = _answer_from_context(user_text) or self.default_response
        return LLMResponse(
            text=text,
            input_tokens=_estimate_tokens(json.dumps(messages)),
            output_tokens=_estimate_tokens(text),
            model="mock",
        )


def _answer_from_context(prompt: str) -> str | None:
    """Build a deterministic cited answer from the prompt's context blocks.

    Returns ``None`` when the prompt carries no context blocks (the caller
    then falls back to the refusal-style default response).
    """
    blocks = _CHUNK_BLOCK_RE.findall(prompt)
    if not blocks:
        return None
    question = ""
    qmatch = re.search(r"Question:\s*(.*?)\s*\n\nAnswer using only", prompt, re.DOTALL)
    if qmatch:
        question = qmatch.group(1).strip()
    sentences: list[str] = []
    # A real LLM reads all the context chunks; the mock reads the top 3
    # so the answer can draw on more than the single best chunk.
    for doc_id, page, chunk_text in blocks[:3]:
        sentence = _best_sentence(chunk_text, question) if question else _first_sentence(chunk_text)
        if sentence:
            sentences.append(f"{sentence} [{doc_id} p.{page}]")
    if not sentences:
        return None
    return " ".join(sentences)
