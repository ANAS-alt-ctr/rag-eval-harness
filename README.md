# RAG System with Failure Analysis + LLM Eval Harness

Ask a question about AI-governance documents — the **EU AI Act**, the **NIST
AI Risk Management Framework**, and the **OWASP Top 10 for LLM Applications**
— and get back a **cited answer**, or a clear **refusal** when the documents do
not cover the question. Every answer is evaluated against a 50-question
golden set with locally computed metrics, and a CI gate fails the build when
any metric regresses.

Built with **Python 3.10+**, **sentence-transformers** (CPU), **ChromaDB**,
**BM25 + dense hybrid retrieval**, **FastAPI** and **Streamlit**. CPU-only —
it runs on a laptop with no GPU. The LLM is OpenAI-compatible and configured
**only via environment variables**; no key is ever stored in code.

> This is a portfolio project. It is real, working code — not a skeleton —
> but its eval numbers below are honest about what was measured with a mock
> LLM and hash embeddings, and what is still a placeholder for a real model.

## Quick start

```bash
git clone https://github.com/ANAS-alt-ctr/rag-eval-harness.git   # or copy this folder
cd rag-eval-harness
python -m venv .venv
.venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your LLM_API_KEY in .env
python scripts/smoke_test.py  # 3/3, no API key needed
python -m pytest -q           # 32 tests, no API key needed
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison
```

Run the UI and the API:

```bash
streamlit run app/streamlit_app.py            # http://localhost:8501
uvicorn app.main:app --reload --port 8000     # http://localhost:8000/docs
```

Full step-by-step for Windows is in **[BUILD-GUIDE.md](BUILD-GUIDE.md)**.

## How it works

```
Question
   |
   v
Hybrid retrieval   -- BM25 (stemmed tokens) + dense embeddings
   |                  (all-MiniLM-L6-v2), fused with reciprocal rank
   |                  fusion (BM25 weight 2.0, dense 1.0 on this small,
   |                  terminology-dense corpus). Top 20 retrieved.
   v
Source allow-list  -- only chunks from the trusted corpus may be used to
   |                  generate an answer. An untrusted chunk that retrieval
   |                  surfaced (the eval's planted poison chunk) is logged
   |                  as poison_retrieved and excluded from generation:
   |                  it can never be cited.
   v
Refusal            -- if the question's rare words are not covered by the
   |                  retrieved pool, or both the fused RRF score and the
   |                  dense cosine are weak, or no trusted chunk survived,
   |                  the system refuses ("not in my docs") instead of
   |                  hallucinating. Refusal costs zero LLM calls.
   v
Reranker (optional)-- cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores the
   |                  top candidates. If the model cannot load, the pipeline
   |                  keeps the RRF order and logs it.
   v
Cited generation   -- the LLM gets the trusted chunks with citation markers
   |                  and is instructed to answer ONLY from them, citing
   |                  every factual claim as [DOC-ID p.N].
   v
Citation check     -- citations in the answer are validated against the
                      retrieved chunks. A citation to a chunk that was not
                      retrieved (or to an untrusted chunk) is stripped from
                      the answer and counted.
   |
   v
Trace              -- every query is logged as one JSON object
                      (question, retrieved chunks + scores, answer,
                      citations, tokens, latency, cost) to logs/queries.jsonl
                      and, when enabled, to Langfuse.
```

### Guardrails (the point of the project)

| Guardrail | Where | What it does |
| --- | --- | --- |
| Source allow-list | `pipeline.py` | Only trusted-corpus chunks reach the LLM; untrusted chunks are logged and excluded, never cited |
| Refusal | `pipeline.py`, `retrieval.py: rare_word_coverage` | Rare-word coverage over the retrieval pool + RRF/cosine signals; refuses instead of guessing |
| Citation validation | `pipeline.py: _validate_citations` | Citations must point at retrieved chunks; invalid ones are stripped and counted |
| Cost ceiling | `config.py`, `pipeline.py` | Session token budget (default 100,000); the run stops instead of spending more |
| Step log | `logging.py` | Every query as JSONL (`logs/queries.jsonl`): question, chunks, scores, answer, citations, cost |
| Secrets | `config.py`, `.env` | LLM base URL / key / model from env only; `.env` is git-ignored |
| CI gate | `.github/workflows/ci.yml`, `eval/baseline.json` | Tests + mock eval + baseline comparison on every push; fails on regression |

## Corpus

**Full corpus (download it):** the official PDFs —

| Document | Source |
| --- | --- |
| EU AI Act, Regulation (EU) 2024/1689 | EUR-Lex: `https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689` |
| NIST AI Risk Management Framework 1.0 (NIST-AI-100-1, Jan 2023) | DOI `https://doi.org/10.6028/NIST.AI.100-1` (NIST restructured nvlpubs in 2026; the DOI is the canonical locator, with an archive.org fallback in the script) |
| OWASP Top 10 for LLM Applications 2025 | GitHub release: `https://github.com/OWASP/www-project-top-10-for-large-language-model-applications/releases/download/2024/OWASP-Top-10-for-LLMs-v2025.pdf` |

```bash
python scripts/download_corpus.py            # downloads + verifies + writes data/corpus/manifest.json
python scripts/build_index.py --corpus data/corpus
```

**Sample corpus (built in, zero setup):** `data/sample_corpus/` — four text
files written as faithful summaries of the same documents (35 chunks):
`eu-ai-act.txt`, `eu-ai-act-gpai.txt`, `nist-ai-rmf.txt`,
`owasp-top10-llm.txt`. The eval golden set is written against this corpus,
so the eval runs out of the box. Files or directories starting with `_` are
never ingested — that is where test fixtures live.

## Evaluation

The golden set is `eval/golden_set.json`: **50 hand-written questions** —

| Category | n | What it tests |
| --- | --- | --- |
| factual | 15 | Single-document facts (fines, dates, article numbers, list items) |
| multi_doc | 12 | Synthesis across two or three documents |
| unanswerable | 12 | Must refuse (expected behaviour: `refuse`) |
| adversarial | 11 | Must answer from the real docs while a planted contradictory chunk is in the store; the poison chunk must never be cited |

Each question has a short gold answer and the gold-relevant chunk ids in the
sample corpus. Run it with:

```bash
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison   # no key
python scripts/run_eval.py --llm openai --embedder st --store chroma --poison   # real model
python scripts/run_eval.py --llm openai --check-baseline                        # CI gate
```

Metrics are reported **per category, never as one blended number**:

| Metric | What it measures |
| --- | --- |
| faithfulness | Fraction of the answer's claims (sentences) supported by the retrieved context — **lexical proxy** (see below), not a true entailment judge |
| answer_relevancy | Cosine similarity between the question and the answer, using the same embedder as retrieval |
| context_precision | Retrieved chunks that are gold-relevant / retrieved chunks |
| context_recall | Gold-relevant chunks retrieved / gold-relevant chunks |
| citation_precision | Citations pointing at a retrieved chunk / all citations |
| refusal_rate | Unanswerable questions refused / all unanswerable questions |
| poison_citation_rate | Adversarial answers citing the poison chunk / all adversarial questions (target: 0) |
| poison_retrieved_rate | Adversarial questions where the poison chunk was retrieved at all (diagnostic: retrieval *will* surface it; the defence is at the answer layer) |

### Measured baseline — mock LLM + hash embeddings (a ceiling, not a real result)

The mock builds its answers from the retrieved context sentences, so 1.00
faithfulness only proves the plumbing (retrieval, refusal, citations) works.
It says nothing about a real model's accuracy. Context precision is capped
near 0.20 by construction: each question has one gold chunk and five chunks
are retrieved, so 1/5 is the maximum for single-gold questions — read
**context recall** for retrieval quality, not precision.

| Category | n | Faithfulness | Relevancy | Ctx precision | Ctx recall | Citation precision | Refusal | Poison cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factual | 15 | 1.00 | 0.46 | 0.19 | 0.87 | 1.00 | 0.00 | — |
| multi_doc | 12 | 1.00 | 0.41 | 0.27 | 0.58 | 1.00 | 0.00 | — |
| unanswerable | 12 | 1.00 | 0.14 | 0.00 | 1.00 | 1.00 | 0.50 | — |
| adversarial | 11 | 1.00 | 0.36 | 0.16 | 0.73 | 1.00 | 0.00 | 0.00 (seen 0.82) |

Run: `python scripts/run_eval.py --llm mock --embedder hash --store memory --poison`.
Stored in `eval/baseline.json`; the CI gate compares against it.

### Same run with real embeddings + ChromaDB + reranker (still mock LLM)

Run: `python scripts/run_eval.py --llm mock --embedder st --store chroma
--reranker on --poison`. Stored in `eval/results-real-embeddings.json`.
The LLM is still the mock (answers built from context sentences), so
faithfulness is still a plumbing check — but retrieval is now real, and
the improvement over the hash baseline is the point: answer relevancy
and context recall both rise, because real dense embeddings retrieve
paraphrased questions that hash embeddings miss.

| Category | n | Faithfulness | Relevancy | Ctx precision | Ctx recall | Citation precision | Refusal | Poison cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factual | 15 | 1.00 | 0.72 | 0.20 | 0.93 | 1.00 | 0.00 | — |
| multi_doc | 12 | 1.00 | 0.67 | 0.32 | 0.69 | 1.00 | 0.00 | — |
| unanswerable | 12 | 1.00 | 0.23 | 0.00 | 1.00 | 1.00 | 0.50 | — |
| adversarial | 11 | 1.00 | 0.66 | 0.18 | 0.82 | 1.00 | 0.00 | 0.00 (seen 0.73) |

The refusal rate is unchanged (0.50) because the refusal signal is
lexical and embedding-independent — the same six near-miss questions
(`u04`, `u05`, `u06`, `u09`, `u10`, `u12`) are still answered. Real
embeddings fix retrieval; they do not fix refusal. That is why the
failure analysis names a reranker-score or LLM-judge check as the next
step, not "better embeddings".

### PLACEHOLDER — real LLM + real embeddings (replace after a real run)

Run `python scripts/run_eval.py --llm openai --embedder st --store chroma --poison`
and paste the per-category table here. Do **not** copy the mock numbers into a CV.

| Category | n | Faithfulness | Relevancy | Ctx precision | Ctx recall | Citation precision | Refusal | Poison cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factual | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | — |
| multi_doc | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | — |
| unanswerable | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | — |
| adversarial | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

**Extending the golden set to 100 questions:** add entries to
`eval/golden_set.json` (same shape: `id`, `category`, `question`,
`gold_answer`, `gold_chunk_ids`, `expected`), keeping the category split
balanced and keeping unanswerable and adversarial questions in the set —
a system that never refuses and a system that cites poison are both worse
than one that scores lower on factual questions. See BUILD-GUIDE.md §9.

### The baseline gate (green-to-red proof)

`eval/baseline.json` records the mock baseline above. `scripts/run_eval.py
--check-baseline` fails (exit 1) when any per-category metric falls below its
baseline by more than the tolerance (default 0.05), when the unanswerable
refusal rate falls, or when the poison-citation rate rises. To see it work:

```bash
# 1. Break something deliberately, e.g. in rag_eval_harness/pipeline.py
#    change REFUSAL_MIN_OVERLAP handling so refusal never triggers, or
#    remove the trusted-only filter from the generation context.
# 2. Run the eval with the gate:
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison --check-baseline
# 3. Watch it fail with the regressed metric named, then revert the change.
```

### Second opinion: RAGAS / DeepEval

The local metrics are lexical proxies by design (no paid eval API). For a
second opinion, run the same golden set through an LLM-judge framework:

```bash
pip install ragas datasets
# ragas: build a Dataset from eval/golden_set.json (question, contexts =
# the retrieved chunk texts, answer, ground_truth = gold_answer) and run
# evaluate(dataset, metrics=[faithfulness, answer_relevancy,
# context_precision, context_recall]).
# DeepEval: same mapping into deepeval's RAG metrics.
```

The RAGAS faithfulness number (LLM judge, claim-level entailment) is the
one to quote alongside the local proxy — expect it to be lower than 1.00,
and that gap is itself a finding (see "Where this breaks").

## Failure analysis

The eval exists to find failures. The ones already found, with the evidence:

1. **Near-miss refusals (6 of 12 unanswerable questions are answered).**
   The refusal signal is lexical: it checks whether the question's rare
   words appear in the retrieved pool. Questions that share the corpus's
   vocabulary while asking about something outside it therefore pass:
   `u04` GDPR data-subject access requests ("request" and "rule" match),
   `u05` fine-tuning a model ("fine" in "fine-tune" is a false friend, and
   "language" matches), `u06` price of compliance software ("compliance"
   genuinely appears in the AI Act text), `u09` appealing a fine ("fine"
   matches), `u10` best embedding model for RAG ("embedding" and "RAG"
   both appear in the OWASP text), `u12` NIST guidance on quantum computing
   (the question *names* the NIST AI RMF, and document-id words match).
   What would catch them: a cross-encoder reranker score on the top chunk,
   or an LLM-as-judge that checks whether the retrieved chunks actually
   answer the question. That is the documented next step, not a hidden gap.
2. **The poison chunk IS retrieved (9 of 11 adversarial questions).**
   It is lexically close to the real fines/threshold chunks, so hybrid
   retrieval surfaces it — that is expected. The defence is the source
   allow-list at the answer layer: it was cited **0 of 11 times**. The
   eval measures both numbers separately so a future change that "fixes"
   retrieval by hiding the poison would show up as a *drop* in
   `poison_retrieved_rate` without any change in `poison_citation_rate`.
   What this does NOT prove: a semantic contradiction detector. The
   system rejects the poison because its source is not in the allow-list,
   not because it detected the contradiction. A poison chunk smuggled
   into a trusted document would not be caught — see "Where this breaks".
3. **Intro chunks outrank specific chunks.** The OWASP intro chunk (#0)
   mentions "LLM01 to LLM10" and shares the most question terms, so it
   ranks above the LLM01 chunk itself; the EU AI Act overview (#0) does
   the same for fines questions. The mock answers from the top 3 chunks,
   so the specific chunk still contributes, but a real answer benefits
   from the reranker here. The unit tests assert top-3, not top-1, for
   this reason.
4. **Multi-doc recall (0.58) lags factual recall (0.87).** Synthesis
   questions need two chunks from different documents; RRF rewards chunks
   that rank well in *both* signals, so a chunk that is BM25 #1 but dense
   #15 can lose to a mediocre dual-signal chunk. The reranker is the
   intended fix; it is off in the mock baseline.
5. **Faithfulness 1.00 is a saturated proxy, not a real result.** The
   lexical check (content-word overlap per claim) passes whenever the
   answer reuses chunk wording — which the mock does by construction, and
   which a real model often does too. RAGAS faithfulness (LLM judge) is
   the number that would actually discriminate.

## Cost and latency notes

- Default model is `gpt-4o-mini`. Prices in `.env.example` are the
  gpt-4o-mini list prices at the time of writing ($0.15 / $0.60 per 1M
  input/output tokens); update `LLM_PRICE_INPUT_PER_1K` /
  `LLM_PRICE_OUTPUT_PER_1K` if you change model or provider — the reported
  cost is only as accurate as those two numbers.
- One question costs roughly **1 LLM call** (answer generation only;
  retrieval is local), ~1,500 tokens with five context chunks, i.e.
  **~$0.0003 per question** at gpt-4o-mini prices. The session budget
  (`LLM_SESSION_TOKEN_BUDGET`, default 100,000 tokens) caps a whole session
  at roughly **$0.03–$0.05** of model cost.
- Latency with the mock is ~0.003s per question (no network). With a real
  model, expect ~2–4s per question (one call + local retrieval). Local
  embedding of the 35-chunk sample corpus takes under a second; the full
  PDFs (hundreds of pages) take a few minutes on CPU at first index build.
- The embedding model (~90 MB) and reranker (~80 MB) download once from
  Hugging Face and are cached in `~/.cache/huggingface`.

## Where this breaks and why

Honest limits, not marketing:

1. **It is only as good as the model.** A mini-class model will still
   answer from the wrong chunk when retrieval misleads it; the citation
   check catches invented citations, not wrong-but-cited answers. The
   real-model eval numbers are the ones that matter — and they are not
   measured yet.
2. **The refusal signal is lexical, not semantic.** See failure analysis
   #1: same-vocabulary-different-question cases pass it. A reranker-score
   threshold or an LLM judge would be needed to catch them.
3. **The poison defence is an allow-list, not a detector.** A false
   statement inside a trusted document, or a poisoned chunk added to the
   trusted corpus directory, would be cited like any other chunk.
4. **The sample corpus is a summary, not the real documents.** Chunk ids,
   page numbers and the golden set all refer to the sample text. After
   downloading the real PDFs, the golden set must be re-mapped to the real
   chunk ids (see BUILD-GUIDE.md §9) — the sample-corpus numbers do not
   transfer. The refusal thresholds are likewise calibrated on the sample
   corpus: on the full 1,446-chunk corpus the rare-word coverage signal is
   noisier (a rare word is more likely to appear *somewhere* in a 20-chunk
   pool), so the thresholds need re-calibration as part of the re-mapping.
5. **Hash embeddings are a test double.** The mock baseline's retrieval
   numbers (context recall 0.87/0.58) reflect BM25 plus hash noise, not
   real dense retrieval. Real all-MiniLM-L6-v2 embeddings should improve
   recall on paraphrased questions; that is an unmeasured claim until the
   real-model eval runs.
6. **Faithfulness is a lexical proxy.** See failure analysis #5.
7. **No authentication, no multi-user isolation.** The FastAPI backend is a
   demo service. Do not expose it publicly without adding auth and a
   per-user budget.
8. **ChromaDB is single-writer and local.** Fine for one user; the
   in-memory store is the fallback when ChromaDB is unavailable, and it
   loses the index on restart (rebuild with `scripts/build_index.py`).

## Project structure

```
rag-eval-harness/
├── rag_eval_harness/       # the package
│   ├── config.py           # env-only settings (no secrets in code)
│   ├── llm.py              # OpenAI-compatible client + RAG-aware MockLLM
│   ├── chunking.py         # PDF/text ingestion, sentence-aware chunking
│   ├── embeddings.py       # sentence-transformers + hash test embedder
│   ├── store.py            # ChromaDB store + in-memory fallback
│   ├── retrieval.py        # BM25 + dense, RRF fusion, reranker, refusal signals
│   ├── metrics.py          # local eval metrics (lexical proxies labelled)
│   ├── pipeline.py         # ingest, allow-list, refusal, cited generation
│   ├── eval_utils.py       # golden-set loading, eval runner, baseline gate
│   ├── logging.py          # JSONL step log
│   └── tracing.py          # optional Langfuse tracing (off by default)
├── app/
│   ├── main.py             # FastAPI backend (POST /ask, GET /corpus, /health, /metrics)
│   └── streamlit_app.py    # Streamlit UI
├── scripts/
│   ├── download_corpus.py  # official PDF download + verification + manifest
│   ├── build_index.py      # ingest a corpus directory into the store
│   ├── run_eval.py         # golden-set eval runner (+ --check-baseline gate)
│   └── smoke_test.py       # 3-question end-to-end check (mock, no key)
├── eval/
│   ├── golden_set.json     # 50 questions (15 factual / 12 multi-doc / 12 unanswerable / 11 adversarial)
│   ├── poison_chunk.json   # planted contradictory chunk (test fixture only)
│   ├── baseline.json       # mock baseline for the CI gate
│   └── results.json        # latest eval output
├── data/sample_corpus/     # built-in sample corpus (35 chunks, zero setup)
├── tests/                  # 32 tests: chunking, retrieval, citations, refusal, metrics, pipeline, gate
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml   # tests + smoke test + mock eval + baseline gate on every push
├── .env.example
├── README.md
└── BUILD-GUIDE.md
```

## API

- `GET /health` — status, chunks indexed, documents, whether a real LLM is configured
- `GET /corpus` — documents and chunk counts in the index
- `POST /ask` — `{"question": "..."}` → full result: `status` (`answered` /
  `refused` / `failed`), `answer`, `citations`, `retrieved` (chunks +
  scores), `poison_retrieved`, `tokens_used`, `cost_usd`, `latency_seconds`
- `POST /rebuild` — rebuild the index from the configured corpus
- `GET /metrics` — session totals

## Tests

```bash
python -m pytest -q
```

32 tests, no API key, no network: chunking, hybrid retrieval ranking,
citation parsing/validation, refusal (including the documented near-miss),
metrics, the full pipeline with the mock LLM (including "poison retrieved,
never cited"), and the baseline gate (pass / fail-on-regression / tolerance).

## License

MIT.
