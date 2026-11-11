# BUILD-GUIDE — RAG System with Failure Analysis + LLM Eval Harness

A step-by-step guide for building, running, evaluating and deploying this
project. Written for a **Windows laptop with Python 3.12** (the author's
machine is a Tecno MegaBook T1, no GPU — nothing here needs one). macOS/Linux
notes are given where a command differs.

> **Do not put any API key in the code.** Keys live only in the `.env` file
> (which is git-ignored) or in the environment. If you ever paste a key into
> a chat, a file, or a screenshot, rotate it.

---

## 1. Prerequisites

| What | How to check | Install if missing |
| --- | --- | --- |
| Python 3.10+ (3.12 used here) | `python --version` in PowerShell | [python.org](https://www.python.org/downloads/) — tick **"Add Python to PATH"** during install |
| pip | `pip --version` | comes with Python |
| Git | `git --version` | [git-scm.com](https://git-scm.com/download/win) |
| A code editor | — | VS Code is fine |

You do **not** need Docker, a GPU, or a database server to run this locally.
ChromaDB stores its data in files under `data/chroma/`; SQLite is not used.

## 2. Get the project

```powershell
# If you have the project as a folder already, just cd into it:
cd C:\Users\anasr\path\to\rag-eval-harness

# Otherwise clone your repo:
git clone https://github.com/ANAS-alt-ctr/rag-eval-harness.git
cd rag-eval-harness
```

## 3. Create a virtual environment and install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

This installs the heavy packages too: `sentence-transformers` (which pulls
PyTorch CPU, ~200 MB) and `chromadb`. On a slow connection this step takes
several minutes — that is normal. The first run also downloads the
embedding model (~90 MB) and, if enabled, the reranker (~80 MB) from
Hugging Face; they are cached in `~/.cache/huggingface` afterwards.

Verify the install:

```powershell
python -c "import fastapi, streamlit, chromadb, sentence_transformers, rank_bm25, pypdf; print('imports OK')"
python -m pytest -q
```

Expected: **32 passed**. The tests use the built-in mock LLM and hash
embeddings — no API key, no network, no model download.

## 4. Configure the LLM (environment variables only)

```powershell
copy .env.example .env
notepad .env
```

Fill in at least:

```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<redacted>
LLM_MODEL=gpt-4o-mini
```

**How to get a key:**

- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  → Create new secret key → paste it into `.env`. Set a usage limit on your
  OpenAI account (Billing → Usage limits) — $5 is plenty for this project.
- **Any OpenAI-compatible gateway** (Azure OpenAI, a local server, another
  provider): change `LLM_BASE_URL` and `LLM_MODEL` to that provider's chat
  completions endpoint and model name. The client only speaks the
  `/chat/completions` API, so anything compatible works.

Without a key, the API and UI run with the **mock LLM** and say so on
`/health` and in the sidebar. The mock answers from the retrieved context,
which is enough to demo retrieval, refusal and citations — but its numbers
are plumbing checks, not model results (see README).

## 5. Run the smoke test

```powershell
python scripts/smoke_test.py
```

Expected: **SMOKE TEST PASSED (3/3)** — one factual question answered with
a citation, one unanswerable question refused, one adversarial question
answered from the real docs with the poison chunk retrieved but never cited.

## 6. Run the API and the UI

```powershell
# Terminal 1 — API on http://localhost:8000, docs at /docs
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI on http://localhost:8501
streamlit run app/streamlit_app.py
```

Try these in the UI (or `POST /ask`):

| Question | Expected behaviour |
| --- | --- |
| What is the maximum fine for breaching the EU AI Act's prohibited practices? | Answered, citing `[EU-AI-ACT p.6]` |
| What are the four functions of the NIST AI Risk Management Framework? | Answered, citing the NIST chunks |
| What is LLM01 in the OWASP Top 10 for LLM Applications 2025? | Answered, citing `[OWASP-TOP10-LLM p.2]` |
| What is the capital of France? | Refused — "I don't have information about that in the provided documents." |

The UI shows the retrieved chunks with their RRF scores, so you can see
*why* an answer was given or refused.

## 7. Download the full corpus and build the real index

The sample corpus is a summary. The real corpus is the three official PDFs:

```powershell
python scripts/download_corpus.py
```

This downloads the EU AI Act (EUR-Lex), the NIST AI RMF 1.0 (via its DOI,
with an archive.org fallback — NIST restructured its download site in 2026
and the old direct link 404s), and the OWASP Top 10 for LLM 2025 (GitHub
release asset). Each file is checked (HTTP 200, PDF content type, sensible
size) and recorded in `data/corpus/manifest.json` with its SHA-256. If a
download fails, the script says which URL failed — download that PDF by hand
from the official page listed in the script's docstring and put it in
`data/corpus/`.

Then build the index (first build downloads the embedding model if needed):

```powershell
python scripts/build_index.py --corpus data/corpus
```

Expected output: the chunk count and the document list. The index lives in
`data/chroma/` and persists between runs. To rebuild from scratch, re-run
the same command (it resets the collection first).

Point the API/UI at the full corpus by setting `CORPUS_DIR=data/corpus` in
`.env` — or, simpler for a first try, run the eval against it (next section)
before changing the default.

## 8. Run the eval

**Mock baseline (no key, ~10 seconds):**

```powershell
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison
```

This prints the per-category table and writes `eval/results.json`. The
numbers should match `eval/baseline.json` (the stored mock baseline) within
the 0.05 tolerance — verify with the gate:

```powershell
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison --check-baseline
```

Expected: **Baseline check passed.**

**Real-model eval (needs a key, a few minutes, a few cents):**

```powershell
python scripts/run_eval.py --llm openai --embedder st --store chroma --poison --out eval/results-real.json
```

This uses the real embedding model, the ChromaDB store and your configured
LLM. It is the run whose numbers belong in the README's placeholder table
and, eventually, on the CV. Expect it to take a few minutes (50 questions,
one LLM call each, plus embedding the corpus on first run).

**What to look at first in the results:** context recall per category
(did it retrieve the right chunks?), refusal rate on unanswerable (did it
refuse?), and poison_citation_rate on adversarial (must be 0). Faithfulness
is a lexical proxy — read it alongside a RAGAS run (§10), not alone.

## 9. Extending the golden set to 100 questions

1. Add entries to `eval/golden_set.json` with the same shape:
   `id`, `category` (`factual` / `multi_doc` / `unanswerable` /
   `adversarial`), `question`, `gold_answer`, `gold_chunk_ids`,
   `expected` (`answer` or `refuse`), and for adversarial questions
   `poison_chunk_id`.
2. Keep the split balanced and keep unanswerable + adversarial questions
   in the set — they are the failure-analysis half of the project.
3. For the **sample corpus**, `gold_chunk_ids` are chunk ids like
   `EU-AI-ACT#8` (list them with `python scripts/build_index.py` output
   or by reading the corpus files). For the **full PDFs**, chunk ids
   change: rebuild the index, then find the right chunk ids by asking the
   system the question and reading the retrieved chunk ids from the trace
   (`logs/queries.jsonl`), and record those as gold.
4. Re-run the eval and update `eval/baseline.json` with the new results
   (copy `eval/results.json` over it) **only after checking the new
   numbers by hand** — the baseline is a record of measured behaviour,
   not a target to tune toward.

**Green-to-red proof (do this once, so you trust the gate):**

```powershell
# 1. Deliberately break the allow-list: in rag_eval_harness/pipeline.py,
#    comment out the `trusted = [c for c in candidates if c.chunk.trusted]`
#    filter (or set REFUSAL_MIN_OVERLAP=0.0 in .env to disable refusal).
# 2. Run the eval with the gate and watch it fail, naming the regressed metric:
python scripts/run_eval.py --llm mock --embedder hash --store memory --poison --check-baseline
# 3. Revert the change and confirm the gate passes again.
```

## 10. Second opinion: RAGAS / DeepEval (optional)

The local metrics are lexical proxies. To quote a stronger faithfulness
number, run the same golden set through RAGAS (LLM judge):

```powershell
pip install ragas datasets
```

Then build a RAGAS dataset from `eval/golden_set.json` + the eval trace:
for each question, `question`, `contexts` (the retrieved chunk texts from
`logs/queries.jsonl` or from re-running the pipeline), `answer` (the
generated answer), `ground_truth` (the gold answer). Evaluate with
`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
Report the RAGAS numbers next to the local ones in the README — if they
disagree, the disagreement is a finding, not an error to hide.

## 11. Langfuse tracing (optional, off by default)

```powershell
pip install langfuse
```

Set in `.env`:

```ini
LANGFUSE_ENABLED=1
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Every query is then also sent to Langfuse as a trace. With
`LANGFUSE_ENABLED=0` (the default) the pipeline works identically and logs
only to the local JSONL file.

## 12. Docker

```powershell
docker compose up --build
# API at http://localhost:8000, UI at http://localhost:8501
```

Set `LLM_API_KEY` in `.env` first. The image pre-downloads the embedding
model at build time so the first query is fast.

## 13. Deploying the demo (free options)

The honest version: this is a demo, not a production service, and the free
tiers below change their terms — check the linked pages at deploy time.

- **Streamlit Cloud (free tier):** deploy the UI from your repo; set the
  secrets (`LLM_API_KEY` etc.) in the app settings, never in the repo.
  The app rebuilds the index from the sample corpus on first run.
- **Hugging Face Spaces (free CPU tier):** same idea, with a `requirements.txt`
  and an `app.py` that launches Streamlit. Good for a live link on the CV.
- **Render / Railway (free tier):** run the FastAPI backend with
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`; the Streamlit UI can
  point at it, or run Streamlit alone.

Whichever you pick: **add authentication before sharing the link publicly**
(the API has none), and set a low `LLM_SESSION_TOKEN_BUDGET` so a visitor
cannot run up your bill.

## 14. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'dotenv'` (or similar) | You are using the system Python instead of the venv. Activate it first (`.venv\Scripts\activate`), or use `.venv\Scripts\python.exe`. |
| `pip install` fails with `No space left on device` | pip unpacks wheels in the system temp dir, which can be tiny (512 MB on some VMs). Point it elsewhere: `set TMPDIR=C:\Users\anasr\pip-tmp` (PowerShell: `$env:TMPDIR = "C:\Users\anasr\pip-tmp"`) and add `--no-cache-dir` to the pip install. |
| `sentence-transformers` import is slow on first use | It downloads the model (~90 MB) on first use. Check `~/.cache/huggingface` — if the download was interrupted, delete the partial model folder and retry. |
| `chromadb` fails to import | Use the in-memory store instead: set `VECTOR_STORE=memory` in `.env`. You lose persistence (rebuild the index each start), but everything else works. |
| The API answers with the mock LLM even though you set a key | The `.env` file is not in the project root, or the key line has quotes/spaces around it. `LLM_API_KEY` must be non-empty; restart uvicorn after editing `.env`. |
| Eval refusal rate is 0.50 and you expected higher | Read the per-question rows in `eval/results.json`: the six answered unanswerable questions are the documented near-misses (README, "Failure analysis"). They need a reranker-score threshold or an LLM judge — that is the next step, not a configuration error. |
| `download_corpus.py` fails on the NIST PDF | NIST restructured nvlpubs in 2026. The script tries the DOI, then the direct URL, then archive.org. If all three fail, download NIST-AI-100-1 from [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework) by hand into `data/corpus/nist-ai-rmf.pdf`. |
| Tests fail after you change the corpus | The golden set and several tests are written against the sample corpus chunk ids. If you change `data/sample_corpus/`, re-check the chunk ids (`python scripts/build_index.py --corpus data/sample_corpus`) and update `eval/golden_set.json` and the tests that assert exact counts. |

## 15. What to do next (in order)

1. Run the real-model eval (§8) and fill in the README placeholder table.
2. Enable the reranker (`RERANKER=on`, the default) for the real-model run
   and compare context recall with it on vs off — that is a clean,
   measurable improvement to report.
3. Add 50 more golden questions (§9) to reach 100.
4. Run RAGAS (§10) and report both sets of numbers.
5. Deploy the Streamlit demo (§13) and put the live link on the CV.
6. Pick one failure from the README's failure analysis (the near-miss
   refusals are the best story) and fix it with a reranker-score
   threshold or an LLM-as-judge refusal check — then show the before/after
   numbers in the eval table.
