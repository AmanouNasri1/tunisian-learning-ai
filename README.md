# BacPilot AI — Exam Intelligence Layer

AI-powered exam-preparation backend for the Tunisian Baccalauréat.

**Current scope (intentionally narrow):**
- Bac Mathématiques (section)
- Bac Sciences Expérimentales (section)

This repository is **only** the exam-intelligence and AI/data layer. It does NOT contain
(and must not grow): frontend, payments, teacher marketplace, video platform, model fine-tuning.

## What this is

A structured Bac knowledge database + retrieval (RAG) + a strong reasoning LLM.
We do **not** train or fine-tune a model. We transform Bac PDFs and corrections into a
clean, tagged, curriculum-weighted database, and let a strong LLM reason over retrieved,
**cited** sources. If sources are weak or missing, the AI says so — it does not hallucinate.

## Stack

- Django 5.1 + Django REST Framework
- PostgreSQL + pgvector
- LLM via swappable provider (Claude / OpenAI) — `ai/llm_client.py`
- Embeddings via swappable provider (must handle FR / AR / Darja) — `ai/embeddings.py`

---

## Local setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ with the **pgvector** extension available
  (package `postgresql-NN-pgvector` on Linux, or the Windows pgvector build).

### 1. Create the database (PostgreSQL)

Using `psql` as a superuser:

```sql
CREATE DATABASE bacpilot;
CREATE USER bacpilot WITH PASSWORD 'bacpilot';
GRANT ALL PRIVILEGES ON DATABASE bacpilot TO bacpilot;
\c bacpilot
CREATE EXTENSION IF NOT EXISTS vector;
-- give the app user rights on the public schema (PostgreSQL 15+):
GRANT ALL ON SCHEMA public TO bacpilot;
```

> The first migration also runs `CREATE EXTENSION IF NOT EXISTS vector;`. That only
> works if the connecting user is a superuser. If your app user is not a superuser,
> run the `CREATE EXTENSION` line above manually first (as a superuser).

### 2a. Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # then edit .env (DATABASE_URL, keys)
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py createsuperuser
python manage.py smoke_test_exam_intelligence
python manage.py runserver
```

### 2b. Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # then edit .env (DATABASE_URL, keys)
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py createsuperuser
python manage.py smoke_test_exam_intelligence
python manage.py runserver
```

Then open:
- Admin: http://127.0.0.1:8000/admin/
- Read-only API: http://127.0.0.1:8000/api/ (sections, subjects, chapters, concepts, exams, exercises)
- RAG context API: http://127.0.0.1:8000/api/rag/context/?q=fonction
- Tutor API: POST http://127.0.0.1:8000/api/tutor/ask/

### Quick schema check without PostgreSQL (optional)

The core schema migrates on SQLite too (the pgvector HNSW index is skipped on
non-Postgres backends). Useful for a fast local sanity check only — **not** for real
use, because vector search needs Postgres+pgvector:

```bash
# bash
DATABASE_URL='sqlite:///dev.sqlite3' python manage.py migrate
DATABASE_URL='sqlite:///dev.sqlite3' python manage.py smoke_test_exam_intelligence
```

```powershell
# PowerShell
$env:DATABASE_URL='sqlite:///dev.sqlite3'; python manage.py migrate
$env:DATABASE_URL='sqlite:///dev.sqlite3'; python manage.py smoke_test_exam_intelligence
```

---

## Management commands

| Command | Purpose |
|---|---|
| `load_reference_data <fixture.json>` | Load sections, subjects, coefficients, eras (idempotent). |
| `load_example_exercises <dir-or-file>` | Load processed-exercise JSON into the structured DB (idempotent). |
| `prepare_embedding_chunks [--mock]` | Create `EmbeddingChunk` rows from loaded exam data (idempotent). Default: status `pending`, no vectors, no API calls. `--mock` fills deterministic placeholder vectors (NOT real embeddings). |
| `embed_chunks [--provider mock\|openai] [--limit N] [--force] [--dry-run]` | Embed eligible chunks. Default with no provider is dry-run only. `--provider mock` writes deterministic local vectors. `--provider openai --limit 20` is the explicit real-embedding path and requires `OPENAI_API_KEY`. |
| `smoke_test_exam_intelligence` | Pass/fail report on DB, pgvector, tables, data, services. |
| `smoke_test_api` | Hits the read-only API via the in-process test client; checks status + shape. |
| `smoke_test_retrieval` | Real pgvector **hybrid** search (vector + keyword) when PostgreSQL + pgvector + ready embeddings are present; otherwise keyword fallback. Prints accurate diagnostics + the precise fallback reason. No paid APIs (query uses the deterministic mock embedder to match `--mock` chunks). |
| `smoke_test_rag_context` | Builds structured RAG context packages for representative queries. No LLM calls and no paid APIs. |
| `ask_tutor "<question>" [--provider mock\|openai\|anthropic] [--section CODE] [--subject CODE] [--chapter CODE] [--top-k N] [--json]` | Ask the source-grounded tutor. Default provider is safe deterministic `mock`; real providers require explicit selection and API keys. |
| `smoke_test_tutor` | Exercises grounded mock tutor answers and an out-of-scope refusal. No paid APIs. |
| `evaluate_tutor [--cases evaluation/tutor_cases.yaml] [--json] [--fail-under 0.8] [--verbose]` | Runs deterministic golden tutor cases with the mock provider. No LLM judge and no paid APIs. |

### Embedding workflow

Local/test workflow, no paid APIs:

```powershell
python manage.py prepare_embedding_chunks
python manage.py embed_chunks
python manage.py embed_chunks --provider mock --limit 20
python manage.py prepare_embedding_chunks --mock
python manage.py smoke_test_rag_context
```

Real OpenAI embedding workflow, explicit opt-in:

```powershell
$env:OPENAI_API_KEY='sk-...'
$env:EMBEDDING_MODEL='text-embedding-3-small'
python manage.py embed_chunks --provider openai --limit 20
python manage.py embed_chunks --provider openai --force --limit 20
```

If `OPENAI_API_KEY` is missing, `embed_chunks --provider openai` prints a clear warning and exits without writing fake success. Tests and smoke tests use mock/keyword paths only.

### RAG context package

`rag/context_builder.py` assembles retrieval context for the future tutor. It returns the original query, filters, retrieval mode, selected chunks, grouped context, citations, and warnings such as weak retrieval, missing correction/rubric, or mock embeddings in use.

It does not call an LLM. It also does not call OpenAI embeddings implicitly; vector retrieval is used only for the deterministic mock-vector smoke path unless an explicit caller injects a real query embedder in the future.

### Source-grounded tutor

The first tutor layer lives in `rag/tutor.py`. It consumes `rag/context_builder.py`, checks
whether retrieved context is answerable, returns citations, and writes an `AIInteraction`
audit row for both answers and refusals.

Safe mock command:

```powershell
python manage.py ask_tutor "Explique la loi binomiale" --provider mock
python manage.py ask_tutor "Donne-moi une recette de pizza" --provider mock
python manage.py smoke_test_tutor
```

API example:

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/tutor/ask/" `
  -ContentType "application/json" `
  -Body '{"query":"Explique la loi binomiale","provider":"mock","section":"SC_EXP","subject":"MATH","chapter":"PROBA"}'
```

Refusal behavior: the tutor refuses when no chunks/citations are retrieved, when keyword
evidence is absent, or when the meaningful words in the student question are not grounded in
the selected context. The refusal is structured and logged instead of hallucinating.

Real LLM providers are explicit only:

```powershell
$env:OPENAI_API_KEY='sk-...'
$env:LLM_MODEL='gpt-4o'
python manage.py ask_tutor "Explique la loi binomiale" --provider openai --section SC_EXP --subject MATH --chapter PROBA
```

Mock embeddings and the mock tutor are not production semantic AI; they are deterministic
local safety paths for testing retrieval, grounding, citations, and audit logging.

### Tutor evaluation

Golden tutor cases live in `evaluation/tutor_cases.yaml`. They check whether mock tutor
answers/refusals are grounded before any real LLM provider is enabled.

Run:

```powershell
python manage.py evaluate_tutor --verbose
python manage.py evaluate_tutor --json
python manage.py evaluate_tutor --verbose --fail-under 0.8
```

Add a case by appending YAML like:

```yaml
- id: new_case_id
  query: "Explique ..."
  section: "SC_EXP"
  subject: "MATH"
  chapter: "PROBA"
  expected_refused: false
  expected_subject: "MATH"
  expected_chapter: "PROBA"
  required_terms: ["terme"]
  forbidden_terms: ["pizza"]
  required_citation_chapters: ["PROBA"]
  minimum_citations: 2
```

The evaluator checks refusal correctness, citations, citation chapters, required/forbidden
terms, diagnostics, provider metadata, and that only `provider=mock` is used. It is not an
LLM judge; it is a deterministic safety gate.

## Testing & verification

> **SQLite is for code sanity only. PostgreSQL + pgvector is the real acceptance gate.**
> Vector search (pgvector) does not run on SQLite — the retrieval smoke test falls back
> to keyword/metadata matching and says so.

### Quick sanity pass on SQLite (no PostgreSQL needed)

```bash
# bash
export DATABASE_URL='sqlite:///dev.sqlite3'
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py prepare_embedding_chunks
python manage.py embed_chunks --provider mock --limit 20
python manage.py smoke_test_exam_intelligence
python manage.py smoke_test_api
python manage.py smoke_test_retrieval
python manage.py smoke_test_rag_context
python manage.py smoke_test_tutor
python manage.py evaluate_tutor --verbose
python manage.py test backend.exam_intelligence
```

```powershell
# PowerShell
$env:DATABASE_URL='sqlite:///dev.sqlite3'
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py prepare_embedding_chunks
python manage.py embed_chunks --provider mock --limit 20
python manage.py smoke_test_exam_intelligence
python manage.py smoke_test_api
python manage.py smoke_test_retrieval
python manage.py smoke_test_rag_context
python manage.py smoke_test_tutor
python manage.py evaluate_tutor --verbose
python manage.py test backend.exam_intelligence
```

### Real verification on PostgreSQL + pgvector (the acceptance gate)

Use the Docker flow below (or a local Postgres). Then run the **same** commands without
the `DATABASE_URL=sqlite...` override (point `DATABASE_URL` at Postgres in `.env`).
On Postgres, `smoke_test_exam_intelligence` additionally confirms the `vector` extension,
and the HNSW index is created.

```bash
# Postgres via Docker (once Docker is available)
docker run --name bacpilot-pg -e POSTGRES_DB=bacpilot \
  -e POSTGRES_USER=bacpilot -e POSTGRES_PASSWORD=bacpilot \
  -p 5432:5432 -d pgvector/pgvector:pg16
# .env: DATABASE_URL=postgres://bacpilot:bacpilot@localhost:5432/bacpilot
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py prepare_embedding_chunks
python manage.py embed_chunks --provider mock --limit 20
python manage.py smoke_test_exam_intelligence   # pgvector check should PASS here
```

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Django project (settings, urls, wsgi/asgi) |
| `backend/exam_intelligence/` | Core app: models, admin, API, services, management commands |
| `ingestion/` | PDF → structured JSON pipeline (digital works; OCR stubbed) |
| `ai/` | LLM + embedding abstractions, prompt templates |
| `rag/` | Hybrid retrieval, RAG context assembly, and source-grounded tutor service |
| `evaluation/` | Golden-set format + metrics (cases authored, runner TBD) |
| `seed_data/` | JSON schema, worked examples, reference fixture |
| `docs/` | Architecture, data model, sprint report |

## Known limitations / not done yet

- **Real embeddings are opt-in only.** `prepare_embedding_chunks` creates chunks with
  status `pending` (text + metadata, no vector). `--mock` or
  `embed_chunks --provider mock` fills DETERMINISTIC placeholder
  vectors and marks them `ready` (transparent via `model_name='mock-deterministic-v1'`) —
  enough to exercise the real pgvector path, but NOT semantically meaningful. Real OpenAI
  embeddings require `embed_chunks --provider openai ...` plus `OPENAI_API_KEY`.
- **RAG retrieval backends are implemented** (`rag/retriever.py`): pgvector cosine vector
  search + portable keyword search, fused with RRF. Vector search runs only on PostgreSQL +
  pgvector with `ready` embeddings; reranker is still optional/none.
- **RAG context assembly and tutor answering are implemented** (`rag/context_builder.py`,
  `rag/tutor.py`). The tutor is source-grounded, writes `AIInteraction` audit rows, and
  defaults to deterministic mock output. Real LLM calls require explicit provider selection.
- **Tutor evaluation is deterministic, not a teacher replacement.** `evaluate_tutor` checks
  grounding, citations, refusal behavior, and required/forbidden terms; a teacher-authored
  larger golden set is still needed before production use.
- **OCR** for scanned PDFs is stubbed; digital-PDF extraction works.
- **No correction engine or production tutor UI yet** - only read-only reference browsing,
  retrieval-context preview, and a backend tutor answer endpoint.
- **Coefficients and curriculum-era boundaries are placeholders** — verify with a Tunisian
  Bac teacher before trusting any readiness score.
- `migrate` against **real PostgreSQL + pgvector** has not been run yet (no Postgres in the
  build environment); everything was verified end-to-end on SQLite, which skips pgvector.
  PostgreSQL + pgvector remains the real acceptance gate. See `docs/sprint_report.md`.

## Not building (deliberately)

Frontend · payments · marketplace · video · handwriting OCR · fine-tuning.
