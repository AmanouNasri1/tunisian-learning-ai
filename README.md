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
| `smoke_test_exam_intelligence` | Pass/fail report on DB, pgvector, tables, data, services. |
| `smoke_test_api` | Hits the read-only API via the in-process test client; checks status + shape. |
| `smoke_test_retrieval` | Real pgvector **hybrid** search (vector + keyword) when PostgreSQL + pgvector + ready embeddings are present; otherwise keyword fallback. Prints accurate diagnostics + the precise fallback reason. No paid APIs (query uses the deterministic mock embedder to match `--mock` chunks). |

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
python manage.py smoke_test_exam_intelligence
python manage.py smoke_test_api
python manage.py smoke_test_retrieval
python manage.py test backend.exam_intelligence
```

```powershell
# PowerShell
$env:DATABASE_URL='sqlite:///dev.sqlite3'
python manage.py migrate
python manage.py load_reference_data seed_data/reference/01_reference.json
python manage.py load_example_exercises seed_data/examples
python manage.py prepare_embedding_chunks
python manage.py smoke_test_exam_intelligence
python manage.py smoke_test_api
python manage.py smoke_test_retrieval
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
python manage.py smoke_test_exam_intelligence   # pgvector check should PASS here
```

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Django project (settings, urls, wsgi/asgi) |
| `backend/exam_intelligence/` | Core app: models, admin, API, services, management commands |
| `ingestion/` | PDF → structured JSON pipeline (digital works; OCR stubbed) |
| `ai/` | LLM + embedding abstractions, prompt templates |
| `rag/` | Hybrid retrieval (vector/keyword backends stubbed pending DB wiring) |
| `evaluation/` | Golden-set format + metrics (cases authored, runner TBD) |
| `seed_data/` | JSON schema, worked examples, reference fixture |
| `docs/` | Architecture, data model, sprint report |

## Known limitations / not done yet

- **Real embeddings are not generated.** `prepare_embedding_chunks` creates chunks with
  status `pending` (text + metadata, no vector). `--mock` fills DETERMINISTIC placeholder
  vectors and marks them `ready` (transparent via `model_name='mock-deterministic-v1'`) —
  enough to exercise the real pgvector path, but NOT semantically meaningful. A real
  embedding job (calling an embedding API) is the next step.
- **RAG retrieval backends are implemented** (`rag/retriever.py`): pgvector cosine vector
  search + portable keyword search, fused with RRF. Vector search runs only on PostgreSQL +
  pgvector with `ready` embeddings; reranker is still optional/none.
- **OCR** for scanned PDFs is stubbed; digital-PDF extraction works.
- **No AI/RAG/correction HTTP endpoints yet** — only read-only reference browsing.
- **Coefficients and curriculum-era boundaries are placeholders** — verify with a Tunisian
  Bac teacher before trusting any readiness score.
- `migrate` against **real PostgreSQL + pgvector** has not been run yet (no Postgres in the
  build environment); everything was verified end-to-end on SQLite, which skips pgvector.
  PostgreSQL + pgvector remains the real acceptance gate. See `docs/sprint_report.md`.

## Not building (deliberately)

Frontend · payments · marketplace · video · handwriting OCR · fine-tuning.
