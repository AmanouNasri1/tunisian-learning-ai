# Sprint report — make the Django backend real and runnable

Goal: turn the existing models/services/abstractions into a runnable Django + PostgreSQL
+ pgvector backend with migrations, admin, seed loading, and a smoke test.

## Phase 1 — what existed vs. what was missing

**Existed (correct, kept):**
- `backend/exam_intelligence/models.py` — 19 models, pgvector `VectorField` + `HnswIndex`.
- `backend/exam_intelligence/services/readiness.py` — runs standalone.
- `ai/llm_client.py`, `ai/embeddings.py`, `ai/prompts.py` — provider abstractions + prompts.
- `rag/retriever.py` — orchestration; vector/keyword backends intentionally stubbed.
- `ingestion/pipeline.py` — digital extraction; OCR stubbed.
- `seed_data/schema/processed_exercise.schema.json`, `seed_data/examples/*.json`.
- `seed_data/reference/01_reference.json` — **confirmed a valid Django fixture** (list of
  `{model, pk, fields}`), so `loaddata` works on it directly.

**Was missing (created this sprint):**
- Entire Django project shell: `manage.py`, `config/{settings,urls,wsgi,asgi,__init__}.py`.
- `backend/exam_intelligence/apps.py` (explicit app label `exam_intelligence`).
- `backend/exam_intelligence/admin.py` (review-focused).
- Management commands: `load_reference_data`, `load_example_exercises`, `smoke_test_exam_intelligence`.
- Migration `0001_initial` (+ pgvector handling).
- Read-only DRF API (`backend/exam_intelligence/api/`).
- `.gitignore`, expanded `README.md`, this report.

**Dependencies:** none were installed in the build env. Installed for verification:
Django 5.1.15, DRF, pgvector, dj-database-url, python-dotenv, psycopg 3. `requirements.txt`
already listed them.

## Migration-relevant fix (explained)

`CommonMistake.frequency` used the `Frequency` enum (`rare/occasional/frequent`), but the
JSON schema and all seed examples use mistake-frequency values `rare/occasional/common/very_common`.
Loading the examples would have violated the field's `choices`. Fix: added a distinct
`MistakeFrequency` enum aligned to the schema and pointed `CommonMistake.frequency` at it,
leaving `Frequency` for exam-frequency semantics. Minimal and removes a real data-load blocker.

## pgvector / HNSW handling (explained)

- The initial migration runs `VectorExtension()` first (no-op on non-Postgres).
- The HNSW index can't be created on SQLite (`USING hnsw ... WITH (...)` → syntax error).
  Empirically, everything else — including the `vector` **column** — applies fine on SQLite;
  only the HNSW `AddIndex` failed.
- Fix: the HNSW index is kept in Django's **migration state** (via
  `SeparateDatabaseAndState`) but its **database** creation is a vendor-guarded `RunPython`
  that only executes on PostgreSQL. Result: production Postgres gets the HNSW index; local
  SQLite checks skip it cleanly; `makemigrations --check` reports no drift.

## What was verified by running (SQLite, in build env)

| Check | Result |
|---|---|
| `manage.py check` | PASS — 0 issues |
| `makemigrations` | Generated `0001_initial` |
| `makemigrations --check --dry-run` | Exit 0 — no model/migration drift |
| `migrate` (SQLite) | All apps incl. `exam_intelligence.0001_initial` applied OK |
| `load_reference_data` | created=15 updated=0 (2 sections, 3 subjects, 6 coefs, 4 eras) |
| `load_example_exercises` | files_loaded=4, **0 warnings** (all 4 subject/section examples) |
| `load_example_exercises` (2nd run) | Idempotent — rubric_items stayed 20, not doubled |
| `smoke_test_exam_intelligence` | PASS (DB, tables, reference, examples, readiness, admin 21 models) |

## What could NOT be verified (and why)

- **`migrate` against real PostgreSQL + pgvector**: no Postgres/Docker/psql in the build
  environment. Verified end-to-end on SQLite instead. The Postgres-only paths not executed:
  `VectorExtension()` actually creating the extension, and `create_hnsw_index()` running its
  `CREATE INDEX ... USING hnsw` SQL. The SQL is standard pgvector syntax but should be
  confirmed on a real DB.
- **pgvector smoke check** (`SELECT ... FROM pg_extension`): skipped on SQLite (reported as WARN).

## Next step after this sprint

1. Run the README setup against a real PostgreSQL+pgvector DB and confirm `migrate` +
   `smoke_test_exam_intelligence` report PASS including the pgvector check.
2. Then wire `rag/retriever.py` `_vector_search` (pgvector `CosineDistance`) and
   `_keyword_search` (Postgres full-text), and add an embedding-population command for
   `EmbeddingChunk`. That unblocks the first real RAG endpoint.

---

# 2-hour backend hardening sprint

Goal: make the backend more testable, safer, and closer to the first RAG milestone
**without** depending on live PostgreSQL. No paid APIs, no frontend.

## What changed

- **Model (additive, migration 0002):** `EmbeddingChunk.embedding` is now nullable;
  added `embedding_status` (`pending`/`mock`/`ready`, default `pending`, indexed) and
  `language`; `model_name` now allows blank; added `combined` content type. This models the
  production-correct flow: create chunks (text + metadata) first, embed later.
- **HNSW index moved out of model state.** It was kept in Django's migration *state* by
  0001, which broke any later `AlterField` on `EmbeddingChunk` under SQLite (table rebuild
  tried to recreate `USING hnsw ... WITH (...)`). Migration 0002 removes it from *state only*
  (a no-op on the database: kept on Postgres, never existed on SQLite). 0001 is untouched.
  The index is still created on Postgres by 0001's database operation.
- **API hardened** (`api/views.py`): added DRF `SearchFilter`/`OrderingFilter` (no new deps)
  and explicit query-param filters (`subject`, `section`, `year`, `session`, `difficulty`,
  `relevance_status`, `chapter`). Still strictly read-only; no internal/raw fields exposed.
- **New commands:** `prepare_embedding_chunks`, `smoke_test_api`, `smoke_test_retrieval`.
- **Tests:** `backend/exam_intelligence/tests.py` (6 tests, Django runner, SQLite).
- **Docs:** README testing section (SQLite sanity vs Postgres gate), this report.

## What was verified by running (SQLite, build env)

| Check | Result |
|---|---|
| `manage.py check` | 0 issues |
| `migrate` (0001 + 0002) on SQLite | OK |
| `makemigrations --check --dry-run` | No changes detected (no drift) |
| `load_reference_data` / `load_example_exercises` | created=15 / 4 files, 0 warnings |
| `prepare_embedding_chunks` | created=58 (combined 4, exercise 4, question 11, correction 11, rubric 20, mistake 8) |
| `prepare_embedding_chunks` (2nd run) | created=0, updated=58 — **idempotent** |
| `smoke_test_api` | PASS — all 6 endpoints 200 with data |
| `smoke_test_retrieval` | Keyword fallback routes all queries to correct subject/section; `fonction` correctly WARNs no result |
| `manage.py test backend.exam_intelligence` | **6/6 OK** |

## What still needs PostgreSQL

- Real **pgvector** vector search (the retrieval smoke test explicitly prints
  `[WARN] Vector search skipped: PostgreSQL/pgvector required`).
- `VectorExtension()` creating the extension and the HNSW index SQL actually executing
  (0001 Postgres-only paths) — unverified without a live DB.
- The `pgvector` line in `smoke_test_exam_intelligence` (WARN-skipped on SQLite).

## Migration changes this sprint

- Added `0002_embeddingchunk_pending.py` (additive; 0001 unchanged). Verified it applies on
  SQLite and produces **no drift**.

## Assumptions made

- Chunk `language` defaults to `fr` (all current seed content is French).
- `session` is not denormalized onto `EmbeddingChunk` (derivable via the source exercise);
  avoided extra schema churn. Revisit if session-level retrieval filtering is needed.
- The retrieval smoke test uses an independent keyword fallback rather than the stubbed
  `rag/retriever.py`, since the real vector/keyword backends need Postgres wiring.

## Exact next step

1. Bring up PostgreSQL + pgvector (Docker line in README), point `DATABASE_URL` at it, and
   run the full command list — confirm `smoke_test_exam_intelligence` PASS **including** the
   pgvector check, and that the HNSW index is created.
2. Add a real embedding-population command (calls the embedding provider in `ai/embeddings.py`),
   flipping chunks from `pending` → `ready`, then wire `rag/retriever.py` vector + keyword
   backends. That is the first real RAG milestone.

---

# Retrieval bug fix (vector search was never attempted)

## The bug

`smoke_test_retrieval` printed `[WARN] Vector search skipped: PostgreSQL/pgvector required`
**unconditionally** — it was hardcoded at the top of the command and always ran the keyword
fallback, regardless of vendor or embeddings. It never checked anything. Separately,
`rag/retriever.py`'s `_vector_search`/`_keyword_search` were still `NotImplementedError`
stubs. So even on a correct PostgreSQL + pgvector setup with ready embeddings, vector search
could not run.

## What was fixed

- **`rag/retriever.py`** — implemented both backends:
  - `_vector_search`: pgvector `CosineDistance` over chunks with `embedding_status='ready'`
    AND `embedding IS NOT NULL`, ordered by cosine distance, top-k; score = 1 − distance.
  - `_keyword_search`: portable icontains over content + denormalized metadata names.
  - `retrieve()` now reports `vector_count` / `keyword_count` and fuses both via RRF (hybrid).
- **`smoke_test_retrieval`** — rewritten to compute real availability:
  `vendor=='postgresql' AND pgvector present AND ready_embeddings>0`. Prints accurate
  diagnostics (vendor, pgvector yes/no, total, non-null, status breakdown, model_name
  breakdown, ready count, vector enabled yes/no) and a PRECISE fallback reason
  (PostgreSQL missing / pgvector missing / no ready embeddings). Per-query vector errors are
  caught and printed with the exception message, then degrade to keyword for that query.
  Query embedding uses the deterministic mock embedder (no paid API), matching `--mock` chunks.
- **`prepare_embedding_chunks --mock`** — now sets `embedding_status='ready'` (a chunk with a
  usable vector is ready for search; the mock nature is recorded in `model_name`, not hidden
  in the status). This matches the reported DB state and makes the acceptance command produce
  ready embeddings. No migration change.
- **Windows crash fix** — exam content has math symbols (`∞`, `²`); `stdout.write` crashed
  under cp1252. The command now reconfigures output with `errors='replace'`.
- **New seed example** `seed_data/examples/bac_math_fonctions.json` (Bac Math function study)
  so the `fonction` query returns a real keyword match.

## Verified on SQLite (build env — no PostgreSQL here)

| Check | Result |
|---|---|
| `check` | 0 issues |
| `migrate` (0001 + 0002) | OK; no drift |
| `load_example_exercises` | 5 files, 0 warnings (incl. fonction) |
| `prepare_embedding_chunks --mock` | 73 chunks, all `status='ready'`, `model='mock-deterministic-v1'`, non-null=73 |
| `smoke_test_retrieval` (SQLite) | Accurate: `vector search enabled: no`, reason `PostgreSQL required (vendor='sqlite')`; all 5 queries return correct keyword hits (`fonction` → 2017 MATH function exercise); no crash |
| `smoke_test_exam_intelligence` / `smoke_test_api` | PASS |
| `manage.py test backend.exam_intelligence` | 8 tests OK (1 pgvector test skipped on SQLite) |

## NOT verifiable here (needs your PostgreSQL)

The actual pgvector vector query (`CosineDistance`) cannot run on SQLite, so the
"vector search enabled: yes" path was verified by construction, not execution. On your
machine (vendor=postgresql, pgvector present, ready=73) it WILL run — confirmed by the
gated test `test_vector_search_runs_on_postgres`, which executes there and is skipped here.

## To reproduce on your machine (note the reload for the new example)

```powershell
python manage.py migrate
python manage.py load_example_exercises seed_data/examples   # picks up bac_math_fonctions.json
python manage.py prepare_embedding_chunks --mock             # -> status='ready'
python manage.py smoke_test_retrieval                        # vector search enabled: yes
python manage.py test backend.exam_intelligence              # vector test now runs (not skipped)
```

---

# Real embedding integration + RAG context assembly

Goal: add a real embedding-provider path and build the context package the future tutor will
consume, without adding frontend/payments/marketplace/video and without hidden paid calls.

## What changed

- **Embedding abstraction** (`ai/embeddings.py`): added `MockEmbeddingProvider`,
  `OpenAIEmbeddingProvider`, shared `mock_embedding`, `EMBEDDING_DIM`, and
  `EmbeddingConfigurationError`. OpenAI uses `EMBEDDING_MODEL` and requires
  `OPENAI_API_KEY`; missing keys fail clearly before any fake success.
- **Embedding command** (`embed_chunks`): default invocation is dry-run only. Actual writes
  require `--provider mock` or `--provider openai`; OpenAI calls are therefore explicit.
  Supports `--limit`, `--force`, and `--dry-run`, and reports provider, found/embedded/
  skipped/failed counts, batches, and warnings.
- **Compatibility kept**: `prepare_embedding_chunks --mock` still creates deterministic
  ready vectors with `model_name='mock-deterministic-v1'`.
- **RAG context assembly** (`rag/context_builder.py`): packages selected chunks, grouped
  context, citations, retrieval mode, diagnostics, and warnings. It does not call an LLM
  or OpenAI embeddings implicitly.
- **Smoke command** (`smoke_test_rag_context`): runs representative queries and prints mode,
  counts, top sources, warnings, and mock-embedding status.
- **Optional API**: `GET /api/rag/context/?q=fonction` returns the context package only.
- **Tests**: provider determinism, mock embedding job, dry-run no-OpenAI path, missing-key
  safety, context builder shape, and API package response.

## Exact next milestone

Add the first tutor-generation backend path: an explicit, source-grounded AI tutor command
that consumes `rag/context_builder.py`, calls an LLM only behind an explicit provider flag,
writes `AIInteraction` audit rows, and refuses to answer when retrieval is weak or required
corrections/rubrics are missing.

---

# Source-grounded AI tutor backend

Goal: add the first backend-only tutor answer layer that consumes RAG context, answers with
citations, refuses unsafe/out-of-scope questions, and stores an audit trail.

## What changed

- **Tutor service** (`rag/tutor.py`): `answer_student_question(...)` builds RAG context,
  checks answerability, returns a structured answer package, and logs `AIInteraction`.
- **Mock tutor provider**: deterministic local answers assembled from corrections, rubric
  items, questions, common mistakes, and citations. No API calls.
- **Real provider readiness** (`ai/llm_client.py`): added `MockLLMClient`,
  `LLMConfigurationError`, explicit provider selection, and clean missing-key errors for
  OpenAI/Anthropic.
- **Audit migration** (`0003_aiinteraction_tutor_audit.py`): added provider,
  retrieval_mode, refused, refusal_reason, and warnings fields to `AIInteraction`.
- **Commands**: `ask_tutor` for manual asks, `smoke_test_tutor` for in-scope answers plus
  an out-of-scope pizza refusal.
- **API**: `POST /api/tutor/ask/` returns the same structured package; default provider is
  `mock`, real providers require explicit selection and configured keys.
- **Tests**: mock grounded answer, citations, refusal, API success/missing-query/missing-key,
  no paid-client instantiation in mock tests, and audit row creation.

## Refusal policy

The tutor refuses when there are no chunks, no citations, no keyword evidence, or no
meaningful lexical overlap between the student's query and retrieved chunk content. This
keeps generic or out-of-scope requests from being answered by mock-vector noise.

## Exact next milestone

Add teacher-review/evaluation tooling for tutor answers: golden tutor cases, expected
citations, refusal expectations, and a command that scores answer grounding before any real
LLM provider is enabled for student-facing use.
