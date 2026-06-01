# SD Learning Assistant — CLAUDE.md

A FastAPI-based RAG chatbot that teaches system design concepts using FAISS vector search + LLM generation.

---

## Architecture

```
User → POST /api/v1/chat
         → IntentService   (classify: CONCEPT_EXPLANATION / SYSTEM_DESIGN_QUESTION / DESIGN_REVIEW / OUT_OF_SCOPE)
         → RetrieverService (FAISS similarity search → context string)
         → OrchestratorService (build prompt → LLM)
         → EvaluatorService (score response; retry on refusal; fallback on low confidence)
         → Response
```

## Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app factory, CORS, lifespan hooks |
| `app/api/routes/chat.py` | POST /api/v1/chat |
| `app/api/routes/admin.py` | Session inspect/clear, RAG probe |
| `app/services/orchestrator_service.py` | Main pipeline coordinator |
| `app/services/intent_service.py` | LLM-based query classifier |
| `app/services/retriever_service.py` | FAISS RAG pipeline |
| `app/services/llm_service.py` | OpenAI / Gemini LLM calls |
| `app/services/evaluator_service.py` | Response scoring + retry logic |
| `app/services/memory_service.py` | In-memory session history (5 turns) |
| `app/services/prompts/system_prompt.txt` | Main generation prompt |
| `app/services/prompts/confidence_scoring.txt` | Evaluator scoring prompt |
| `app/services/prompts/fallback_clarification.txt` | Fallback prompt |
| `scripts/ingest_docs.py` | **Run this to build the FAISS index** |
| `data/faiss_index/<provider>/` | Persisted FAISS index (git-ignored) |

## Supported Providers

The system supports two LLM/embedding providers, selected per-request via `"provider"` field:

| Provider | LLM Model | Embedding Model | Embed Dim |
|----------|-----------|-----------------|-----------|
| `gemini` | `gemini-2.5-flash` | `models/gemini-embedding-001` | 3072 |
| `openai` | `gpt-4-turbo` | `text-embedding-3-small` | 1536 |

**Important:** Each provider has its own separate FAISS index at `data/faiss_index/<provider>/`.
You must ingest for each provider you plan to use.

## FAISS Index — Building & Updating

The FAISS index **must exist** before the retriever can serve context.
The index is **not committed to git** (binary files).

### Build the index from `staticdocs/`

```bash
# Using Gemini embeddings — reads all PDFs/md/txt in staticdocs/
python -m scripts.ingest_docs --provider gemini

# Custom source dir
python -m scripts.ingest_docs --provider gemini --source /path/to/my/docs

# Tune chunk size (larger = fewer chunks = faster but less granular retrieval)
python -m scripts.ingest_docs --provider gemini --chunk-size 800 --overlap 100
```

**Python 3.9+ required** for `google-genai`. The local venv is Python 3.7 — use a system Python:

```bash
# One-time env setup
python3 -m venv /tmp/ingest_env
/tmp/ingest_env/bin/pip install google-genai openai faiss-cpu numpy pypdf

# Run (can take 15-30 min for 60+ PDFs on Gemini free tier)
/tmp/ingest_env/bin/python3 -m scripts.ingest_docs --provider gemini --chunk-size 800

# Run in background (logging to file)
/tmp/ingest_env/bin/python3 -m scripts.ingest_docs --provider gemini --chunk-size 800 \
  > /tmp/ingest_log.txt 2>&1 &
tail -f /tmp/ingest_log.txt
```

### Rate limits on Gemini free tier

Gemini free tier is ~10 RPM. The script handles this automatically with:
- Batch size of 20 chunks per API call
- 7-second sleep between batches
- Exponential backoff (5s → 10s → 20s → 40s → 80s) on 429 errors

For ~3000 chunks (63 docs at chunk-size=800): expect **~25-40 minutes** on free tier.

### Supported file types

| Extension | Reader |
|-----------|--------|
| `.pdf` | `pypdf` (install separately) |
| `.md` | built-in |
| `.txt` | built-in |

### Add new documents

Drop files into `staticdocs/` and re-run `ingest_docs.py`. The index is fully rebuilt.

### Render deployment

The FAISS index **does not persist** between Render deploys (ephemeral filesystem).
To fix this, either:
1. Commit the `data/faiss_index/` directory to the repo (simplest for small indexes)
2. Use Render persistent disk
3. Add a startup script that rebuilds the index on first boot

**Quick fix:** Add `data/` to git and commit the built index:
```bash
# Remove data/ from .gitignore if present, then:
git add data/faiss_index/
git commit -m "Add pre-built FAISS index"
git push
```

## Critical Design Decisions

### System Prompt Philosophy
The system prompt instructs the LLM to **answer from expertise first** — the FAISS context
is supplementary, not the sole source of truth. The model should never say "I don't have
enough information" for well-known system design topics.

### Evaluator Logic
- **Refusal signals** (e.g. "I do not have enough information to answer"): triggers `_retry_without_restriction()` immediately
- **Low score (< 0.6)**: triggers `_trigger_fallback()` which asks clarifying questions
- **Optimistic default**: score defaults to 0.85 (not 1.0) so the evaluator only needs to flag bad responses

### Evaluator Scoring Prompt
A refusal from the LLM should score **0.1–0.3**, not 1.0. The old scoring prompt
incorrectly rewarded "correctly identifying lack of context" — this has been fixed.

## Environment Variables

```env
OPENAI_API_KEY=sk-...      # Required for openai provider
GEMINI_API_KEY=AIza...     # Required for gemini provider
CORS_ORIGINS=              # Optional: comma-separated extra allowed origins
FAISS_INDEX_DIR=           # Optional: override index path (default: data/faiss_index)
```

## API Reference

### POST /api/v1/chat
```json
{
  "message": "What is consistent hashing?",
  "session_id": "uuid-v4",
  "provider": "gemini"
}
```
Response:
```json
{
  "response": "...",
  "intent": "CONCEPT_EXPLANATION",
  "confidence_score": 0.92,
  "is_fallback": false,
  "session_id": "uuid-v4"
}
```

### GET /api/v1/admin/session/{session_id}
Inspect raw session memory (last 5 turns).

### DELETE /api/v1/admin/session/{session_id}
Clear session memory.

### POST /api/v1/admin/retrieve
Debug RAG retrieval — see exactly what context would be injected for a query:
```json
{ "query": "consistent hashing", "top_k": 5 }
```

## Common Gotchas

1. **"No FAISS index found"** → Run `python -m scripts.ingest_docs --provider <provider>`
2. **OpenAI 429 quota error** → The key has no paid credits. Use `--provider gemini` or add billing.
3. **`google-genai` won't install in venv** → The local venv is Python 3.7; `google-genai` needs 3.9+. Use system Python.
4. **`admin.py` crashes on `/admin/retrieve`** → Fixed: was referencing `_pipeline` (singular) instead of `_pipelines[provider]`
5. **Model answers "I don't have enough info"** → Fixed: system prompt now instructs model to use expertise first; evaluator now catches refusals and retries.
6. **Gemini embedding 404** → Fixed: `text-embedding-004` was deprecated. Now using `models/gemini-embedding-001` (dim=3072).

## Running Locally

```bash
# Start server
uvicorn app.main:app --reload --port 8000

# Or with Gunicorn
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:8000
```

## Deployment (Render)

```dockerfile
CMD gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-10000}
```

Workers: 4 is fine for Render's starter tier. Reduce to 2 if OOM errors occur.
