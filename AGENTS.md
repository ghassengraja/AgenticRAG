# AgenticRAG repository guide

## Workflow

- The current integration work is on `dev`. Check `git status --short --branch` before editing because the working tree may contain work from earlier sessions.
- Use uv from the repository root. `.python-version` selects Python 3.12, `pyproject.toml` declares dependencies, and `uv.lock` pins them. Use `uv add` or `uv remove` for dependency changes. Jupyter is in the development group.
- Django loads `agenticRag/.env` next to `manage.py`. It is Git-ignored and holds environment-specific URLs and any credentials. Keep its contents out of commits and command output.

## Entry points and data

| Path | Responsibility |
| --- | --- |
| `agenticRag/agenticRag/api.py` and `urls.py` | Django's OpenAI-compatible `/v1/chat/completions`, `/v1/models`, and `/v1/` endpoints |
| `agenticRag/ai/supervisor.py` and `agents.py` | LangGraph supervisor and the document, RAG, and memory agents |
| `agenticRag/ai/llm.py` | `get_model()` creates the chat client. `get_embedding_model()` is shared by PDF indexing, PDF retrieval, and episodic memory. |
| `agenticRag/ai/indexer.py` and `tools_retriever.py` | Build and query Chroma collection `docs` under `agenticRag/chroma_db_docs`. The PDF source path is currently hard-coded to a Windows directory in `indexer.py`. |
| `agenticRag/ai/tools_memory.py` and `vectorstore.py` | Store episodic facts and context in SQLite and index facts in Chroma collection `episodic_memory` under `agenticRag/chroma_db_episodic_memory`. Vector searches filter by `owner_id`. |
| `agenticRag/ai/short_term_mem.py` | Process-local LangGraph `MemorySaver`. Django uses a thread ID derived from the user ID, and checkpoints are lost on restart. Redis integration is pending. |
| `agenticRag/documents/models.py` and `agenticRag/ai/tools_documents.py` | Django ORM models and tools for user-owned documents. The default database is `agenticRag/db.sqlite3`. |

`agenticRag/ai/rag.py` is an unused prototype with a pre-existing syntax error. The active RAG agent is `rag_document_agent()` in `agents.py`. Permit integration in `tools_documents.py` is unfinished. The permission checks reference an uninitialized `permit` client and document search references `Q` without importing it.

## Remote vLLM deployment

- `10.140.173.65` and `192.168.10.100` are addresses for the same remote server. The local Django `.env` was configured with `10.140.173.65` for both services. Use the interface reachable from the Django host.
- `VLLM_BASE_URL` and `VLLM_MODEL` select chat on port `8123` with `RedHatAI/Qwen3.8-27B-INT4`. `VLLM_EMBEDDINGS_BASE_URL` and `VLLM_EMBEDDINGS_MODEL` select embeddings on port `8124` with `Qwen/Qwen3-Embedding-0.6B`. Both base URLs end in `/v1`.
- `docker-compose.yaml` defines the vLLM stack for the remote GPU server. Both services run together. Embeddings starts after chat is healthy. The GPU memory defaults are `0.82` for chat and `0.12` for embeddings. The configured context limits are 16,384 and 1,024 tokens, respectively.
- `VLLMEmbeddings` sends text batches directly to `/v1/embeddings` without an authorization header. Chat still uses `ChatOpenAI`. The supervisor construction smoke test used a dummy `OPENAI_API_KEY` for SDK initialization and made no model request.
- The owner plans to remove the old Gemini Chroma stores. The Qwen change is for future data. Existing vectors were neither converted nor re-indexed. Leave deletion and backfills to an explicit request.

## Verification

```bash
uv lock --check
uv sync --locked
uv run --locked python agenticRag/manage.py check
uv run --locked python agenticRag/manage.py test agenticRag
```

A fresh locked install, Django system checks, PDF indexer import, and supervisor compilation passed. Django discovered zero automated tests. A local mock passed the embedding request check under uv. Both remote addresses were unreachable from the development workspace during the last probe, so verify live model calls from a host with access to the server.
