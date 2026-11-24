# 🧠 SmartAssist AI — Intelligent Multi-Tool RAG Agent

An agentic document assistant: upload PDF / DOCX / TXT / CSV / XLSX files, then **ask questions, summarise, compare documents,
analyse datasets and run calculations**. A **LangGraph** workflow lets a **Groq LLM** decide which of six real Python tools to
call; answers are grounded in **ChromaDB** retrieval and carry **source citations** (file + page).

> Stack: Python · LangChain · LangGraph · Groq · Hugging Face embeddings (`all-MiniLM-L6-v2`) · ChromaDB · PostgreSQL + SQLAlchemy · FastAPI · Streamlit · Pydantic · Pytest

---

## 1. Problem statement
People keep knowledge in many files. Searching them by hand, comparing reports, or computing figures from spreadsheets is slow and
error-prone, and plain chatbots hallucinate because they do not see your files. SmartAssist AI answers **from your documents**,
shows **where** each answer came from, uses **exact tools** for numbers, and says so when the information **is not** in the documents.

## 2. Objectives
1. Ingest heterogeneous documents into a persistent vector index with metadata (page, sheet, chunk).
2. Answer questions with retrieval-augmented generation and verifiable source references.
3. Let the LLM choose among tools (native tool calling) inside a bounded LangGraph workflow.
4. Persist documents, conversations, tool executions and query history in PostgreSQL.
5. Provide a professional Streamlit UI, a REST API, tests, evaluation and docs.

## 3. Features
| Area | Details |
|---|---|
| Documents | multi-upload, type/size/content validation, SHA-256 duplicate detection, status tracking, delete, reprocess, search |
| RAG | clean → chunk (configurable size/overlap) → embed → Chroma; cosine relevance scores; metadata filters; page tracking; relevance threshold |
| Agent | query rewrite (follow-ups) → intent classification → LLM tool selection → RAG / tools → grounded answer; step limits; error recovery |
| Tools | `calculator`, `document_search`, `csv_excel_analyzer`, `document_summarizer`, `document_comparator`, `document_manager` |
| UI | Dashboard, AI Assistant, Document Library, RAG Explorer, Tool Center, Analytics, Settings (all backed by real data) |
| Ops | env-based config, rotating logs, friendly errors, path-traversal protection, no `eval`/`exec`, optional API key |

## 4. System architecture
```mermaid
flowchart LR
  U[User] --> UI[Streamlit frontend]
  U -.-> API[FastAPI backend]
  UI --> SVC[services.py]
  API --> SVC
  SVC --> AG[LangGraph agent]
  SVC --> ING[Ingestion pipeline]
  AG --> LLM[(Groq LLM)]
  AG --> TOOLS[6 LangChain tools]
  TOOLS --> RET[Retriever]
  TOOLS --> PD[(pandas / files)]
  ING --> EMB[HF embeddings]
  ING --> CH[(ChromaDB)]
  RET --> CH
  SVC --> PG[(PostgreSQL)]
  TOOLS --> PG
  ING --> FS[(data/uploads)]
```
The Streamlit app calls the service layer in-process (one command to run); the FastAPI app exposes the **same** service layer for other clients.

## 5. RAG architecture
```mermaid
flowchart TD
  A[Upload] --> B[Validate: ext, size, magic bytes, SHA-256]
  B --> C[Store under random name in data/uploads]
  C --> D[Extract: pypdf pages / docx2txt / text / pandas rows]
  D --> E[Clean text]
  E --> F[RecursiveCharacterTextSplitter]
  F --> G[all-MiniLM-L6-v2 embeddings]
  G --> H[(ChromaDB: vectors + doc_id, filename, page, sheet, chunk_index)]
  C --> M[(PostgreSQL: document metadata + status)]
  Q[Question] --> R[Embed query] --> S[Cosine top-K + metadata filter] --> T[Relevance threshold]
  H --> S
  T --> X[Numbered context] --> L[LLM answer with citations]
  T -->|nothing relevant| N["'Not found in your documents' (no LLM guess)"]
```
Spreadsheets are indexed as text (first `MAX_INDEXED_ROWS`=5000 rows per sheet) for search, while the analysis tool always reads the **full** file with pandas.

## 6. LangGraph workflow
Generated from the compiled graph (`docs/langgraph.mmd`):
```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	query_understanding(query_understanding)
	intent_classification(intent_classification)
	agent_decision(agent_decision)
	rag_retrieval(rag_retrieval)
	tool_execution(tool_execution)
	response_generation(response_generation)
	final_response(final_response)
	__end__([<p>__end__</p>]):::last
	__start__ --> query_understanding;
	agent_decision -. error .-> final_response;
	agent_decision -. rag .-> rag_retrieval;
	agent_decision -. respond .-> response_generation;
	agent_decision -. tools .-> tool_execution;
	intent_classification -. agent .-> agent_decision;
	intent_classification -. error .-> final_response;
	intent_classification -. general .-> response_generation;
	query_understanding -. error .-> final_response;
	query_understanding -. ok .-> intent_classification;
	rag_retrieval -. agent .-> agent_decision;
	rag_retrieval -. respond .-> response_generation;
	response_generation --> final_response;
	tool_execution -. agent .-> agent_decision;
	tool_execution -. respond .-> response_generation;
	final_response --> __end__;

```
| Node | Role |
|---|---|
| `query_understanding` | validates input; rewrites follow-ups into self-contained queries using history |
| `intent_classification` | LLM labels intent: document_qa, summarization, comparison, data_analysis, calculation, document_management, general_chat |
| `agent_decision` | LLM (tools bound with `bind_tools`) chooses tool calls or `READY` |
| `rag_retrieval` | runs `document_search` calls; also a deterministic fallback so document questions are always grounded |
| `tool_execution` | runs any tool calls, logs each to PostgreSQL |
| `response_generation` | writes the final answer from context + tool results (skipped when nothing relevant was found) |
| `final_response` | sets status, action, and step-limit notes |

Conditional routing: `general_chat` skips tools; tool calls loop back to the agent for **multi-step** reasoning; the loop is capped by
`MAX_AGENT_STEPS` (default 4) plus a LangGraph `recursion_limit`; identical repeated calls are de-duplicated; unknown tools / invalid
arguments become error messages the agent can recover from.

## 7. Tool-calling architecture
Each tool is a LangChain `@tool` with a Pydantic `args_schema`; the LLM receives the JSON schemas through native tool calling.
Tools never run user code: the calculator uses an AST whitelist, the dataset tool exposes predefined pandas operations, and all arguments are validated.
Tools return JSON (`status: ok|error`) so the agent can react to errors. `document_search` uses `response_format="content_and_artifact"` to hand retrieved chunks (with scores) to the graph state.

## 8. Database architecture
```mermaid
erDiagram
  DOCUMENTS { int id PK  string filename  string stored_name  string file_hash UK  string file_type  bigint size_bytes  string status  text error_message  int page_count  int chunk_count  bigint char_count  json extra_metadata  datetime created_at  datetime processed_at }
  CONVERSATIONS { int id PK  string title  datetime created_at  datetime updated_at }
  MESSAGES { int id PK  int conversation_id FK  string role  text content  json sources  json tool_calls  string intent  datetime created_at }
  TOOL_EXECUTIONS { int id PK  int conversation_id FK  string tool_name  json input_args  text output  string status  text error_message  int duration_ms  string source  datetime created_at }
  QUERY_HISTORY { int id PK  int conversation_id FK  text query  text rewritten_query  string intent  string selected_action  json tools_used  int num_sources  string status  int latency_ms  datetime created_at }
  CONVERSATIONS ||--o{ MESSAGES : has
  CONVERSATIONS ||--o{ TOOL_EXECUTIONS : logs
  CONVERSATIONS ||--o{ QUERY_HISTORY : records
```
Vectors live in ChromaDB (`data/chroma_db`); chunk metadata links back to `documents.id`. There is **no login system** in this version, so no `users` table (see Future enhancements).

---

## 9. Installation (Windows, PowerShell)

**Prerequisites:** Python 3.11+ ([python.org](https://www.python.org/downloads/), tick *Add to PATH*), PostgreSQL 14+ ([postgresql.org](https://www.postgresql.org/download/windows/) – remember the `postgres` password), a free Groq API key ([console.groq.com/keys](https://console.groq.com/keys)), internet access for the first run (downloads the ~90 MB embedding model).

```powershell
# 1. open the project folder
cd SmartAssist-AI

# 2. virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass

# 3. dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt       # add -r requirements-dev.txt instead if you want to run tests

# 4. configuration
copy .env.example .env
notepad .env                          # set GROQ_API_KEY and POSTGRES_PASSWORD
```

## 10. Environment configuration
All settings are in `.env` (see `.env.example`). Important ones:

| Variable | Meaning |
|---|---|
| `GROQ_API_KEY` | your Groq key (never shown in the UI, never sent to the browser) |
| `GROQ_MODEL` | any Groq model with tool calling; change here if Groq retires a model |
| `POSTGRES_*` or `DATABASE_URL` | database connection (URL-encode special characters in URLs, `@` → `%40`; `POSTGRES_*` handles them automatically) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_RELEVANCE_SCORE` | RAG tuning |
| `MAX_UPLOAD_MB`, `MAX_AGENT_STEPS`, `API_AUTH_TOKEN` | safety limits |

## 11. Run
```powershell
python run.py setup     # creates the PostgreSQL database if missing, tables, folders, ChromaDB, downloads embeddings
python run.py all       # Streamlit  http://localhost:8501   +   API docs  http://localhost:8000/docs
```
Other commands: `python run.py ui` (frontend only) · `python run.py api` · `python run.py check` (diagnostics) · `python run.py test` · `python run.py eval`.
Manual alternative: `uvicorn backend.main:app --reload` and `streamlit run frontend/app.py`.

*No PostgreSQL yet?* For a quick local demo only, set `DATABASE_URL=sqlite:///data/smartassist.db`. PostgreSQL is the supported production database.

## 12. Testing
```powershell
pip install -r requirements-dev.txt
python run.py test                                   # 112 tests (SQLite, offline)
$env:TEST_DATABASE_URL="postgresql+psycopg2://postgres:PASSWORD@localhost:5432/smartassist_test"; python run.py test   # whole suite on PostgreSQL
$env:RUN_PG_TESTS="1"; $env:POSTGRES_PASSWORD="PASSWORD"; pytest tests/test_postgres.py   # DB auto-creation tests
python run.py eval                                   # retrieval/tool metrics (offline) ; add --live for routing + groundedness (needs GROQ key)
```
| Suite | Covers |
|---|---|
| `test_rag.py` | loading (all 5 formats, PDF pages), cleaning, chunk size/overlap, embeddings, Chroma store/search/delete/**persistence**, metadata filters, source references, missing-information handling, validation, duplicates, path traversal, failed processing |
| `test_tools.py` | every tool incl. GST maths, unsafe-expression rejection, pandas cross-checks, Excel sheets, map-reduce summarisation, comparison metrics |
| `test_agents.py` | intent parsing, routing functions, full graph runs: RAG with citations, fallback retrieval, multi-tool, parallel calls, general chat, no-hallucination guard, step limit, de-duplication, bad tool calls, LLM failure, rewrite of follow-ups, persistence |
| `test_database.py` / `test_postgres.py` | connection, CRUD, constraints, messages, tool logs, stats, cascade, rollback, DB auto-creation |
| `test_api.py` / `test_frontend.py` | all REST endpoints + API-key guard; every Streamlit page and a chat round-trip via Streamlit `AppTest` |

**What the tests do and do not prove.** The agent tests run the *real* LangGraph, tools, RAG, SQL and API code, but the chat model is a scripted
stand-in implementing LangChain's `BaseChatModel` (so decisions are deterministic), and embeddings use an offline hash embedder. They verify the
**plumbing and safety behaviour**, not Groq's answer quality or MiniLM retrieval quality. Use `python run.py eval --live` on your machine for those.

## 13. Screenshots
Take these after your first run and save them in `docs/screenshots/`:
`dashboard.png`, `assistant.png`, `library.png`, `rag_explorer.png`, `tool_center.png`, `analytics.png`, `settings.png`.

## 14. Project structure
```
backend/  agents/ (graph, nodes, routing, state) · chains/ (prompts, response chain) · rag/ (loader, chunking, embeddings, vector store, retriever, ingestion)
          tools/ (6 tools + common) · llm/model.py · database/ (db, models, repository) · api/routes.py · config/settings.py · services.py · main.py
frontend/ app.py · components/ui.py · pages/ (dashboard, assistant, documents, rag_explorer, tools, analytics, settings)
tests/ · evaluation/ · sample_data/ · scripts/ · docs/ · run.py · requirements*.txt · .env.example
```
Deviations from the suggested layout: added `services.py` (shared service layer), `schemas.py`, `exceptions.py`; PDF/DOCX are read with `pypdf` / `docx2txt` directly
(producing LangChain `Document`s) because `langchain-community` loaders are being sunset.

## 15. Security notes
Secrets only via environment · uploads validated by extension, size **and** content signature · stored under random names, resolved inside the upload folder ·
filename sanitising · no `eval`/`exec`/shell · calculator AST whitelist · pandas operations are predefined · internal errors never shown to users (logged to `logs/smartassist.log`) ·
DB password masked in UI · optional `API_AUTH_TOKEN`. Uploaded text is sent to Groq when you ask questions about it – do not upload data you may not share with a third-party LLM provider.

## 16. Future enhancements
User accounts & per-user document spaces (`users` table) · OCR for scanned PDFs · streaming answers · hybrid (BM25 + vector) search and re-ranking ·
Alembic migrations · background ingestion queue for large files · charts from dataset questions · Docker Compose · RAGAS-style automatic evaluation.

## 17. Troubleshooting
| Symptom | Fix |
|---|---|
| "database is not reachable" | start PostgreSQL service, check `POSTGRES_*`, run `python run.py setup` |
| "LLM is not configured" | set `GROQ_API_KEY` in `.env`, then Settings → *Reload settings* |
| "model not available" | Groq retired the model: set a current tool-calling model in `GROQ_MODEL` |
| embedding model cannot download | check internet/proxy; first run only |
| PDF "No text could be extracted" | scanned image PDF – OCR not supported yet |
