# 🎓 EduAgent — AI Personal Tutoring Assistant

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-Agent%20Framework-1C3C3C" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-RAG-DA5B0B" alt="ChromaDB RAG"></a>
  <a href="https://www.sqlite.org/"><img src="https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License"></a>
</p>

EduAgent is a production-oriented AI tutoring system built on **LangGraph**, **FastAPI**, **Streamlit**, **SQLite**, and **ChromaDB**. It runs a multi-agent reasoning graph that plans each tutoring turn, teaches topics, generates practice quizzes, grades submissions, extracts long-term memory, and recommends next steps — all grounded in the student's own uploaded notes via retrieval-augmented generation (RAG).

**📖 Documentation:** [Project Summary](docs/PROJECT_SUMMARY.md) · [Architecture & Diagrams](docs/architecture.md) · [RAG Experiments & Findings](experiments/findings.md)

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│  Streamlit │───▶│  FastAPI   │───▶│  LangGraph │
│  frontend  │    │  /api/v1   │    │  workflow  │
└────────────┘    └────────────┘    └─────┬──────┘
                                          │
              ┌────────────┬──────────────┼───────────────┐
              ▼            ▼              ▼               ▼
          ┌─────────┐  ┌────────┐   ┌────────────┐   ┌─────────┐
          │ Planner │  │Teacher │   │ Quiz/Evalu │   │ Memory  │
          │ RAG     │  │        │   │ Recommend  │   │         │
          └─────────┘  └────────┘   └────────────┘   └─────────┘
              │            │              │               │
              ▼            ▼              ▼               ▼
          ┌─────────┐  ┌────────┐   ┌────────────┐   ┌─────────┐
          │ChromaDB │  │ SQLite │   │  LLM       │   │ .env    │
          │(vectors)│  │ (ORM)  │   │ openai/    │   │ config  │
          └─────────┘  └────────┘   │ gemini/    │   └─────────┘
                                    │ demo       │
                                    └────────────┘
```

---

## ✨ Features

| Area | Capability |
|---|---|
| **Agentic tutoring** | LangGraph workflow with 7 nodes: `planner → (teach / quiz / answer) → memory → recommendation` |
| **Lesson generation** | Structured explanations with examples, analogies, recap — tuned to the student's level & learning style |
| **Adaptive quizzes** | Fixed-composition quizzes (5 MCQ + 2 numerical + 1 subjective) with auto-adapting difficulty |
| **Deterministic grading** | Objective questions scored in code; subjective answers graded by the LLM with graceful fallback |
| **Long-term memory** | Facts extracted after every turn (strengths, preferences, goals, misconceptions) and reused in later prompts |
| **Recommendations** | Data-driven next-step suggestions from concept scores, weak areas and quiz history |
| **RAG note-grounding** | Upload PDF/TXT/MD notes → chunked & embedded in ChromaDB → retrieved to ground every answer with page citations |
| **Personal dashboard** | Score trends, weak concepts, topics learned, memory and recommendations in one Streamlit view |
| **Offline demo mode** | Run the whole system without any API key via a deterministic demo LLM provider |
| **API-first** | Full REST API under `/api/v1` with typed Pydantic schemas and consistent error responses |

---

## 🧱 Architecture

Full diagrams (system context, both LangGraph workflows, the RAG pipeline, the
SQLite data model, and the chat sequence) live in
**[`docs/architecture.md`](docs/architecture.md)**. A compact layout:

```
D:\EduAgent
├── api/                  # FastAPI backend
│   ├── main.py           #   app factory, lifespan, CORS
│   ├── dependencies.py   #   DI: container & agent context
│   ├── exception_handlers.py
│   └── routers/          #   health, chat, students, quizzes, documents, dashboard
├── agents/               # LangGraph nodes (one module per agent)
│   ├── planner.py        #   intent classification + graph routing
│   ├── teacher.py        #   lesson generation + persistence
│   ├── quiz.py           #   quiz generation + persistence
│   ├── evaluation.py     #   grading + student-model update
│   ├── memory.py         #   fact extraction + transcript log
│   ├── recommendation.py #   next-step suggestions
│   ├── rag.py            #   retrieval + context assembly
│   ├── context.py        #   AgentContext dataclass (DI into nodes)
│   └── base.py           #   shared prompt-section helpers
├── graphs/
│   └── workflow.py       # build_tutoring_graph / build_evaluation_graph
├── services/
│   ├── llm_client.py     #   OpenAI / Gemini / demo providers (lazy init)
│   ├── llm_demo.py       #   deterministic offline provider
│   ├── container.py      #   process-wide dependency wiring
│   └── rag/              #   chunking, embeddings, vector store, ingestion
├── database/             # SQLAlchemy ORM + repositories
│   └── repositories/     #   student, quiz, conversation, memory, lesson, doc, progress
├── models/               # Pydantic schemas, enums, LangGraph state
├── prompts/              # YAML prompt templates (string.Template)
├── config/               # typed settings (pydantic-settings) + constants
├── utils/                # logging, errors, helpers, async_retry
├── frontend/             # Streamlit dashboard + httpx API client
├── tests/                # pytest suite (unit + integration)
├── main.py               # uvicorn entry point
└── requirements*.txt
```

### How a tutoring turn flows

1. **Planner** classifies intent (`learn` / `practice` / `answer_question` / `general`) and picks the topic.
2. **Teacher** (optionally after **RAG**) generates a structured lesson and persists it.
3. **Quiz** agent (when `with_quiz`) generates a fixed-composition quiz; **answer IDs** are deterministic.
4. **Memory** extracts durable facts and logs the transcript.
5. **Recommendation** suggests the next topic from the updated student model.

A separate **evaluation graph** (`evaluation → memory → recommendation`) grades quiz submissions: MCQ/numerical scored in code, the written answer graded by the LLM, and every result persisted to the student model.

---

## 🚀 Quickstart

### 1. Prerequisites

- Python **3.11 – 3.12** (3.13+/3.14 may lack wheels for `torch`/`sentence-transformers`)
- (Optional) API keys for [OpenAI](https://platform.openai.com/) or [Google Gemini](https://aistudio.google.com/)

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests/linting
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER (openai|gemini|demo) and your key.
```

> No key? Set `LLM_PROVIDER=demo` — the system runs fully offline with a deterministic demo tutor.

### 4. Run the backend

```bash
python main.py
# or: uvicorn api.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok","demo_mode":true}`. Interactive docs at `http://localhost:8000/docs`.

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501, enter a student ID, and start chatting.

---

## 🔌 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health + mode probe |
| `POST` | `/api/v1/chat` | One tutoring turn (`ChatRequest` → `ChatResponse`) |
| `POST` | `/api/v1/chat/submit-quiz` | Grade a quiz submission → `EvaluationResult` |
| `GET` | `/api/v1/students/{id}` | Get / lazily create a student profile |
| `PUT` | `/api/v1/students/{id}` | Update name/level/learning style |
| `GET` | `/api/v1/quizzes/{student_id}` | List quiz ids |
| `GET` | `/api/v1/quiz/{quiz_id}` | Full quiz with questions |
| `GET` | `/api/v1/dashboard/{student_id}` | Progress summary for the dashboard |
| `POST` | `/api/v1/documents/{student_id}` | Upload & index a document (multipart) |
| `GET` | `/api/v1/documents/{student_id}` | List uploaded documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete document + embeddings |

Example chat turn:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"student_id":"student-1","message":"Teach me about fractions","with_quiz":true}'
```

---

## 🧪 Testing

```bash
pytest -q                # run the full suite
pytest tests/ -k rag     # run only RAG tests
ruff check .             # lint (config in ruff.toml)
```

The suite covers the LLM abstraction (incl. demo provider), chunking & embeddings, the vector store, repositories, every agent, both graphs, the API, and the RAG ingestion path. Tests run against an in-memory SQLite DB and a temporary ChromaDB so they never touch your data. The `ruff.toml` codifies the lint policy (errors, imports, typing, bugbear, complexity, pylint subset) with three documented project idioms ignored.

---

## ⚙️ Configuration

All configuration flows through [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) from environment variables / `.env`. Key groups (see `.env.example`):

| Prefix | Controls |
|---|---|
| `APP_*` | name, version, host, port, CORS origins |
| `LLM_*` | provider, API keys, models, temperature, timeouts |
| `DATABASE_*` | SQLAlchemy URL (SQLite default, any DB works) |
| `VECTOR_*` | ChromaDB dir, collection, embedding model, chunking |
| `API_*` | API prefix, upload dir, pagination |
| `LOGGING_*` | level, JSON format, rotation |
| `FRONTEND_*` | API base URL used by the Streamlit client |

---

## 🗂️ Storage

| Store | Technology | Location | Contents |
|---|---|---|---|
| Relational | SQLite (SQLAlchemy) | `./data/edueagent.db` | students, lessons, quizzes, attempts, conversations, memory, documents, progress |
| Vector | ChromaDB | `./data/chroma/` | embedded note chunks with source/page metadata |
| Uploads | local filesystem | `./data/uploads/` | original uploaded documents |
| Logs | rotating files | `./logs/` | application logs |

---

## 🛡️ Reliability & correctness

- **Deterministic grading** — objective question scores are computed in code; the LLM only grades written answers, with a neutral fallback on failure.
- **Graceful degradation** — any LLM failure leaves the turn intact; RAG retrieval failing returns an empty context instead of crashing.
- **Lazy LLM init** — the app boots without an API key; the key is only needed when a chat is actually attempted.
- **Typed outputs** — every agent returns a validated Pydantic model, and output parsing is fenced-JSON tolerant.
- **Consistent errors** — one `EduAgentError` tree maps to structured HTTP responses.

---

## 📄 License

MIT — see `LICENSE`.

---

*Generated with [Claude Code](https://claude.com/claude-code).*
