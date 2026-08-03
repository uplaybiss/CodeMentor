# 🎓 EduAgent — Project Summary

*Complete overview of the EduAgent AI tutoring system: what it is, what is
inside it, what has been proven, and how to run it. Written 2026-08-03.*

---

## 1. What it is

EduAgent is a **production-oriented AI personal tutoring system**: you chat, it
teaches you a topic, generates an adaptive quiz, grades you, remembers who you
are across sessions, and recommends what to study next — and it grounds every
answer in **your own uploaded notes** via retrieval-augmented generation (RAG).

Built on **LangGraph + FastAPI + Streamlit + SQLite + ChromaDB**.

**Scale:** 91 Python files · ~9,560 lines · 12 test files.

## 2. The stack

```
Streamlit frontend ──▶ FastAPI /api/v1 ──▶ LangGraph workflow
                                                 │
        ┌───────────────┬───────────────┬───────┼───────────────┬──────────────┐
        ▼               ▼               ▼       ▼               ▼              ▼
     Planner        Teacher       Quiz/Evalu    Memory      Recommendation     RAG
        │               │               │       │               │              │
        ▼               ▼               ▼       ▼               ▼              ▼
    ChromaDB          SQLite          LLM     .env/config      RAG service    .env
    (vectors)         (ORM)     openai/gemini/demo
```

## 3. Agents and graphs

**Two LangGraph workflows, 7 agents** (one module per agent in `agents/`):

- **Tutoring graph:** `planner → (teach / answer_question / quiz / end) →
  rag→teacher → memory → recommendation`
- **Evaluation graph:** `evaluation → memory → recommendation`

| Agent | File | Responsibility |
|---|---|---|
| Planner | `planner.py` | Classify intent (`learn`/`practice`/`answer_question`/`general`) + route the graph |
| Teacher | `teacher.py` | Generate structured lessons (examples, analogies, recap) + persist |
| Quiz | `quiz.py` | Generate fixed-composition quizzes (5 MCQ + 2 numerical + 1 subjective), deterministic answer IDs |
| Evaluation | `evaluation.py` | Grade submissions; objective questions scored in code, subjective by LLM; update student model |
| Memory | `memory.py` | Extract durable facts after every turn + log transcript |
| Recommendation | `recommendation.py` | Suggest next topic from concept scores, weak areas, quiz history |
| RAG | `rag.py` | Retrieve note chunks + assemble grounded context |

## 4. Frontend pages and REST API

**4 Streamlit pages**, all wired to the API:

| Page | Endpoint | Function |
|---|---|---|
| Chat | `POST /api/v1/chat` | Teach / answer, with a "take a quiz" toggle |
| Quiz | `POST /api/v1/chat/submit-quiz` | Grade a submission → score + weak concepts |
| Dashboard | `GET /api/v1/dashboard/{student_id}` | Score trend, weak concepts, topics, memory, recommendations |
| Documents | `POST/GET/DELETE /api/v1/documents/*` | Upload PDF/TXT/MD → chunk → embed → index |

Plus students, quizzes, and health endpoints — **11 endpoints total**, typed
with Pydantic, one consistent error tree.

## 5. What has been proven — evaluation suite (M1–M8)

Eight benchmarks (`evaluation/`) run against the real application, results in
`evaluation/results/` (CSV + interactive chart per metric):

| Metric | Result |
|---|---|
| M1 Response Accuracy | **1.0** (8 topics) |
| M2 Quiz Generation Quality | **1.0** (6 quizzes) |
| M3 Retrieval Accuracy (RAG) | hit@1 **1.00** · recall@4 **1.00** · MRR **1.00** |
| M4 Hallucination Rate | **0.0** (4 grounded turns) |
| M5 Response Latency | p95 **0.256 s** |
| M6 Memory Recall | recall **1.00** (precision 0.41) |
| M7 Agent Execution | full teach-turn **0.085 s** |
| M8 End-to-End Workflow | full study cycle **0.90 s** |

Plus a **12-file pytest suite** (agents, both graphs, RAG, vector store,
chunking, repositories, API, documents) that runs against in-memory SQLite +
temporary ChromaDB — never touches real data.

## 6. RAG configuration experiments (Task C)

A clean-room grid search in `experiments/`: **5 embedding models × 4 chunking
strategies = 20 combinations**, each in an isolated ChromaDB, scored with M3
methodology (hit@1, precision@4, recall@4, MRR) plus latency.

Outputs: `experiments/results/experiment_results.csv`, `summary.json`,
4 interactive charts, and `findings.md`.

**The verdict (findings.md):**

- ✅ **Current defaults are optimal** — `all-MiniLM-L6-v2` + 600/100-token
  chunking scores perfect retrieval. **No change required.**
- ⚡ Want faster? `multi-qa-MiniLM-L6-cos-v1` matches quality at **~35% faster**
  queries (0.026 s vs 0.041 s).
- ❌ `all-mpnet-base-v2` costs 3–5× ingest and ~2× queries for **zero** quality
  gain — avoid.
- 18 of 20 configs hit perfect hit@1; the 2 misses are a property of
  `paraphrase-MiniLM-L6-v2`, not the chunking strategy.

## 7. Live end-to-end demonstration (2026-08-03)

Full pass through the running app:

1. **Chat** — "Teach me about Newton's laws" → planner → teacher lesson.
2. **Quiz** — 8 questions generated → submitted → **72.5%** → weak concept
   flagged → recommendation produced.
3. **Documents (RAG)** — 2 docs uploaded → chunked + embedded → question asked →
   `notes_newton.txt` ranked **#1** → grounded answer.
4. **Dashboard** — student model accumulated: level, 72.5% attempt, 2 documents,
   weak concepts, 9 memory facts, next-step recommendation.

## 8. Reliability & correctness

- **Deterministic grading** — objective scores computed in code; LLM only grades
  written answers with a neutral fallback.
- **Graceful degradation** — any LLM failure leaves the turn intact; RAG failure
  returns empty context instead of crashing.
- **Lazy LLM init** — app boots without an API key; key only needed on first chat.
- **Typed outputs** — every agent returns a validated Pydantic model; fenced-JSON
  tolerant parsing.
- **Consistent errors** — one `EduAgentError` tree → structured HTTP responses.

## 9. How to run

```bash
# install
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt

# configure (copy .env.example → .env, set LLM_PROVIDER=demo|openai|gemini)
cp .env.example .env

# backend
python main.py                # → http://localhost:8000/docs

# frontend
streamlit run frontend/app.py # → http://localhost:8501

# tests / lint
pytest -q
ruff check .
```

No API key is required in `demo` mode — the whole system runs offline with a
deterministic demo tutor.

---

*See [`architecture.md`](architecture.md) for full system diagrams, both
workflows, the RAG pipeline, the SQLite data model, and the chat sequence.*
