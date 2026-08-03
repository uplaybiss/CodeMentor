# EduAgent — Architecture

This document describes the system architecture of EduAgent. All diagrams are
[Mermaid](https://mermaid.js.org/) and render on GitHub; they can also be edited
in any Mermaid-aware editor.

---

## 1. System context

```mermaid
flowchart LR
    subgraph UI["Streamlit frontend (frontend/)"]
        S[app.py — Chat / Quiz / Dashboard / Documents]
        C[api_client.py — httpx client]
    end

    subgraph API["FastAPI backend (api/)"]
        R[/REST /api/v1 routers/]
        D[Container — DI & process-wide singletons]
    end

    subgraph GRAPH["LangGraph workflows (graphs/)"]
        T[build_tutoring_graph]
        E[build_evaluation_graph]
    end

    subgraph AGENTS["Agents (agents/)"]
        P[Planner] --> RA[RAG]
        TCH[Teacher]
        Q[Quiz] --> EV[Evaluation]
        M[Memory] --> REC[Recommendation]
    end

    subgraph DATA["Persistence"]
        DB[(SQLite — SQLAlchemy ORM)]
        VS[(ChromaDB — vector store)]
        LLM[LLM providers: OpenAI / Gemini / demo]
    end

    S --> C
    C <-->|HTTP /api/v1| R
    R --> D
    D --> GRAPH
    T --> AGENTS
    E --> AGENTS
    AGENTS --> DB
    RA --> VS
    AGENTS --> LLM
```

**Layers**

| Layer | Path | Responsibility |
|---|---|---|
| Presentation | `frontend/` | Streamlit UI, session state, httpx API client |
| API | `api/` | REST routers, Pydantic request/response models, DI |
| Orchestration | `graphs/` | Compiled LangGraph state machines |
| Agents | `agents/` | One node per agent; structured LLM I/O via `AgentContext` |
| Domain | `models/` | Pydantic schemas, enums, LangGraph state `TypedDict` |
| Services | `services/` | LLM client, RAG (chunk → embed → retrieve), DI container |
| Persistence | `database/` | SQLAlchemy ORM models, repositories, connection/session mgmt |
| Config | `config/` | pydantic-settings models (env-driven), constants, prompt loader |
| Utils | `utils/` | Logging, error types, dependency-free helpers |

---

## 2. Tutoring workflow (chat turn)

Every `/api/v1/chat` call runs the tutoring graph. The **planner** classifies
the student's intent and emits a deterministic `next_agent` value that drives a
conditional edge — routing stays out of the LLM's hands.

```mermaid
flowchart TD
    START([START]) --> P[planner\nclassify intent → next_agent]

    P -->|next_agent = teach| TCH[teacher\nlesson + explanation]
    P -->|next_agent = answer_question| RA[RAG\nretrieve relevant notes]
    P -->|next_agent = quiz| Q[quiz\ngenerate practice quiz]
    P -->|next_agent = end| END([END])

    RA --> TCH

    TCH -->|with_quiz = true| Q
    TCH -->|else| M[memory\nextract & persist long-term facts]

    Q --> M
    M --> REC[recommendation\nsuggest next topic + plan]
    REC --> END
```

### Intent routing

| Planner intent | Route | Chain |
|---|---|---|
| `learn` | `teach` | teacher → (quiz if `with_quiz`) → memory → recommendation |
| `answer_question` | `answer_question` | rag → teacher → memory → recommendation |
| `practice` | `quiz` | quiz → memory → recommendation |
| unclassified / end | `end` | terminate |

---

## 3. Evaluation workflow (quiz submission)

`/api/v1/chat/submit-quiz` runs a shorter linear graph that grades the answers,
persists the attempt, updates memory and issues the next recommendation.

```mermaid
flowchart LR
    START([START]) --> EV[evaluation\nscore answers, weak concepts, feedback]
    EV --> M[memory\nupdate concept proficiency]
    M --> REC[recommendation\nnext lesson + daily plan]
    REC --> END([END])
```

---

## 4. RAG pipeline

Documents are ingested once; retrieval happens on every `answer_question` turn.

```mermaid
flowchart LR
    subgraph INGEST["Ingestion (api/routers/documents.py → services/rag/ingestion.py)"]
        F[PDF / TXT / MD] --> EX[extract text\nPyMuPDF]
        EX --> CH[chunk\ntiktoken-aware, overlap]
        CH --> EM[embed\nsentence-transformers / ONNX fallback]
        EM --> VS[(ChromaDB)]
    end

    subgraph RETRIEVE["Retrieval (services/rag/service.py)"]
        Q[student question] --> QE[embed query]
        QE --> SR[similarity search top-k]
        SR --> CTX[rag_context + sources\nfed to teacher]
    end
```

Notes:

* Chunking is token-aware when `tiktoken` is available, with a word-based
  fallback.
* Embeddings default to local `sentence-transformers` (no API key); a
  deterministic trigram-hash embedder exists for tests.
* Re-ingesting a `doc_id` replaces its prior chunks; deleting a document also
  clears its vectors.

---

## 5. Data model (SQLite)

All tables reference `students.id`; list-valued columns use SQLite `JSON`.

```mermaid
erDiagram
    students ||--o{ topics : "topics_learned"
    students ||--o{ concept_scores : "proficiency"
    students ||--o{ lessons : "taught"
    students ||--o{ quizzes : "generated"
    students ||--o{ quiz_attempts : "submitted"
    students ||--o{ conversations : "transcript"
    students ||--o{ memory_entries : "long-term facts"
    students ||--o{ documents : "uploaded"
    students ||--o{ recommendations : "latest"
    students ||--o{ progress_snapshots : "learning curve"

    quizzes ||--o{ quiz_questions : "contains"

    students {
        string id PK
        string name
        string level
        string learning_style
        json topics_learned
        json weak_concepts
        json quiz_scores
    }
    quizzes {
        string id PK
        string student_id FK
        string topic
        string difficulty
    }
    quiz_questions {
        int id PK
        string quiz_id FK
        string qtype
        text question
        json options
        text correct_answer
        int points
    }
    quiz_attempts {
        string id PK
        string quiz_id FK
        json answers
        float percent
        json weak_concepts
        text feedback
    }
```

### Repository ↔ table mapping

| Repository | Tables |
|---|---|
| `StudentRepository` | `students`, `topics`, `concept_scores` |
| `QuizRepository` | `quizzes`, `quiz_questions`, `quiz_attempts` |
| `LessonRepository` | `lessons` |
| `MemoryRepository` | `memory_entries`, `recommendations` |
| `ConversationRepository` | `conversations` |
| `DocumentRepository` | `documents` |
| `ProgressRepository` | `progress_snapshots` |

---

## 6. Chat request sequence

```mermaid
sequenceDiagram
    participant UI as Streamlit
    participant API as FastAPI /api/v1
    participant G as Tutoring graph
    participant LLM as LLM provider
    participant DB as SQLite
    participant VS as ChromaDB

    UI->>API: POST /chat {student_id, message, with_quiz}
    API->>G: ainvoke(state)
    G->>LLM: planner classify intent
    LLM-->>G: intent=answer_question
    G->>VS: similarity search(query)
    VS-->>G: top-k chunks
    G->>LLM: teacher answer (grounded in chunks)
    LLM-->>G: lesson
    G->>DB: persist lesson + memory
    G-->>API: final state
    API-->>UI: ChatResponse {messages, lesson, ...}
```

---

## 7. LLM provider selection

```mermaid
flowchart LR
    ENV[LLM_PROVIDER] -->|openai| O[ChatOpenAI]
    ENV -->|gemini| GE[ChatGoogleGenerativeAI]
    ENV -->|demo| DM[DemoProvider\nno API key — offline/CI]
    O --> J[json-mode validation]
    GE --> J
    DM --> J
```

Providers are constructed **lazily** — the app boots with no API keys, and a
missing/invalid provider raises `ProviderConfigurationError` only when a model
is first requested. `demo` makes the whole stack runnable offline and in CI.
