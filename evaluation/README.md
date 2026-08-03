# EduAgent Evaluation Framework

A benchmark suite for the EduAgent tutoring system. It measures eight quality
and performance metrics, writes per-metric CSV reports and plotly HTML charts,
and aggregates everything into a summary report and dashboard.

The framework is **read-only tooling**: it never edits application code. It
talks to the running HTTP API and, for three metrics, drives the already
compiled LangGraph through a throw-away container.

## Quick start

```bash
# 1) Create the ground-truth corpus used by the RAG metrics (one-time)
python evaluation/make_sample_docs.py

# 2) Start the server in one terminal
python main.py

# 3) Run the full suite in another
python evaluation/run_all.py

# Or run specific metrics
python evaluation/run_all.py --only m1 m3 m5 m8
```

> The hermetic benchmarks (M3 retrieval, M4 hallucination, M7 agent timing)
> need **no** running server — they build an isolated container with their own
> temporary SQLite DB and ChromaDB directory, so they are safe to run anywhere.

## Outputs

Everything lands under `evaluation/results/`:

| Path                        | Contents                                        |
|-----------------------------|-------------------------------------------------|
| `csv/summary_metrics.csv`   | One row per metric (id, name, value, unit, detail) |
| `summary.json`              | Full structured report (all per-metric dicts)   |
| `graphs/index.html`         | Dashboard linking every chart                   |
| `csv/<metric>.csv`          | Per-case rows for each benchmark                |
| `graphs/<metric>.html`      | Interactive plot for each benchmark             |

Tunables (environment variables):

| Variable        | Default | Meaning                              |
|-----------------|---------|--------------------------------------|
| `EVAL_RUNS`     | `5`     | Repetitions for latency / agent / e2e timing |
| `EDUAGENT_BASE_URL` | `http://localhost:8000` | Server address for HTTP benchmarks |

## Metric reference

Every metric below documents **what it means**, **how it is calculated**, and
**where the benchmark lives**.

### M1 — Response Accuracy  (`benchmark_accuracy.py`)

*What.* How often the tutoring system produces a complete, on-topic, well
structured lesson rather than an error, an empty answer, or a stub.

*How.* Eight topics (photosynthesis, Newton's laws, water cycle, periodic
table, machine learning, fractions, human heart, American Revolution) are each
sent as a teach turn over HTTP. Every response is scored against 6 binary
checks:

1. HTTP request succeeded (`http_ok`)
2. A lesson/explanation object was returned (`lesson_present`)
3. The explanation is substantive — at least 40 characters (`substantive`)
4. The explanation is on-topic — a topic keyword appears in the text (`topic_relevant`)
5. It includes an example or analogy (`examples_analogies`)
6. It ends with a recap or summary line (`recap`)

Per-topic score = `passed / 6`. **Headline** = mean score across all 8 topics.

### M2 — Quiz Generation Quality  (`benchmark_quiz.py`)

*What.* How faithfully generated quizzes match the required spec: composition,
structure, answer validity, and topic correctness.

*How.* Six topics are each asked for a quiz (`with_quiz=True`). Each quiz is
scored against 13 binary checks, covering:

- **Structure** — exactly 8 questions total; 5 MCQ, 2 numerical, 1 subjective
- **Content** — every question has text; MCQs have 4 unique options; the
  correct answer appears among the options; numerical questions have numeric
  answers; the subjective question has a model answer; topic is set
- **Quality** — question IDs are unique; difficulty values are valid

Per-quiz score = `passed / 13`. **Headline** = mean across all 6 topics.

### M3 — Retrieval Accuracy (RAG)  (`benchmark_rag.py`, hermetic)

*What.* How well the RAG pipeline retrieves the right source chunks for a
question. Runs against a controlled two-topic corpus
(`data/notes_photosynthesis.txt`, `data/notes_newton.txt`).

*How.* Seven queries — 3 photosynthesis, 3 Newton, 1 unrelated negative
control — are run through the tutoring graph hermetically. The `rag` node's
retrieved chunks are compared to the query's expected source document:

- **hit@1** — is the best-ranked chunk from the expected source?
- **precision@4** — fraction of the top-4 chunks from the expected source
- **recall@4** — fraction of expected-source chunks retrieved in top-4
- **MRR** — reciprocal rank of the first expected-source chunk

Per-query scores are aggregated over the 6 relevant queries (the negative
control verifies it *does not* match). **Headline** = mean per-query score of
the four metrics.

### M4 — Hallucination Rate  (`benchmark_hallucination.py`, hermetic)

*What.* The fraction of *specific factual claims* in a tutor response that are
NOT supported by the retrieved notes. This is the RAG grounding check — the
part of hallucination a retrieval-augmented system is meant to eliminate.

*How.* Four answer-question turns are routed through the `rag` node against the
photosynthesis corpus. From each explanation we extract *specific claims*:
numeric quantities (`430 nm`, `9.8`), chemical formulas (`CO2`, `H2O`,
`C6H12O6`, `ATP`, `NADPH`), and mid-sentence proper nouns. We then count how
many of those appear verbatim in any retrieved chunk:

```
hallucination_rate(turn) = (specific claims NOT in retrieved chunks) / (specific claims)
```

We also report **grounding coverage** — the fraction of response sentences
whose tokens overlap a retrieved chunk (token Jaccard ≥ 0.25). Low coverage
means the model wrote generic prose instead of quoting the notes; that is *not*
counted as hallucination but is reported separately.

> With the deterministic demo provider the system emits template prose and no
> fabricated specifics, so the headline rate is ~0. The same harness measures
> real hallucination when a live LLM provider is configured.

### M5 — Response Latency  (`benchmark_latency.py`, HTTP)

*What.* Wall-clock round-trip time of each tutoring endpoint, measured on the
client side — includes network, FastAPI overhead, the LangGraph run, LLM call,
persistence and serialization.

*How.* Six endpoint classes (`chat_learn`, `chat_quiz`, `chat_answer`,
`submit_quiz`, `upload_doc`, `dashboard`) are each repeated `EVAL_RUNS` times.
Per endpoint we report p50 / p90 / p95 / mean / median / min / max.
**Headline** = p95 across *all* samples (the worst-class percentile), so it
represents the slow end of the real user experience.

### M6 — Memory Recall Accuracy  (`benchmark_memory.py`, HTTP)

*What.* How faithfully what the student says is stored into long-term memory
and then re-surfaced by the dashboard.

*How.* Five conversations, each with a fresh student. After one tutoring turn
we read `recent_memory_facts` from the dashboard and compare the message's
keywords with the keywords stored in the facts:

```
recall    = |message keywords ∩ fact keywords| / |message keywords|
precision = |message keywords ∩ fact keywords| / |fact keywords|
```

- **recall** — did the student's content get stored?
- **precision** — is everything stored relevant to that message?

**Headline** = mean recall across conversations; precision is reported
alongside it for context.

### M7 — Agent Execution Time  (`benchmark_agents.py`, hermetic)

*What.* Wall-clock execution time of each LangGraph node (planner, rag,
teacher, quiz, evaluation, memory, recommendation) and of a full tutoring turn.
This isolates the *orchestration* cost of the multi-agent workflow.

*How.* A hermetic container runs the *same compiled graphs* the API uses. For
each turn we iterate `graph.astream(state, stream_mode="updates")`; LangGraph
emits one `{node: update}` per node as it completes, so the wall-clock between
successive emissions approximates each node's execution time. Four turn types
are exercised (`teach`, `quiz`, `answer`, `submit`), each repeated
`EVAL_RUNS` times.

**Headline** = mean full teach-turn graph time. Per-node mean / p50 / p90 / std
are reported for every node.

### M8 — End-to-End Workflow Time  (`benchmark_e2e.py`, HTTP)

*What.* Wall-clock time for a complete learning session a student would
perceive: lesson → practice quiz → submission & evaluation → dashboard refresh
→ document upload.

*How.* Each case is a full study cycle over HTTP with a fresh student, with
every step timed individually:

1. `POST /api/v1/chat` — teach turn
2. `POST /api/v1/chat` (`with_quiz=True`) — lesson + quiz generation
3. `POST /api/v1/chat/submit-quiz` — evaluation + memory update
4. `GET  /api/v1/dashboard/{student}` — recall + recommendations
5. `POST /api/v1/documents/{student}` — document upload + embedding

Per-case total = sum of the five step times. **Headline** = mean total across
`EVAL_RUNS` cases; per-step stats are reported for bottleneck analysis.

## Interpreting the results

Run against the deterministic **demo LLM provider** the suite measures the
*framework* — routing, structure, retrieval, orchestration, latency — with
zero variance. Expected shape:

| Metric | Demo-provider expectation                    |
|--------|----------------------------------------------|
| M1 Accuracy            | ~1.0 (template lessons pass all structural checks) |
| M2 Quiz quality        | ~1.0 (deterministic quizzes meet the spec)    |
| M3 Retrieval           | hit@1 / MRR near 1.0; precision@4 < 1.0 (top-4 has off-topic neighbours) |
| M4 Hallucination       | ~0.0 (no fabricated specifics), low grounding coverage (generic prose) |
| M5 Latency             | sub-second warm; document upload is the slowest class |
| M6 Memory recall       | ~1.0 recall, moderate precision (facts include scaffolding) |
| M7 Agent time          | RAG node dominates; planner is the cheapest   |
| M8 E2E time            | sum of the five steps, dominated by quiz + upload |

To measure *content quality* (real hallucination, real teaching quality) set a
live provider in `.env` and re-run — the harness is provider-agnostic.

## Adding a metric

1. Create `benchmark_<name>.py` exposing `run() -> dict` with keys
   `metric_id`, `metric_name`, `value`, `unit`, `detail`, `csv`, `graph`.
2. Add it to `ALL_METRICS` in `run_all.py`.
3. Reuse `common.py` helpers (`ensure_server`, `timed_call`, `latency_stats`,
   `save_csv`, `write_chart`, `hermetic_container`, `initial_state`).
