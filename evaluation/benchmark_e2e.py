"""M8 — End-to-End Workflow Time.

Definition
----------
Wall-clock time to complete a *full learning session* through the live HTTP API:
lesson → practice quiz → submission & evaluation → dashboard refresh. This is
the metric a real student would perceive as "how long does one study cycle
take", and it includes network, FastAPI overhead, every LangGraph node, LLM
calls, persistence, embedding uploads and serialization.

Method
------
Each test *case* is a complete workflow driven over HTTP with a fresh student:

    step 1  POST /api/v1/chat                        teach turn
    step 2  POST /api/v1/chat (with_quiz=True)       lesson + quiz generation
    step 3  POST /api/v1/chat/submit-quiz            evaluation + memory update
    step 4  GET  /api/v1/dashboard/{student}         recall + recommendations
    step 5  POST /api/v1/documents/{student}         document upload + embed (RAG)

Every step is timed individually; the per-case total is the sum of its steps.

Calculation
-----------
    e2e_workflow_time = mean per-case total wall-clock (seconds)
    per-step stats    = {mean, p50, p90, min, max} for each step across cases
"""

from __future__ import annotations

from pathlib import Path

import httpx

from common import (
    BASE_URL,
    DEFAULT_RUNS,
    STUDENT_PREFIX,
    base_fig,
    ensure_server,
    latency_stats,
    save_csv,
    timed_call,
    write_chart,
)

METRIC_ID = "m8"
METRIC_NAME = "End-to-End Workflow Time"
UNIT = "seconds"

_HERE = Path(__file__).resolve().parent
# Prefer the evaluation-local copy; fall back to the app's data/ dir.
SAMPLE_DOC = (
    _HERE / "data" / "sample_photosynthesis.pdf"
    if (_HERE / "data" / "sample_photosynthesis.pdf").exists()
    else _HERE.parent / "data" / "sample_photosynthesis.pdf"
)

STEPS = ["lesson", "quiz", "submit", "dashboard", "upload"]


def _run_workflow(client: httpx.Client, student: str,
                  sample_bytes: bytes) -> tuple[dict[str, float], float]:
    """One full study-cycle; returns (per-step seconds, total seconds)."""
    step_times: dict[str, float] = {}

    # 1) Teach turn
    _, lesson_t = timed_call(
        client.post, "/api/v1/chat",
        json={"student_id": student, "message": "Teach me about the water cycle"},
    )
    step_times["lesson"] = lesson_t

    # 2) Lesson + quiz generation
    quiz_resp, quiz_t = timed_call(
        client.post, "/api/v1/chat",
        json={"student_id": student,
              "message": "Give me a practice quiz on the periodic table",
              "with_quiz": True},
    )
    step_times["quiz"] = quiz_t

    # 3) Submit quiz answers
    quiz = quiz_resp.json().get("quiz")
    if quiz:
        answers = [
            {"question_id": q["id"], "answer": q["correct_answer"]}
            for q in quiz.get("questions", [])
        ]
        _, submit_t = timed_call(
            client.post, "/api/v1/chat/submit-quiz",
            json={"quiz_id": quiz["id"], "student_id": student, "answers": answers},
        )
    else:
        submit_t = 0.0
    step_times["submit"] = submit_t

    # 4) Dashboard refresh
    _, dash_t = timed_call(client.get, f"/api/v1/dashboard/{student}")
    step_times["dashboard"] = dash_t

    # 5) Document upload + embed
    _, upload_t = timed_call(
        client.post, f"/api/v1/documents/{student}",
        files={"file": ("sample_photosynthesis.pdf", sample_bytes, "application/pdf")},
    )
    step_times["upload"] = upload_t

    return step_times, sum(step_times.values())


def run() -> dict:
    ensure_server()
    runs = DEFAULT_RUNS
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        try:
            with open(SAMPLE_DOC, "rb") as fh:
                sample_bytes = fh.read()
        except FileNotFoundError:
            sample_bytes = b""

        per_step: dict[str, list[float]] = {s: [] for s in STEPS}
        totals: list[float] = []
        for i in range(runs):
            student = f"{STUDENT_PREFIX}_e2e_{i}"
            step_times, total = _run_workflow(client, student, sample_bytes)
            totals.append(total)
            for step in STEPS:
                per_step[step].append(step_times.get(step, 0.0))

    rows: list[dict] = []
    for step in STEPS:
        s = latency_stats(per_step[step])
        rows.append({
            "step": step,
            "n": s["n"],
            "mean_s": round(s["mean"], 4),
            "median_s": round(s["median"], 4),
            "p50_s": round(s["p50"], 4),
            "p90_s": round(s["p90"], 4),
            "min_s": round(s["min"], 4),
            "max_s": round(s["max"], 4),
        })
    t = latency_stats(totals)
    rows.append({"step": "TOTAL", "n": t["n"], "mean_s": round(t["mean"], 4),
                 "median_s": round(t["median"], 4), "p50_s": round(t["p50"], 4),
                 "p90_s": round(t["p90"], 4), "min_s": round(t["min"], 4),
                 "max_s": round(t["max"], 4)})

    csv_path = save_csv(rows, "e2e_workflow")

    # Chart: stacked bars per case (each step's contribution) + mean total line.
    fig = base_fig("End-to-End Workflow Time (full study-cycle)", "seconds")
    case_steps = [[per_step[s][i] for s in STEPS] for i in range(runs)]
    colors = {"lesson": "#4C78A8", "quiz": "#F58518", "submit": "#54A24B",
              "dashboard": "#B279A2", "upload": "#E45756"}
    for j, step in enumerate(STEPS):
        fig.add_trace(
            dict(type="bar", name=step,
                 x=[f"Case {i+1}" for i in range(runs)],
                 y=[case_steps[i][j] for i in range(runs)],
                 marker_color=colors[step]),
        )
    fig.add_hline(y=t["mean"], line_dash="dot", line_color="#333333",
                  annotation_text=f"mean total = {t['mean']:.2f}s",
                  annotation_position="top right")
    fig.update_layout(barmode="stack", yaxis=dict(rangemode="tozero"))
    graph_path = write_chart(fig, "e2e_workflow")

    return {
        "metric_id": METRIC_ID,
        "metric_name": METRIC_NAME,
        "value": round(t["mean"], 4),
        "unit": UNIT,
        "detail": f"mean full study-cycle = {t['mean']:.2f}s across {runs} cases "
                  f"(lesson+quiz+submit+dashboard+upload)",
        "csv": str(csv_path),
        "graph": str(graph_path),
        "num_cases": runs,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
