"""M5 — Response Latency.

Definition
----------
Wall-clock time of the full HTTP round-trip for each tutoring endpoint,
measured on the client side (includes network, FastAPI overhead, graph
execution, LLM call, persistence and serialization).

Method
------
Each test *case* is an endpoint / operation class. We repeat every operation
``EVAL_RUNS`` times and report percentiles:

    p50  p90  p95  mean  median  min  max

Cases
-----
* ``chat_learn``   — POST /api/v1/chat, a teach turn
* ``chat_quiz``    — POST /api/v1/chat with ``with_quiz=True`` (lesson + quiz)
* ``chat_answer``  — POST /api/v1/chat, an answer-question turn (RAG path)
* ``submit_quiz``  — POST /api/v1/chat/submit-quiz (evaluation graph)
* ``upload_doc``   — POST /api/v1/documents/{student} (parse + chunk + embed)
* ``dashboard``    — GET  /api/v1/dashboard/{student}

Calculation
-----------
    latency_stats(endpoint) = {p50, p90, p95, mean, ...} over the repeats
    response_latency        = p95 across all samples (worst-class headline)
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

METRIC_ID = "m5"
METRIC_NAME = "Response Latency"
UNIT = "seconds"

_HERE = Path(__file__).resolve().parent
# Prefer the evaluation-local copy; fall back to the app's data/ dir.
SAMPLE_DOC = (
    _HERE / "data" / "sample_photosynthesis.pdf"
    if (_HERE / "data" / "sample_photosynthesis.pdf").exists()
    else _HERE.parent / "data" / "sample_photosynthesis.pdf"
)

OPERATIONS = ["chat_learn", "chat_quiz", "chat_answer", "submit_quiz", "upload_doc", "dashboard"]


def _collect_op(client: httpx.Client, op: str, student: str,
                quiz_id: str | None, answers: list[dict], sample_bytes: bytes,
                runs: int) -> list[float]:
    times: list[float] = []
    for _ in range(runs):
        if op == "chat_learn":
            _, elapsed = timed_call(
                client.post, "/api/v1/chat",
                json={"student_id": student, "message": "Teach me about the water cycle"},
            )
        elif op == "chat_quiz":
            _, elapsed = timed_call(
                client.post, "/api/v1/chat",
                json={"student_id": student,
                      "message": "Give me a practice quiz on the periodic table",
                      "with_quiz": True},
            )
        elif op == "chat_answer":
            _, elapsed = timed_call(
                client.post, "/api/v1/chat",
                json={"student_id": student,
                      "message": "What is the chemical equation for photosynthesis?"},
            )
        elif op == "submit_quiz":
            _, elapsed = timed_call(
                client.post, "/api/v1/chat/submit-quiz",
                json={"quiz_id": quiz_id, "student_id": student, "answers": answers},
            )
        elif op == "upload_doc":
            _, elapsed = timed_call(
                client.post, f"/api/v1/documents/{student}",
                files={"file": ("sample_photosynthesis.pdf", sample_bytes, "application/pdf")},
            )
        elif op == "dashboard":
            _, elapsed = timed_call(client.get, f"/api/v1/dashboard/{student}")
        else:  # pragma: no cover
            continue
        times.append(elapsed)
    return times


def run() -> dict:
    ensure_server()
    student = f"{STUDENT_PREFIX}_lat"
    runs = DEFAULT_RUNS

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        # Prepare quiz id + answers once, reused for every submit iteration.
        quiz_resp = client.post(
            "/api/v1/chat",
            json={"student_id": student,
                  "message": "Give me a practice quiz on fractions",
                  "with_quiz": True},
        )
        quiz = quiz_resp.json().get("quiz")
        quiz_id = quiz["id"] if quiz else None
        answers = (
            [{"question_id": q["id"], "answer": q["correct_answer"]}
             for q in quiz["questions"]]
            if quiz else []
        )
        with open(SAMPLE_DOC, "rb") as fh:
            sample_bytes = fh.read()

        times_by_op: dict[str, list[float]] = {}
        for op in OPERATIONS:
            times_by_op[op] = _collect_op(
                client, op, student, quiz_id, answers, sample_bytes, runs
            )

    rows: list[dict] = []
    for op in OPERATIONS:
        stats = latency_stats(times_by_op[op])
        rows.append({"operation": op, "n": stats["n"],
                     **{k: round(v, 4) for k, v in stats.items()}})

    csv_path = save_csv(rows, "latency")

    fig = base_fig("Response Latency by Endpoint (percentiles)", "seconds")
    for pct, color in (("p50", "#4C78A8"), ("p90", "#F58518"), ("p95", "#E45756")):
        fig.add_trace(
            dict(type="bar", name=pct.upper(),
                 x=[r["operation"] for r in rows],
                 y=[r[pct] for r in rows],
                 marker_color=color, opacity=0.9)
        )
    fig.update_layout(barmode="group", yaxis=dict(rangemode="tozero"))
    graph_path = write_chart(fig, "response_latency")

    overall = latency_stats([t for ts in times_by_op.values() for t in ts])
    worst = max(rows, key=lambda r: r["p95"])

    return {
        "metric_id": METRIC_ID,
        "metric_name": METRIC_NAME,
        "value": round(overall["p95"], 4),
        "unit": UNIT,
        "detail": f"p95 across {len(OPERATIONS)} endpoint classes; "
                  f"worst-class p95 = {worst['p95']:.2f}s ({worst['operation']})",
        "csv": str(csv_path),
        "graph": str(graph_path),
        "num_cases": len(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
