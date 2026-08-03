"""M1 — Response Accuracy.

Definition
----------
The fraction of tutoring turns where the returned lesson is correct, relevant
to the asked topic and well-formed. This is the headline quality of the
Teacher/Planner chain: does a student get a substantive, on-topic explanation
for every question?

Method
------
For each test *case* (topic) we POST ``/api/v1/chat`` with ``Explain <topic>``
and score the returned :class:`ChatResponse` against six binary checks:

1. ``http_ok``        — HTTP 200 returned
2. ``lesson_present`` — a ``lesson`` object came back
3. ``substantive``    — explanation text is >= 40 characters
4. ``topic_relevant`` — the topic's keywords appear in the lesson text
5. ``examples``       — at least one example *and* one analogy present
6. ``recap``          — a non-empty recap is included

Calculation
-----------
    accuracy(topic) = (1/6) * sum(check_i)
    response_accuracy = mean over topics of accuracy(topic)

Run
---
Needs the server: ``python main.py`` then ``python evaluation/benchmark_accuracy.py``
"""

from __future__ import annotations

import httpx

from common import (
    BASE_URL,
    DEFAULT_RUNS,
    STUDENT_PREFIX,
    bar_chart,
    ensure_server,
    keywords,
    latency_stats,
    save_csv,
    timed_call,
    write_chart,
)

METRIC_ID = "m1"
METRIC_NAME = "Response Accuracy"
UNIT = "fraction (0-1)"

TOPICS = [
    "Photosynthesis",
    "Newton's laws of motion",
    "The water cycle",
    "The periodic table",
    "Machine learning",
    "Fractions",
    "The human heart",
    "The American Revolution",
]

CHECK_KEYS = ["http_ok", "lesson_present", "substantive", "topic_relevant", "examples", "recap"]


def _score_topic(client: httpx.Client, student: str, topic: str) -> dict:
    payload = {"student_id": student, "message": f"Explain {topic}"}
    resp, elapsed = timed_call(client.post, "/api/v1/chat", json=payload)
    body = resp.json() if resp.content else {}
    lesson = body.get("lesson") or {}

    explanation = lesson.get("explanation") or ""
    full_text = " ".join(
        filter(
            None,
            [
                lesson.get("topic", ""),
                lesson.get("summary", ""),
                explanation,
                " ".join(lesson.get("examples") or []),
                " ".join(lesson.get("analogies") or []),
                lesson.get("recap", ""),
            ],
        )
    ).lower()

    topic_keywords = keywords(topic)
    topic_hit = bool(topic_keywords) and any(w in full_text for w in topic_keywords)

    checks = {
        "http_ok": resp.status_code == 200,
        "lesson_present": bool(lesson),
        "substantive": len(explanation) >= 40,
        "topic_relevant": topic_hit,
        "examples": len(lesson.get("examples") or []) >= 1
        and len(lesson.get("analogies") or []) >= 1,
        "recap": bool(lesson.get("recap")),
    }
    return {
        "topic": topic,
        "latency_s": round(elapsed, 4),
        "intent": body.get("intent"),
        "lesson_topic": lesson.get("topic", ""),
        **checks,
        "accuracy": sum(checks.values()) / len(checks),
    }


def run() -> dict:
    ensure_server()
    student = f"{STUDENT_PREFIX}_acc"

    rows: list[dict] = []
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        for topic in TOPICS:
            rows.append(_score_topic(client, student, topic))

    mean_accuracy = sum(r["accuracy"] for r in rows) / len(rows)
    mean_latency = latency_stats([r["latency_s"] for r in rows])

    csv_path = save_csv(rows, "accuracy")

    fig = bar_chart(
        "Response Accuracy by Topic",
        [r["topic"] for r in rows],
        [round(r["accuracy"], 3) for r in rows],
        "Accuracy (0-1)",
        text=[f"{r['accuracy']:.2f}" for r in rows],
    )
    fig.add_hline(y=mean_accuracy, line_dash="dot", line_color="#d62728",
                  annotation_text=f"mean = {mean_accuracy:.3f}")
    fig.update_yaxes(range=[0, 1])
    graph_path = write_chart(fig, "response_accuracy")

    return {
        "metric_id": METRIC_ID,
        "metric_name": METRIC_NAME,
        "value": round(mean_accuracy, 4),
        "unit": UNIT,
        "detail": f"mean across {len(rows)} topics; p95 latency {mean_latency['p95']:.2f}s",
        "csv": str(csv_path),
        "graph": str(graph_path),
        "num_cases": len(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
