"""M6 — Memory Recall Accuracy.

Definition
----------
How faithfully important conversation content persists into long-term memory
and is later *recalled* (surfaced) by the dashboard. A high score means the
Memory agent stored the substance of what the student said and the system can
re-display it.

Method
------
For each test *case* (conversation) a *fresh* student sends one message. After
the tutoring run we read ``recent_memory_facts`` from the dashboard and compare
the message's keywords with the keywords stored in the facts:

    recall    = |message_keywords ∩ fact_keywords| / |message_keywords|
    precision = |message_keywords ∩ fact_keywords| / |fact_keywords|

* ``recall`` answers "did the student's content get stored?".
* ``precision`` answers "is everything stored relevant to that message?".

Calculation
-----------
    memory_recall_accuracy = mean recall across conversations
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
    save_csv,
    timed_call,
    write_chart,
)

METRIC_ID = "m6"
METRIC_NAME = "Memory Recall Accuracy"
UNIT = "fraction (0-1)"

CONVERSATIONS = [
    "Teach me about photosynthesis",
    "I want to understand Newton's laws of motion",
    "Explain fractions and how to add them",
    "Why is the sky blue?",
    "Help me with the periodic table",
]


def run() -> dict:
    ensure_server()
    rows: list[dict] = []
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        for i, message in enumerate(CONVERSATIONS):
            student = f"{STUDENT_PREFIX}_mem_{i}"

            # One tutoring turn (learn intent).
            client.post("/api/v1/chat", json={"student_id": student, "message": message})

            # Read back what long-term memory retained.
            dash, _ = timed_call(client.get, f"/api/v1/dashboard/{student}")
            facts = dash.json().get("recent_memory_facts") or []
            fact_text = " ".join(f.get("content", "") for f in facts)

            msg_kw = keywords(message)
            fact_kw = keywords(fact_text)
            overlap = msg_kw & fact_kw

            recall = len(overlap) / len(msg_kw) if msg_kw else 1.0
            precision = len(overlap) / len(fact_kw) if fact_kw else 0.0

            rows.append({
                "student": student,
                "message": message,
                "n_facts_stored": len(facts),
                "fact_types": "|".join(f.get("type", "?") for f in facts),
                "n_message_keywords": len(msg_kw),
                "n_fact_keywords": len(fact_kw),
                "overlap": len(overlap),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            })

    mean_recall = sum(r["recall"] for r in rows) / len(rows)
    mean_precision = sum(r["precision"] for r in rows) / len(rows)
    csv_path = save_csv(rows, "memory_recall")

    fig = bar_chart(
        "Memory Recall by Conversation",
        [f"C{i+1}" for i in range(len(rows))],
        [round(r["recall"], 3) for r in rows],
        "Recall (0-1)",
        text=[f"{r['recall']:.2f}" for r in rows],
    )
    fig.add_trace(
        dict(type="bar", name="precision",
             x=[f"C{i+1}" for i in range(len(rows))],
             y=[round(r["precision"], 3) for r in rows],
             marker_color="#E45756", opacity=0.8)
    )
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]))
    fig.add_hline(y=mean_recall, line_dash="dot", line_color="#2CA02C",
                  annotation_text=f"mean recall = {mean_recall:.3f}")
    graph_path = write_chart(fig, "memory_recall")

    return {
        "metric_id": METRIC_ID,
        "metric_name": METRIC_NAME,
        "value": round(mean_recall, 4),
        "unit": UNIT,
        "detail": f"mean recall {mean_recall:.3f}, mean precision {mean_precision:.3f} "
                  f"across {len(rows)} conversations",
        "csv": str(csv_path),
        "graph": str(graph_path),
        "num_cases": len(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
