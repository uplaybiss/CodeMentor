"""M2 — Quiz Generation Quality.

Definition
----------
How well-formed, complete and internally consistent a generated quiz is.
A high-quality quiz must match the configured composition, have valid
questions and self-consistent answer keys.

Method
------
For each test *case* (topic) we POST ``/api/v1/chat`` with
``Give me a practice quiz on <topic>`` and ``with_quiz=True``, then run a
battery of checks against the returned :class:`Quiz`.

Composition checks
------------------
1. ``total_count``  — exactly 8 questions (5 MCQ + 2 numerical + 1 subjective)
2. ``mcq_count``    — exactly 5 MCQ questions
3. ``numerical_count`` — exactly 2 numerical questions
4. ``subjective_count`` — exactly 1 subjective question

Per-question checks
-------------------
5. ``text_present`` — every question has non-empty question text
6. ``explained``    — every question carries an explanation
7. ``unique_ids``   — all question ids are unique

MCQ-specific checks
-------------------
8. ``options_ok``   — every MCQ has exactly 4 *unique* options
9. ``answer_in_options`` — every MCQ's correct_answer is one of its options

Answer-key checks
-----------------
10. ``numeric_answers`` — numerical questions have numeric correct answers
11. ``subjective_answered`` — the subjective question has a non-empty answer

Quiz-level checks
-----------------
12. ``topic_set``    — quiz topic is non-empty
13. ``difficulty_valid`` — difficulty is a recognised level

Calculation
-----------
    quality(quiz) = (number of passing checks) / (total checks run)
    quiz_generation_quality = mean quality across quizzes
"""

from __future__ import annotations

import httpx

from common import (
    BASE_URL,
    DEFAULT_RUNS,
    STUDENT_PREFIX,
    bar_chart,
    ensure_server,
    save_csv,
    timed_call,
    write_chart,
)

METRIC_ID = "m2"
METRIC_NAME = "Quiz Generation Quality"
UNIT = "fraction (0-1)"

TOPICS = [
    "Photosynthesis",
    "Newton's laws of motion",
    "Fractions",
    "The periodic table",
    "World War II",
    "The human heart",
]

VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}


def _quiz_checks(quiz: dict | None) -> dict:
    checks = {
        "total_count": False,
        "mcq_count": False,
        "numerical_count": False,
        "subjective_count": False,
        "text_present": False,
        "explained": False,
        "unique_ids": False,
        "options_ok": False,
        "answer_in_options": False,
        "numeric_answers": False,
        "subjective_answered": False,
        "topic_set": False,
        "difficulty_valid": False,
    }
    if not quiz:
        return checks

    questions = quiz.get("questions") or []

    def qtype(q: dict) -> str:
        return str(q.get("type", "")).lower()

    mcqs = [q for q in questions if qtype(q) == "mcq"]
    numeric = [q for q in questions if qtype(q) == "numerical"]
    subjective = [q for q in questions if qtype(q) == "subjective"]

    checks["total_count"] = len(questions) == 8
    checks["mcq_count"] = len(mcqs) == 5
    checks["numerical_count"] = len(numeric) == 2
    checks["subjective_count"] = len(subjective) == 1
    checks["text_present"] = bool(questions) and all(bool(q.get("question")) for q in questions)
    checks["explained"] = bool(questions) and all(bool(q.get("explanation")) for q in questions)
    ids = [q.get("id") for q in questions]
    checks["unique_ids"] = len(ids) == len(set(ids))
    checks["options_ok"] = all(
        isinstance(q.get("options"), list)
        and len(q["options"]) == 4
        and len(set(o.strip() for o in q["options"])) == 4
        for q in mcqs
    ) if mcqs else False
    checks["answer_in_options"] = all(
        q.get("correct_answer") in (q.get("options") or []) for q in mcqs
    ) if mcqs else False

    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    checks["numeric_answers"] = all(
        _is_number(q.get("correct_answer")) for q in numeric
    ) if numeric else False
    checks["subjective_answered"] = bool(subjective) and bool(subjective[0].get("correct_answer"))
    checks["topic_set"] = bool(quiz.get("topic"))
    checks["difficulty_valid"] = str(quiz.get("difficulty", "")).lower() in VALID_DIFFICULTIES
    return checks


def _score_topic(client: httpx.Client, student: str, topic: str) -> dict:
    payload = {
        "student_id": student,
        "message": f"Give me a practice quiz on {topic}",
        "with_quiz": True,
    }
    resp, elapsed = timed_call(client.post, "/api/v1/chat", json=payload)
    body = resp.json() if resp.content else {}
    quiz = body.get("quiz")
    checks = _quiz_checks(quiz)

    questions = (quiz or {}).get("questions") or []
    counts = {
        "total": len(questions),
        "mcq": sum(1 for q in questions if str(q.get("type", "")).lower() == "mcq"),
        "numerical": sum(1 for q in questions if str(q.get("type", "")).lower() == "numerical"),
        "subjective": sum(1 for q in questions if str(q.get("type", "")).lower() == "subjective"),
    }
    return {
        "topic": topic,
        "latency_s": round(elapsed, 4),
        "http_ok": resp.status_code == 200,
        "quiz_id": (quiz or {}).get("id"),
        "difficulty": (quiz or {}).get("difficulty"),
        **counts,
        **checks,
        "quality": sum(checks.values()) / len(checks) if quiz else 0.0,
    }


def run() -> dict:
    ensure_server()
    student = f"{STUDENT_PREFIX}_quiz"

    rows: list[dict] = []
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        for topic in TOPICS:
            rows.append(_score_topic(client, student, topic))

    mean_quality = sum(r["quality"] for r in rows) / len(rows)
    csv_path = save_csv(rows, "quiz_quality")

    fig = bar_chart(
        "Quiz Generation Quality by Topic",
        [r["topic"] for r in rows],
        [round(r["quality"], 3) for r in rows],
        "Quality (0-1)",
        text=[f"{r['quality']:.2f}" for r in rows],
    )
    fig.add_hline(y=mean_quality, line_dash="dot", line_color="#d62728",
                  annotation_text=f"mean = {mean_quality:.3f}")
    fig.update_yaxes(range=[0, 1])
    graph_path = write_chart(fig, "quiz_quality")

    return {
        "metric_id": METRIC_ID,
        "metric_name": METRIC_NAME,
        "value": round(mean_quality, 4),
        "unit": UNIT,
        "detail": f"mean across {len(rows)} quizzes ({len(TOPICS)} topics)",
        "csv": str(csv_path),
        "graph": str(graph_path),
        "num_cases": len(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
