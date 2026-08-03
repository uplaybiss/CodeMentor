"""End-to-end verification of the complete EduAgent workflow.

Runs every step of the tutoring lifecycle against a running server and
reports pass/fail for each. This is a *verification* script — it does not
modify the application, only exercises it through the public REST API.

Prereq: start the server first (``python main.py``) then run::

    python e2e_verify.py
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = "http://localhost:8000"
STUDENT = "e2e_student_final"

passed = 0
failed = 0
results: list[dict] = []


def record(step: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    passed += ok
    failed += not ok
    results.append({"step": step, "ok": ok, "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {step}" + (f"  -- {detail}" if detail else ""))


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300.0) as client:
        t0 = time.time()

        # ── 1. Create a student ──────────────────────────────────────────────
        r = client.get(f"/api/v1/students/{STUDENT}")
        ok = r.status_code == 200 and r.json().get("student_id") == STUDENT
        record("1. Create student", ok, f"status={r.status_code}")

        # ── 2. Ask a learning question ───────────────────────────────────────
        r = client.post(
            "/api/v1/chat",
            json={"student_id": STUDENT, "message": "Teach me about photosynthesis", "with_quiz": False},
        )
        body = r.json()
        ok = r.status_code == 200 and body.get("intent") in ("learn", "answer_question")
        record("2. Ask learning question", ok, f"intent={body.get('intent')}")

        # ── 3. Generate a lesson ─────────────────────────────────────────────
        lesson = body.get("lesson")
        ok = lesson is not None and bool(lesson.get("explanation"))
        record("3. Generate lesson", ok, f"topic={lesson.get('topic') if lesson else None}")

        # ── 4. Upload a PDF ──────────────────────────────────────────────────
        with open("data/sample_photosynthesis.pdf", "rb") as fh:
            r = client.post(f"/api/v1/documents/{STUDENT}", files={"file": fh})
        doc = r.json()
        ok = r.status_code == 200 and doc.get("status") == "ingested"
        record("4. Upload PDF", ok, f"doc={doc.get('doc_id')} chunks={doc.get('chunk_count')}")

        # ── 5. Verify RAG retrieves the correct information ──────────────────
        r = client.post(
            "/api/v1/chat",
            json={"student_id": STUDENT, "message": "What is the chemical equation for photosynthesis?"},
        )
        rag_body = r.json()
        sources = rag_body.get("rag_sources") or []
        ok = any("photosynthesis" in s.lower() for s in sources)
        record("5. Verify RAG retrieval", ok, f"sources={sources}")

        # ── 6. Generate a quiz ───────────────────────────────────────────────
        r = client.post(
            "/api/v1/chat",
            json={"student_id": STUDENT, "message": "Give me a practice quiz on photosynthesis", "with_quiz": True},
        )
        quiz_body = r.json()
        quiz = quiz_body.get("quiz")
        ok = quiz is not None and len(quiz.get("questions", [])) > 0
        quiz_id = quiz.get("id") if quiz else None
        record("6. Generate quiz", ok, f"quiz_id={quiz_id} questions={len(quiz.get('questions', [])) if quiz else 0}")

        # ── 7 + 8. Submit quiz answers & evaluate ────────────────────────────
        answers = []
        if quiz:
            for q in quiz["questions"]:
                answers.append({"question_id": q["id"], "answer": q["correct_answer"]})
        r = client.post(
            "/api/v1/chat/submit-quiz",
            json={"quiz_id": quiz_id, "student_id": STUDENT, "answers": answers},
        )
        eval_body = r.json()
        evaluation = eval_body.get("evaluation")
        ok = r.status_code == 200 and evaluation is not None
        record("7. Submit quiz answers", ok, f"status={r.status_code}")
        ok = evaluation is not None and "score" in evaluation
        record(
            "8. Evaluate quiz",
            ok,
            f"score={evaluation.get('score') if evaluation else None}",
        )

        # ── 9. Update student memory ─────────────────────────────────────────
        r = client.get(f"/api/v1/dashboard/{STUDENT}")
        dash = r.json()
        facts = dash.get("recent_memory_facts") or []
        ok = len(facts) > 0
        record("9. Update student memory", ok, f"facts={len(facts)}")

        # ── 10. Generate recommendations ─────────────────────────────────────
        rec = dash.get("latest_recommendation")
        ok = rec is not None and bool(rec.get("next_topic"))
        record("10. Generate recommendations", ok, f"next_topic={rec.get('next_topic') if rec else None}")

        # ── 11. Update dashboard ─────────────────────────────────────────────
        ok = (
            dash.get("student_id") == STUDENT
            and dash.get("total_attempts") is not None
            and dash.get("total_documents") is not None
        )
        record(
            "11. Update dashboard",
            ok,
            f"attempts={dash.get('total_attempts')} docs={dash.get('total_documents')}",
        )

        elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TOTAL E2E TIME: {elapsed:.2f}s")
    print(f"PASSED: {passed}  FAILED: {failed}")
    if failed:
        print("FAILURES:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['step']}: {r['detail']}")
    print("=" * 60)
    with open("logs/e2e_results.json", "w") as fh:
        json.dump({"elapsed_s": round(elapsed, 2), "passed": passed, "failed": failed, "results": results}, fh, indent=2)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
