"""Launch a real loopback server, make HTTP requests, save results, stop owned process."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("evidence/http"))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    records = []
    child_env = dict(os.environ, PYTHONUTF8="1")
    with (out / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "demo.local",
                "--port",
                str(port),
                "--data-dir",
                str(out / "business_data"),
            ],
            stdout=log,
            stderr=log,
            env=child_env,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False
            ) as client:
                deadline = time.monotonic() + 25
                while True:
                    try:
                        ready = client.get("/health")
                        if ready.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Local API did not become ready; inspect server.log")
                    time.sleep(0.2)

                def request(name, method, path, expected=200, **kwargs):
                    response = client.request(method, path, **kwargs)
                    records.append(
                        {
                            "scenario": name,
                            "http_status": response.status_code,
                            "response": response.json(),
                        }
                    )
                    assert response.status_code == expected, (
                        name,
                        response.status_code,
                        response.text,
                    )
                    return response.json()

                lesson = request(
                    "learn_and_quiz",
                    "POST",
                    "/api/v1/chat",
                    json={
                        "student_id": "http_alice",
                        "message": "teach Recursion",
                        "topic": "Recursion",
                        "with_quiz": True,
                        "session_id": "http",
                        "run_id": "learning",
                    },
                )
                assert (
                    lesson["lesson"]["topic"] == "Recursion"
                    and len(lesson["quiz"]["questions"]) == 8
                )
                quiz = lesson["quiz"]
                submission = {
                    "student_id": "http_alice",
                    "quiz_id": quiz["id"],
                    "answers": [
                        {"question_id": q["id"], "answer": q["correct_answer"]}
                        for q in quiz["questions"]
                        if q["type"] != "subjective"
                    ],
                }
                grade = request("submit_quiz", "POST", "/api/v1/chat/submit-quiz", json=submission)
                assert grade["evaluation"]["percent"] == 87.5
                repeat = request(
                    "duplicate_submit", "POST", "/api/v1/chat/submit-quiz", json=submission
                )
                assert repeat["runtime"]["thread_id"] == grade["runtime"]["thread_id"]
                assert repeat["replayed"] is True
                request(
                    "cross_student_submit",
                    "POST",
                    "/api/v1/chat/submit-quiz",
                    expected=403,
                    json={**submission, "student_id": "http_bob"},
                )
                paused = request(
                    "interrupt",
                    "POST",
                    "/api/v1/chat",
                    json={
                        "student_id": "http_alice",
                        "message": "teach DFS",
                        "topic": "DFS",
                        "session_id": "http",
                        "run_id": "paused",
                        "pause_before": "teacher",
                    },
                )
                assert paused["runtime"]["status"] == "interrupted"
                resumed = request(
                    "resume",
                    "POST",
                    "/api/v1/chat/resume",
                    json={
                        "student_id": "http_alice",
                        "session_id": "http",
                        "run_id": "paused",
                    },
                )
                assert resumed["runtime"]["status"] == "completed"
                request(
                    "cross_student_resume",
                    "POST",
                    "/api/v1/chat/resume",
                    expected=404,
                    json={
                        "student_id": "http_bob",
                        "session_id": "http",
                        "run_id": "paused",
                    },
                )
                request(
                    "status",
                    "GET",
                    "/api/v1/chat/runs/http/paused",
                    params={"student_id": "http_alice"},
                )
                request(
                    "invalid_input",
                    "POST",
                    "/api/v1/chat",
                    expected=422,
                    json={"student_id": "http_alice", "message": ""},
                )
                (out / "http-results.json").write_text(
                    json.dumps(
                        {
                            "base_url": f"http://127.0.0.1:{port}",
                            "provider": "demo",
                            "scenarios": records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "scenarios_passed": len(records),
                            "output": str(out / "http-results.json"),
                        },
                        ensure_ascii=False,
                    )
                )
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
