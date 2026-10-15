"""Native Streamlit components, client contracts and actual injected FastAPI workflows."""

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from config.presentation import TOPIC_LABELS
from frontend.api_client import EduAgentClient, EduAgentClientError
from frontend.session import (
    accept_chat,
    clear_chat,
    init_session,
    prepare_chat,
    retry_chat,
    submit_once,
    switch_profile,
)


@pytest.fixture
def ui(client, container):
    # The test profile disables the upload UI, without loading a real embedding model.
    container.settings.vector_store.enabled = False
    calls = []

    def bridge(request):
        calls.append((request.method, request.url.path))
        response = client.request(
            request.method,
            request.url.path,
            params=dict(request.url.params),
            content=request.content,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return httpx.Response(response.status_code, json=response.json())

    http = EduAgentClient("http://testserver", transport=httpx.MockTransport(bridge))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "frontend/app.py"), default_timeout=25)
    app.session_state["client"] = http
    app.run()
    yield app, calls, http
    http.close()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_home_is_chinese_empty_state_and_no_json(ui):
    app, calls, _ = ui
    assert not app.exception
    assert app.sidebar.radio[0].options == [
        "学习首页",
        "AI 辅导",
        "专项练习",
        "学习画像",
        "学习资料",
    ]
    assert any("今天准备学习什么" in m.value for m in app.markdown)
    assert [m.value for m in app.metric][:2] == ["0", "0"]
    assert len([b for b in app.button if b.label == "开始学习"]) == 6
    assert not app.get("json")
    assert ("GET", "/health") in calls
    assert not any(path == "/api/v1/chat" for _, path in calls)


def test_complete_ui_learning_quiz_score_profile_and_no_rerun_submit(ui, container):
    app, calls, _ = ui
    app.button(key="start_Recursion").click().run()
    assert not app.exception
    assert app.session_state["page"] == "AI 辅导"
    assert app.session_state["active_quiz"]["topic"] == "Recursion"
    assert any("```python" in m.value for m in app.markdown)
    assert len([p for _, p in calls if p == "/api/v1/chat"]) == 1
    button(app, "进入专项练习").click().run()
    assert not app.exception
    quiz = app.session_state["active_quiz"]
    assert len(quiz["questions"]) == 8
    assert not any("参考答案" in str(m.value) for m in app.markdown)
    # Deliberately miss the first MCQ to verify actual per-question feedback.
    first = quiz["questions"][0]
    app.radio(key=f"answer_{first['id']}").set_value(first["options"][1]).run()
    for q in quiz["questions"][1:]:
        if q["type"] == "mcq":
            app.radio(key=f"answer_{q['id']}").set_value(q["correct_answer"]).run()
        elif q["type"] == "numerical":
            app.text_input(key=f"answer_{q['id']}").set_value(q["correct_answer"]).run()
        else:
            app.text_area(key=f"answer_{q['id']}").set_value(
                "递归需要可以到达的终止条件，并逐步缩小问题。"
            ).run()
    button(app, "提交答案").click().run()
    assert not app.exception
    result = app.session_state["evaluations"][quiz["id"]]
    assert result["evaluation"]["percent"] == 85.0
    assert len(result["question_feedback"]) == 8
    assert result["question_feedback"][0]["is_correct"] is False
    assert any("参考答案" in str(m.value) for m in app.markdown)
    submits = len([p for _, p in calls if p == "/api/v1/chat/submit-quiz"])
    app.run()
    assert len([p for _, p in calls if p == "/api/v1/chat/submit-quiz"]) == submits == 1
    assert len(container.quiz_repo.list_attempts("student-1")) == 1
    button(app, "查看学习画像").click().run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.get("plotly_chart")
    assert any("最新学习建议" in m.value for m in app.markdown)


def test_clear_chat_keeps_business_memory(ui, container):
    app, _, _ = ui
    app.button(key="start_Linked Lists").click().run()
    before = container.conversation_repo.count("student-1")
    assert before > 0
    previous_session = app.session_state["ui_session_id"]
    app.button(key="clear_chat").click().run()
    assert not app.exception
    assert app.session_state["chat_history"] == []
    assert app.session_state["ui_session_id"] != previous_session
    assert container.conversation_repo.count("student-1") == before


def test_disabled_rag_has_no_upload_control(ui):
    app, _, _ = ui
    app.sidebar.radio[0].set_value("学习资料").run()
    assert not app.exception
    assert not app.get("file_uploader")
    assert not any(b.label == "上传资料" for b in app.button)
    assert any("未开启资料检索" in m.value for m in app.markdown)


def test_offline_home_and_all_pages_do_not_crash():
    class Offline:
        def health(self):
            raise EduAgentClientError("学习服务未连接。")

    app = AppTest.from_file(str(Path(__file__).parents[1] / "frontend/app.py"))
    app.session_state["client"] = Offline()
    app.run()
    assert not app.exception
    assert any("未连接" in w.value for w in app.warning)
    assert [m.value for m in app.metric] == ["—"] * 4
    for page in ("AI 辅导", "专项练习", "学习画像", "学习资料"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_dashboard_failure_is_safe(ui, monkeypatch):
    app, _, http = ui

    def failed(_student):
        raise EduAgentClientError("学习概览暂时不可用。")

    monkeypatch.setattr(http, "get_dashboard", failed)
    app.run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["—"] * 4


def test_client_address_and_run_contract(monkeypatch):
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "http://127.0.0.1:8013")
    requests = []

    def capture(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={})

    with_client = EduAgentClient(transport=httpx.MockTransport(capture))
    assert with_client.base_url == "http://127.0.0.1:8013"
    with_client.chat("s", "你好", "Recursion", True, session_id="session", run_id="run")
    assert requests[0]["session_id"] == "session" and requests[0]["run_id"] == "run"
    with_client.submit_quiz("s", "q", [{"question_id": "q1", "answer": "完整选项"}])
    assert "run_id" not in requests[1]
    assert requests[1]["answers"][0]["answer"] == "完整选项"
    with_client.close()


@pytest.mark.parametrize(
    "status,body",
    [
        (500, {"message": "Traceback API_KEY_SECRET"}),
        (422, {"detail": []}),
        (403, {"error": "TOOL_FORBIDDEN"}),
    ],
)
def test_client_errors_never_expose_traceback_or_credentials(status, body):
    http = EduAgentClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    )
    with pytest.raises(EduAgentClientError) as error:
        http.health()
    assert "Traceback" not in str(error.value) and "API_KEY" not in str(error.value)
    assert any("\u4e00" <= c <= "\u9fff" for c in str(error.value))
    http.close()


def test_ui_request_ids_switch_profile_and_cached_submission():
    state = {}
    init_session(state)
    one = prepare_chat(state, "学习递归", "Recursion", True)
    two = prepare_chat(state, "学习递归", "Recursion", True)
    assert one["session_id"] == two["session_id"] and one["run_id"] != two["run_id"]
    clear_chat(state)
    assert state["ui_session_id"] != one["session_id"]

    class Client:
        calls = 0

        def submit_quiz(self, *args):
            self.calls += 1
            return {"evaluation": {"percent": 100}}

    client = Client()
    assert submit_once(state, client, {"id": "q"}, []) == submit_once(
        state, client, {"id": "q"}, []
    )
    assert client.calls == 1
    switch_profile(state, "第二位同学")
    assert state["student_id"] != "student-1" and state["evaluations"] == {}


def test_dashboard_uses_current_mastery_not_stale_weak_list(client, container):
    repo = container.student_repo
    repo.get_or_create("updated")
    repo.add_weak_concept("updated", "Recursion")
    repo.upsert_concept_score("updated", "Recursion", 0.3)
    repo.upsert_concept_score("updated", "Recursion", 1.0)
    assert "Recursion" in repo.get_or_404("updated").weak_concepts
    result = client.get("/api/v1/dashboard/updated").json()
    assert result["weak_concepts"] == []
    assert result["concept_scores"][0]["proficiency"] == 0.72


def test_enabled_document_page_lists_and_removes_real_fake_embedder_document(ui, client, container):
    app, _, _ = ui
    container.settings.vector_store.enabled = True
    uploaded = client.post(
        "/api/v1/documents/student-1",
        files={"file": ("递归笔记.txt", "递归需要终止条件".encode(), "text/plain")},
    )
    assert uploaded.status_code == 200
    doc_id = uploaded.json()["doc_id"]
    app.sidebar.radio[0].set_value("学习资料").run()
    assert not app.exception
    assert app.get("file_uploader")
    assert any("递归笔记.txt" in str(m.value) for m in app.markdown)
    app.button(key=f"delete_{doc_id}").click().run()
    assert not app.exception
    assert client.get("/api/v1/documents/student-1").json() == []


def test_dashboard_count_is_not_capped_by_display_history(client, ctx):
    from models.schemas import EvaluationResult
    from tests.test_harness import seed_quiz

    quiz = seed_quiz(ctx)
    evaluation = EvaluationResult(score=0, max_score=2, percent=0, feedback="测试记录")
    for index in range(52):
        ctx.quiz_repo.save_attempt(
            attempt_id=f"count_{index}",
            student_id="alice",
            quiz_id=quiz.id,
            answers={},
            evaluation=evaluation,
        )
    result = client.get("/api/v1/dashboard/alice").json()
    assert result["total_attempts"] == 52
    assert len(result["attempt_history"]) == 50
    assert all(item["topic"] == "Recursion" for item in result["attempt_history"])


def test_lost_response_retry_uses_existing_run_without_second_lesson(client, container):
    calls = []
    lose_response = True

    def bridge(request):
        nonlocal lose_response
        calls.append(request.url.path)
        response = client.request(
            request.method,
            request.url.path,
            params=dict(request.url.params),
            content=request.content,
            headers={"content-type": "application/json"},
        )
        if request.url.path == "/api/v1/chat" and lose_response:
            lose_response = False
            raise httpx.ReadTimeout("lost response after commit", request=request)
        return httpx.Response(response.status_code, json=response.json())

    http = EduAgentClient("http://testserver", transport=httpx.MockTransport(bridge))
    state = {}
    init_session(state)
    request = prepare_chat(state, "请讲解递归", "Recursion", True)
    with pytest.raises(EduAgentClientError):
        http.chat(**request)
    assert len(container.lesson_repo.list_lessons("student-1")) == 1
    result = retry_chat(http, request)
    accept_chat(state, result)
    assert state["failed_chat"] is None
    assert len(container.lesson_repo.list_lessons("student-1")) == 1
    assert calls.count("/api/v1/chat") == 1 and calls[-1] == "/api/v1/chat/resume"
    http.close()


@pytest.mark.parametrize("topic", list(TOPIC_LABELS))
def test_chinese_quiz_options_are_gradable_and_ids_unchanged(client, topic):
    data = client.post(
        "/api/v1/chat",
        json={"student_id": "zh", "message": "请讲解并生成练习", "topic": topic, "with_quiz": True},
    ).json()
    quiz = data["quiz"]
    assert quiz["topic"] == topic
    assert len({q["question"] for q in quiz["questions"]}) == 8
    for q in quiz["questions"]:
        assert any("\u4e00" <= c <= "\u9fff" for c in q["question"])
        if q["type"] == "mcq":
            assert q["correct_answer"] in q["options"]
    answers = [{"question_id": q["id"], "answer": q["correct_answer"]} for q in quiz["questions"]]
    grade = client.post(
        "/api/v1/chat/submit-quiz",
        json={"student_id": "zh", "quiz_id": quiz["id"], "answers": answers},
    ).json()
    assert grade["evaluation"]["percent"] == 97.5
    assert len(grade["question_feedback"]) == 8
    assert all(q["is_correct"] for q in grade["question_feedback"])
