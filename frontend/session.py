"""A widget rerun is not a new business submission."""

import hashlib
from uuid import uuid4

from frontend.api_client import EduAgentClientError


def init_session(state):
    defaults = {
        "student_id": "student-1",
        "profile_name": "同学01",
        "profile_input": "同学01",
        "page": "学习首页",
        "current_topic": "Python Basics",
        "topic_picker": "Python Basics",
        "chat_history": [],
        "active_quiz": None,
        "evaluations": {},
        "pending_answers": {},
        "failed_chat": None,
        "queued_chat": None,
        "ui_session_id": uuid4().hex,
        "with_quiz": True,
        "busy": False,
        "notice": None,
    }
    for key, value in defaults.items():
        state.setdefault(key, value)


def clear_chat(state):
    state["scroll_to_top"] = True
    state["chat_history"] = []
    state["failed_chat"] = None
    state["queued_chat"] = None
    state["ui_session_id"] = uuid4().hex
    state["notice"] = "已清空当前聊天界面；学习画像、错题和历史记录仍然保留。"


def switch_profile(state, name):
    name = name.strip() or "同学01"
    student_id = (
        "student-1"
        if name == "同学01"
        else "learner_" + hashlib.sha256(name.encode()).hexdigest()[:20]
    )
    if student_id == state["student_id"]:
        return
    clear_chat(state)
    state.update(
        student_id=student_id,
        profile_name=name,
        active_quiz=None,
        evaluations={},
        pending_answers={},
        current_topic="Python Basics",
        topic_picker="Python Basics",
        page="学习首页",
        busy=False,
        notice="已切换学习档案，原档案的历史记录仍然保留。",
    )


def prepare_chat(state, message, topic, with_quiz):
    state["chat_error"] = None
    request = {
        "student_id": state["student_id"],
        "message": message,
        "topic": topic,
        "with_quiz": with_quiz,
        "session_id": state["ui_session_id"],
        "run_id": uuid4().hex,
    }
    state["chat_history"].append({"role": "user", "content": message})
    state["failed_chat"] = request
    return request


def retry_chat(client, request):
    try:
        client.run_status(request["student_id"], request["session_id"], request["run_id"])
    except EduAgentClientError as exc:
        if exc.status == 404:
            return client.chat(**request)
        raise
    return client.resume(request["student_id"], request["session_id"], request["run_id"])


def accept_chat(state, result):
    state["chat_error"] = None
    for message in result.get("messages") or []:
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and message.get("node") != "quiz"
        ):
            state["chat_history"].append({"role": "assistant", "content": message["content"]})
    if result.get("quiz"):
        state["active_quiz"] = result["quiz"]
    state["failed_chat"] = None
    state["last_chat_result"] = result


def submit_once(state, client, quiz, answers):
    quiz_id = quiz["id"]
    if quiz_id in state["evaluations"]:
        return state["evaluations"][quiz_id]
    payload = state["pending_answers"].setdefault(quiz_id, answers)
    # Omit run_id: Runtime hashes the same student/quiz/answers across reruns.
    result = client.submit_quiz(state["student_id"], quiz_id, payload)
    if not result.get("evaluation"):
        raise EduAgentClientError("评分尚未完成，请稍后重试同一份答案。")
    state["evaluations"][quiz_id] = result
    state["pending_answers"].pop(quiz_id, None)
    return result
