"""EduAgent — Streamlit dashboard.

Pages
-----
* **Chat** — talk to the AI tutor (optionally produce a follow-up practice quiz).
* **Quiz** — answer the latest generated quiz and get a graded evaluation.
* **Dashboard** — progress charts, weak concepts, memory and recommendations.
* **Documents** — upload PDF/TXT/MD notes and search them during lessons (RAG).

Run with ``streamlit run frontend/app.py`` from the project root.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import EduAgentClient, EduAgentClientError

# --------------------------------------------------------------------------- #
# Session state + client
# --------------------------------------------------------------------------- #


def get_client() -> EduAgentClient:
    if "client" not in st.session_state:
        st.session_state.client = EduAgentClient()
    return st.session_state.client


def student_id() -> str:
    return st.session_state.get("student_id", "").strip() or "student-1"


def set_quiz_in_state(quiz: dict | None) -> None:
    if quiz:
        st.session_state["active_quiz"] = quiz
        st.session_state["active_quiz_id"] = quiz.get("id")


def init_state() -> None:
    st.session_state.setdefault("student_id", "student-1")
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("active_quiz", None)
    st.session_state.setdefault("active_quiz_id", None)
    st.session_state.setdefault("last_evaluation", None)


# --------------------------------------------------------------------------- #
# Page: Chat
# --------------------------------------------------------------------------- #


def render_chat() -> None:
    st.subheader("💬 AI Tutor")

    if st.button("Clear conversation", use_container_width=False):
        st.session_state["chat_history"] = []
        st.rerun()

    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.write(entry["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(entry["content"])

    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([6, 1])
        message = cols[0].text_input(
            "Your message", label_visibility="collapsed", placeholder="Teach me about fractions…"
        )
        with_quiz = cols[1].checkbox("+quiz")
        submitted = st.form_submit_button("Send")

    if submitted and message.strip():
        _send_chat_message(message.strip(), with_quiz)


def _send_chat_message(message: str, with_quiz: bool) -> None:
    st.session_state["chat_history"].append(
        {"role": "user", "content": message}
    )
    try:
        with st.spinner("The tutor is thinking…"):
            result = get_client().chat(student_id(), message, with_quiz=with_quiz)
    except EduAgentClientError as exc:
        st.error(str(exc))
        return

    # Append all assistant messages from the graph in order.
    for msg in result.get("messages") or []:
        if msg.get("role") == "assistant" and msg.get("content"):
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": msg["content"]}
            )

    # If a quiz was produced, stage it for the Quiz page.
    quiz = result.get("quiz")
    if quiz:
        set_quiz_in_state(quiz)
        st.success(f"Quiz ready ({len(quiz.get('questions', []))} questions) — open the Quiz tab to answer it.")

    st.session_state.setdefault("last_result", result)
    st.rerun()


# --------------------------------------------------------------------------- #
# Page: Quiz
# --------------------------------------------------------------------------- #


def render_quiz() -> None:
    st.subheader("📝 Practice Quiz")

    quiz = st.session_state.get("active_quiz")
    if quiz is None:
        st.info(
            "No quiz yet. Ask the tutor to teach a topic with the **+quiz** "
            "checkbox enabled, or load a quiz id below."
        )
        quiz_id = st.text_input("Load a quiz by id")
        if st.button("Load quiz") and quiz_id.strip():
            try:
                quiz = get_client().get_quiz(quiz_id.strip())
                set_quiz_in_state(quiz)
                st.rerun()
            except EduAgentClientError as exc:
                st.error(str(exc))
        return

    st.markdown(f"### {quiz.get('topic')}  ·  {quiz.get('difficulty')}")
    questions: list[dict] = quiz.get("questions") or []
    st.caption(f"{len(questions)} questions — answer them below, then submit for grading.")

    answers: dict[str, Any] = {}

    with st.form("quiz_answers"):
        for idx, question in enumerate(questions, start=1):
            qtype = question.get("type")
            st.markdown(f"**Q{idx}.** {question.get('question')}")
            if qtype == "mcq":
                options = question.get("options") or []
                answer = st.radio(
                    "Select one",
                    options,
                    key=f"quiz_q_{question['id']}",
                    label_visibility="collapsed",
                )
                answers[question["id"]] = answer
            else:
                hint = {
                    "numerical": "Enter a numeric answer",
                    "subjective": "Answer in your own words",
                }.get(qtype, "Your answer")
                answer = st.text_area(
                    hint, key=f"quiz_q_{question['id']}", height=90 if qtype == "subjective" else 60
                )
                answers[question["id"]] = answer or ""

        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        payload = [
            {"question_id": qid, "answer": ans} for qid, ans in answers.items()
        ]
        try:
            with st.spinner("Grading…"):
                result = get_client().submit_quiz(student_id(), quiz.get("id"), payload)
        except EduAgentClientError as exc:
            st.error(str(exc))
            return
        st.session_state["last_evaluation"] = result
        _render_evaluation(result)


def _render_evaluation(result: dict) -> None:
    evaluation = result.get("evaluation") or {}
    if not evaluation:
        st.warning("No evaluation was returned.")
        return
    percent = evaluation.get("percent", 0)
    st.markdown(f"## Result: **{percent:.1f}%**")
    st.progress(min(percent / 100.0, 1.0))
    st.markdown(f"Score: **{evaluation.get('score')}** / **{evaluation.get('max_score')}**")
    st.write(evaluation.get("feedback", ""))

    weak = evaluation.get("weak_concepts") or []
    if weak:
        st.markdown("**Weak areas:** " + ", ".join(weak))
    if evaluation.get("next_lesson"):
        st.markdown(f"**Next lesson:** {evaluation.get('next_lesson')}")

    rec = result.get("recommendations")
    if rec:
        st.markdown("### Recommended next step")
        st.markdown(f"**{rec.get('next_topic')}**")
        st.write(rec.get("rationale", ""))
        for item in rec.get("daily_plan") or []:
            st.markdown(f"- {item}")

    st.success("Evaluation saved. Open the Dashboard tab to see your progress.")


# --------------------------------------------------------------------------- #
# Page: Dashboard
# --------------------------------------------------------------------------- #


def render_dashboard() -> None:
    st.subheader("📊 Learning Dashboard")
    try:
        data = get_client().get_dashboard(student_id())
    except EduAgentClientError as exc:
        st.error(str(exc))
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Name", data.get("name", "Learner"))
    col2.metric("Level", data.get("level", "—"))
    col3.metric("Quiz attempts", data.get("total_attempts", 0))
    col4.metric("Documents", data.get("total_documents", 0))

    scores = data.get("quiz_scores") or []
    history = data.get("attempt_history") or []

    if history:
        st.markdown("### Score history")
        frame = pd.DataFrame(history)
        fig = px.line(
            frame,
            x="created_at",
            y="percent",
            markers=True,
            labels={"created_at": "Attempt", "percent": "Score %"},
        )
        st.plotly_chart(fig, use_container_width=True)
    elif scores:
        st.caption("No attempt history yet — take a quiz to start the chart.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Weak concepts")
        weak = data.get("weak_concepts") or []
        if weak:
            for concept in weak:
                st.markdown(f"- ⚠️ {concept}")
        else:
            st.caption("None identified yet.")
    with right:
        st.markdown("### Topics learned")
        topics = data.get("topics_learned") or []
        if topics:
            st.write(", ".join(topics))
        else:
            st.caption("No topics recorded yet.")

    memory = data.get("recent_memory_facts") or []
    if memory:
        st.markdown("### Recent memory")
        for fact in memory[-5:]:
            st.markdown(f"- *({fact.get('type')})* {fact.get('content')}")

    rec = data.get("latest_recommendation")
    if rec:
        st.markdown("### Latest recommendation")
        st.markdown(f"**{rec.get('next_topic')}** — {rec.get('rationale', '')}")


# --------------------------------------------------------------------------- #
# Page: Documents
# --------------------------------------------------------------------------- #


def render_documents() -> None:
    st.subheader("📚 Study Notes (RAG)")
    st.caption(
        "Upload lecture notes (PDF, TXT or Markdown). The tutor will retrieve "
        "relevant passages when you ask questions on them."
    )

    uploaded = st.file_uploader(
        "Choose a file", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploaded:
        for file in uploaded:
            try:
                with st.spinner(f"Indexing {file.name}…"):
                    result = get_client().upload_document(
                        student_id(), file.name, file.getvalue()
                    )
                if result.get("status") == "ingested":
                    st.success(
                        f"✅ {file.name} — {result.get('chunk_count')} chunks indexed"
                    )
                else:
                    st.error(f"{file.name}: {result.get('error', 'unknown error')}")
            except EduAgentClientError as exc:
                st.error(str(exc))

    try:
        documents = get_client().list_documents(student_id())
    except EduAgentClientError as exc:
        st.error(str(exc))
        return

    if not documents:
        st.info("No documents uploaded yet.")
        return

    st.markdown("### Your documents")
    for doc in documents:
        col1, col2, col3 = st.columns([4, 2, 1])
        col1.markdown(f"**{doc.get('filename')}**")
        col2.caption(f"{doc.get('status')} · {doc.get('chunk_count', 0)} chunks")
        if col3.button("Delete", key=f"del_{doc.get('doc_id')}"):
            try:
                get_client().delete_document(doc.get("doc_id"))
                st.rerun()
            except EduAgentClientError as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="EduAgent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    st.title("🎓 EduAgent — AI Personal Tutor")

    with st.sidebar:
        st.header("Session")
        st.text_input(
            "Student ID",
            value=st.session_state.get("student_id", "student-1"),
            key="student_id",
        )
        try:
            health = get_client().health()
            demo = health.get("demo_mode", False)
            st.caption(
                f"API: {health.get('status', 'unknown')}"
                + (" · demo mode" if demo else "")
            )
        except EduAgentClientError:
            st.caption("API: unreachable — is the backend running?")

        st.divider()
        page = st.radio(
            "Navigate",
            ["Chat", "Quiz", "Dashboard", "Documents"],
        )

    if page == "Chat":
        render_chat()
    elif page == "Quiz":
        render_quiz()
    elif page == "Dashboard":
        render_dashboard()
    else:
        render_documents()


if __name__ == "__main__":
    main()
