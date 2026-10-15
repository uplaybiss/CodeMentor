"""CodeMentor Chinese student workbench, using native Streamlit components."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config.presentation import (
    DIFFICULTY_LABELS,
    LEVEL_LABELS,
    MEMORY_LABELS,
    TOPIC_DESCRIPTIONS,
    TOPIC_LABELS,
    display_memory,
    recommendation_text,
    topic_label,
)
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

PAGES = ["学习首页", "AI 辅导", "专项练习", "学习画像", "学习资料"]


def get_client():
    if "client" not in st.session_state:
        st.session_state["client"] = EduAgentClient()
    return st.session_state["client"]


def navigate(page):
    st.session_state["page"] = page
    st.session_state["scroll_to_top"] = True


def sidebar_navigation():
    st.session_state["scroll_to_top"] = True


def change_profile():
    switch_profile(st.session_state, st.session_state["profile_input"])


def start_topic(topic, message=None, practice=False):
    st.session_state["current_topic"] = topic
    st.session_state["topic_picker"] = topic
    st.session_state["queued_chat"] = (
        message or f"请带我学习{topic_label(topic)}，用简单的代码示例解释。"
    )
    st.session_state["queued_with_quiz"] = True if practice else st.session_state["with_quiz"]
    navigate("AI 辅导")


def header(title, subtitle):
    st.markdown(
        f'<div class="page-heading"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def overview(data):
    weak = data.get("weak_concepts") or []
    scores = data.get("concept_scores") or []
    recommendation = data.get("latest_recommendation") or {}
    cols = st.columns(4)
    values = [
        ("已完成测验", str(data.get("total_attempts", 0)), "每次成功提交并评分的测验。"),
        (
            "已学习知识点",
            str(len(set(data.get("topics_learned") or []))),
            "已完成讲解的知识点，不代表已掌握。",
        ),
        (
            "当前薄弱点",
            topic_label(weak[0]) if weak else ("暂未发现" if scores else "尚待评估"),
            "依据当前掌握度，低于60%需要复习。",
        ),
        (
            "推荐下一步",
            topic_label(recommendation.get("next_topic")) if recommendation else "尚无建议",
            "根据最近的画像与先修关系制定。",
        ),
    ]
    for col, (label, value, help_text) in zip(cols, values, strict=True):
        col.metric(label, value, help=help_text, border=True)


def render_home(data, ready):
    st.markdown(
        '<div class="hero"><div class="eyebrow">专属编程学习工作台</div>'
        "<h1>今天准备学习什么？</h1>"
        "<p>根据学习记录，制定适合你的编程学习计划。</p></div>",
        unsafe_allow_html=True,
    )
    st.subheader("学习概览")
    if data is not None:
        overview(data)
    else:
        for col, label in zip(
            st.columns(4), ["已完成测验", "已学习知识点", "当前薄弱点", "推荐下一步"], strict=True
        ):
            col.metric(label, "—", border=True)
        st.caption("连接学习服务后，将显示你的真实学习记录。")
    st.subheader("快速开始学习")
    st.caption("选择一个知识点，先听讲解，再用练习检查理解。")
    topics = list(TOPIC_LABELS)
    for offset in (0, 3):
        for col, topic in zip(st.columns(3), topics[offset : offset + 3], strict=True):
            with col.container(border=True):
                st.markdown(f"#### {topic_label(topic)}")
                st.caption(TOPIC_DESCRIPTIONS[topic])
                st.button(
                    "开始学习",
                    key=f"start_{topic}",
                    width="stretch",
                    type="primary" if topic == "Python Basics" else "secondary",
                    disabled=not ready,
                    on_click=start_topic,
                    args=(topic,),
                )
    st.subheader("个性化学习建议")
    with st.container(border=True):
        rec = (data or {}).get("latest_recommendation")
        if rec:
            st.markdown(f"**下一步 · {topic_label(rec.get('next_topic'))}**")
            st.write(recommendation_text(rec, (data or {}).get("concept_scores")))
            if rec.get("next_topic") in TOPIC_LABELS:
                st.button(
                    "按建议开始学习",
                    disabled=not ready,
                    on_click=start_topic,
                    args=(rec["next_topic"],),
                )
        else:
            st.markdown("**从一份小练习开始认识自己。**")
            st.write("先选择感兴趣的知识点。完成讲解或测验后，这里会出现下一步学习建议。")
            st.caption("还没有学习记录也没关系，系统会从基础开始。")


def send_message(message, ready, *, retry=False):
    if not ready:
        st.error("学习服务尚未连接，消息没有发送。")
        return
    request = (
        st.session_state["failed_chat"]
        if retry
        else prepare_chat(
            st.session_state,
            message,
            st.session_state["current_topic"],
            st.session_state.pop("queued_with_quiz", st.session_state["with_quiz"]),
        )
    )
    st.session_state["busy"] = True
    try:
        with st.spinner("正在为你准备讲解和学习建议…"):
            result = retry_chat(get_client(), request) if retry else get_client().chat(**request)
        if result.get("runtime", {}).get("status") == "interrupted":
            st.warning("这次辅导已暂停，可以点击继续完成。")
            return
        accept_chat(st.session_state, result)
    except EduAgentClientError as exc:
        st.session_state["chat_error"] = str(exc)
        st.error(str(exc))
        st.caption("这次请求已保留。继续时会先查询进度，避免重复创建学习记录。")
    finally:
        st.session_state["busy"] = False
    st.rerun()


def render_chat(data, ready):
    header("AI 辅导", "提出你的问题，或让老师带你从一个例子开始。")
    controls = st.columns([3, 3, 2])
    with controls[0]:
        st.selectbox(
            "当前知识点",
            list(TOPIC_LABELS),
            format_func=topic_label,
            key="topic_picker",
            on_change=lambda: st.session_state.update(
                current_topic=st.session_state["topic_picker"]
            ),
        )
    with controls[1]:
        st.checkbox("讲解后生成练习", key="with_quiz")
    with controls[2]:
        st.button(
            "清空界面会话",
            key="clear_chat",
            width="stretch",
            on_click=clear_chat,
            args=(st.session_state,),
        )
    st.caption("清空只影响当前聊天界面，不会删除长期学习画像、错题或历史记录。")
    if not st.session_state["chat_history"]:
        with st.container(border=True):
            st.markdown(f"### 一起学{topic_label(st.session_state['current_topic'])}")
            st.write("描述哪里不理解，老师会用简短讲解和代码示例帮你拆解。")
        weak_topic = ((data or {}).get("latest_recommendation") or {}).get("next_topic")
        if weak_topic not in TOPIC_LABELS:
            weak_topic = st.session_state["current_topic"]
        suggestions = [
            ("用简单的例子讲一下递归", "Recursion"),
            ("我总是搞不懂二叉树遍历", "Binary Trees"),
            ("根据我的薄弱点推荐练习", weak_topic),
            ("请帮我复习链表", "Linked Lists"),
        ]
        for index, (message, topic) in enumerate(suggestions):
            st.button(
                message,
                key=f"suggestion_{index}",
                disabled=not ready,
                on_click=start_topic,
                args=(topic, message, "练习" in message),
            )
    for entry in st.session_state["chat_history"]:
        with st.chat_message(entry["role"]):
            st.caption("你" if entry["role"] == "user" else "CodeMentor 老师")
            st.markdown(entry["content"])
    quiz = st.session_state["active_quiz"]
    if quiz:
        with st.container(border=True):
            st.markdown(f"**{topic_label(quiz.get('topic'))}练习已准备好**")
            st.caption(
                f"共{len(quiz.get('questions') or [])}题 · {DIFFICULTY_LABELS.get(quiz.get('difficulty'), '初级')}"
            )
            st.button("进入专项练习", type="primary", on_click=navigate, args=("专项练习",))
    if st.session_state["failed_chat"]:
        if st.session_state.get("chat_error"):
            st.error(st.session_state["chat_error"])
        st.warning("上一次辅导尚未完成。你可以继续同一次请求，或发送新消息。")
        if st.button("继续这次辅导", disabled=not ready):
            send_message("", ready, retry=True)
    queued = st.session_state.pop("queued_chat", None)
    if queued:
        send_message(queued, ready)
    message = st.chat_input(
        "输入你的学习问题，例如：递归为什么需要终止条件？",
        disabled=not ready or st.session_state["busy"],
    )
    if message and message.strip():
        send_message(message.strip(), ready)


def render_evaluation(result, quiz):
    evaluation = result.get("evaluation") or {}
    percent = float(evaluation.get("percent", 0))
    st.success("测验已评分并保存，学习画像也已更新。")
    cols = st.columns(3)
    cols[0].metric("本次得分", f"{percent:.1f}%", border=True)
    cols[1].metric(
        "获得分数",
        f"{evaluation.get('score', 0):g} / {evaluation.get('max_score', 0):g}",
        border=True,
    )
    cols[2].metric("本次知识点", topic_label(quiz.get("topic")), border=True)
    st.progress(min(max(percent / 100, 0), 1), text="本次测验得分")
    st.write(evaluation.get("feedback", ""))
    details = result.get("question_feedback") or []
    if details:
        st.subheader("逐题反馈")
        for index, detail in enumerate(details, 1):
            status = "回答正确" if detail.get("is_correct") else "需要复习"
            with st.expander(f"第{index}题 · {status}", expanded=not detail.get("is_correct")):
                st.write("你的回答：" + (detail.get("submitted") or "未填写"))
                st.write("参考答案：" + detail.get("reference_answer", ""))
                st.write("解析：" + detail.get("explanation", "请对照讲解再尝试一次。"))
                st.caption(f"获得{detail.get('earned', 0):g} / {detail.get('max_score', 0):g}分")
    else:
        st.info("这份历史测验没有逐题明细，可根据反馈复习对应知识点。")
    rec = result.get("recommendations")
    if rec:
        with st.container(border=True):
            st.markdown(f"### 下一步 · {topic_label(rec.get('next_topic'))}")
            st.write(recommendation_text(rec))
            if rec.get("next_topic") in TOPIC_LABELS:
                st.button("继续学习推荐知识点", on_click=start_topic, args=(rec["next_topic"],))
    nav = st.columns(2)
    nav[0].button("返回学习首页", width="stretch", on_click=navigate, args=("学习首页",))
    nav[1].button("查看学习画像", width="stretch", on_click=navigate, args=("学习画像",))


def render_quiz(ready, demo):
    header("专项练习", "用一份小测验检查理解，逐题反馈帮你找到下一步。")
    quiz = st.session_state["active_quiz"]
    if not quiz:
        with st.container(border=True):
            st.markdown("### 先准备一份适合你的练习")
            st.write("在AI辅导中选择知识点，并勾选“讲解后生成练习”。")
            st.button("去 AI 辅导", type="primary", on_click=navigate, args=("AI 辅导",))
        return
    st.markdown(f"### {topic_label(quiz.get('topic'))}专项练习")
    questions = quiz.get("questions") or []
    st.caption(
        f"{DIFFICULTY_LABELS.get(quiz.get('difficulty'), '初级')} · 共{len(questions)}题 · 答案提交后显示解析"
    )
    if demo:
        st.info("当前为本地演示：客观题按答案评分，非空主观题使用固定评分，仅用于体验流程。")
    result = st.session_state["evaluations"].get(quiz["id"])
    if result:
        render_evaluation(result, quiz)
        return
    frozen = quiz["id"] in st.session_state["pending_answers"]
    answered = sum(bool(st.session_state.get(f"answer_{q['id']}")) for q in questions)
    st.progress(
        answered / len(questions) if questions else 0,
        text=f"作答进度 · {answered} / {len(questions)}",
    )
    answers = []
    for index, question in enumerate(questions, 1):
        with st.container(border=True):
            kind = {"mcq": "选择题", "numerical": "数值题", "subjective": "主观题"}.get(
                question["type"], "题目"
            )
            st.caption(f"第{index}题 · {kind}")
            st.markdown(question.get("question", ""))
            key = f"answer_{question['id']}"
            if question["type"] == "mcq":
                value = st.radio(
                    "请选择一个答案",
                    question.get("options") or [],
                    index=None,
                    key=key,
                    disabled=frozen or not ready,
                )
            elif question["type"] == "numerical":
                value = st.text_input(
                    "填写数值答案",
                    placeholder="请输入计算结果",
                    key=key,
                    disabled=frozen or not ready,
                    max_chars=100,
                )
            else:
                value = st.text_area(
                    "用自己的话回答",
                    placeholder="写下你的理解，也可以补充代码示例。",
                    key=key,
                    height=140,
                    max_chars=8000,
                    disabled=frozen or not ready,
                )
            answers.append({"question_id": question["id"], "answer": value or ""})
    if frozen:
        st.warning("上次提交的答案已保留。重试会提交同一份答案，避免重复记录。")
    st.caption("未填写的题目会按空答案评分。选择题会提交完整选项文字。")
    if st.button(
        "重试评分" if frozen else "提交答案",
        type="primary",
        key="submit_answers",
        disabled=not ready or st.session_state["busy"],
        width="stretch",
    ):
        st.session_state["busy"] = True
        try:
            with st.spinner("正在评分并更新学习画像…"):
                submit_once(st.session_state, get_client(), quiz, answers)
                st.session_state["scroll_to_top"] = True
        except EduAgentClientError as exc:
            st.error(str(exc))
        else:
            st.rerun()
        finally:
            st.session_state["busy"] = False


def render_dashboard(data):
    header("学习画像", "查看真实学习记录，理解系统为什么这样安排你的下一步。")
    if data is None:
        st.info("连接学习服务后，就可以查看画像与学习进度。")
        return
    overview(data)
    st.caption(
        f"学习阶段：{LEVEL_LABELS.get(data.get('level'), '初学者')} · 讲解记录与掌握度分开统计。"
    )
    history = data.get("attempt_history") or []
    st.subheader("历史得分趋势")
    if history:
        frame = pd.DataFrame(history)
        frame["测验次序"] = range(
            data["total_attempts"] - len(frame) + 1, data["total_attempts"] + 1
        )
        frame["知识点"] = frame["topic"].map(topic_label)
        fig = px.line(
            frame,
            x="测验次序",
            y="percent",
            markers=True,
            hover_data=["知识点"],
            labels={"percent": "得分（%）"},
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(
            height=300,
            yaxis_range=[0, 105],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f8fafc",
            margin={"l": 12, "r": 12, "t": 20, "b": 12},
            font={"family": "Microsoft YaHei, sans-serif", "color": "#334155"},
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("展示最近50次实际测验的得分；单次分数波动不等同于学习效果结论。")
    else:
        with st.container(border=True):
            st.write("还没有测验记录。完成一次专项练习后，就能在这里看到得分趋势。")
            st.button("去完成首次练习", on_click=navigate, args=("专项练习",))
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### 当前薄弱知识点")
            weak = data.get("weak_concepts") or []
            scores = data.get("concept_scores") or []
            if weak:
                for item in weak:
                    score = next((s["proficiency"] for s in scores if s["concept"] == item), None)
                    st.write(
                        f"{topic_label(item)}"
                        + (f" · 掌握度{score:.0%}" if score is not None else "")
                    )
            else:
                st.caption(
                    "已评估知识点暂无低于60%的弱项。" if scores else "尚未评估，完成测验后再判断。"
                )
            st.caption("采用当前掌握度，不把已经改善的知识点继续标成弱项。")
    with right:
        with st.container(border=True):
            st.markdown("### 已学习知识点")
            learned = data.get("topics_learned") or []
            if learned:
                st.write("、".join(topic_label(x) for x in dict.fromkeys(learned)))
            else:
                st.caption("开始一轮教学后，这里会记录你学过的知识点。")
            st.caption("学过表示完成讲解，是否掌握还需要测验检验。")
    with st.container(border=True):
        rec = data.get("latest_recommendation")
        st.markdown("### 最新学习建议")
        if rec:
            st.markdown(f"**{topic_label(rec.get('next_topic'))}**")
            st.write(recommendation_text(rec, data.get("concept_scores")))
        else:
            st.write("先开始一次学习，系统会结合画像安排下一步。")
    memory = data.get("recent_memory_facts") or []
    with st.expander("最近的学习记忆", expanded=False):
        if memory:
            for fact in memory[-5:]:
                st.caption(MEMORY_LABELS.get(fact.get("type"), "学习记录"))
                st.write(display_memory(fact.get("content", "")))
        else:
            st.write("还没有长期学习记录。每次辅导和答题都会逐步完善画像。")


def render_documents(ready, rag_enabled):
    header("学习资料", "把自己的笔记整理在这里，在可用时辅助教学与问答。")
    if not ready:
        st.info("学习服务未连接，资料功能暂不可用。")
        return
    if not rag_enabled:
        with st.container(border=True):
            st.markdown("### 当前演示模式未开启资料检索")
            st.write(
                "你仍可以正常使用教学、练习和学习画像。资料上传需要启用文档检索能力，当前不会下载模型或自动开启检索。"
            )
            st.button("继续学习", on_click=navigate, args=("学习首页",))
        return
    st.caption("支持PDF、TXT和Markdown；单个文件不超过20MB。")
    with st.form("upload_notes"):
        uploaded = st.file_uploader("选择学习资料", type=["pdf", "txt", "md"])
        upload = st.form_submit_button("上传资料", type="primary")
    if upload and uploaded:
        try:
            with st.spinner("正在处理学习资料…"):
                result = get_client().upload_document(
                    st.session_state["student_id"], uploaded.name, uploaded.getvalue()
                )
            if result.get("status") == "ingested":
                st.success("资料已上传，后续提问可以结合笔记讲解。")
            else:
                st.error("资料暂未完成处理，请稍后重试。")
        except EduAgentClientError as exc:
            st.error(str(exc))
    try:
        documents = get_client().list_documents(st.session_state["student_id"])
    except EduAgentClientError as exc:
        st.error(str(exc))
        return
    if not documents:
        st.info("还没有上传资料，可以从一份简短笔记开始。")
    for doc in documents:
        with st.container(border=True):
            cols = st.columns([5, 2])
            cols[0].write(doc.get("filename", "学习资料"))
            cols[0].caption(
                {"ingested": "已就绪", "uploaded": "已上传", "failed": "处理失败"}.get(
                    doc.get("status"), "处理中"
                )
            )
            with cols[1]:
                with st.popover("移除资料"):
                    st.write("确认移除这份资料？移除后不再用于辅助讲解。")
                    if st.button("确认移除", key=f"delete_{doc['doc_id']}"):
                        try:
                            get_client().delete_document(doc["doc_id"])
                        except EduAgentClientError as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()


def main():
    st.set_page_config(
        page_title="CodeMentor｜AI 个性化编程学习助手",
        page_icon=":material/school:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session(st.session_state)
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
    ready, health, data = False, {}, None
    try:
        health = get_client().health()
        ready = health.get("status") == "ok"
        if ready:
            data = get_client().get_dashboard(st.session_state["student_id"])
    except EduAgentClientError as exc:
        error_message = str(exc)
    else:
        error_message = "学习服务未连接，请启动后端后重新连接。"
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-symbol">CM</span>'
            "<div><strong>CodeMentor</strong><small>AI 个性化编程学习助手</small></div></div>",
            unsafe_allow_html=True,
        )
        st.caption("学习工作台")
        st.radio(
            "功能导航",
            PAGES,
            key="page",
            label_visibility="collapsed",
            on_change=sidebar_navigation,
        )
        st.divider()
        st.markdown(f"**当前学生 · {st.session_state['profile_name']}**")
        st.caption("学习服务已连接" if ready else "学习服务未连接")
        if health.get("demo_mode"):
            st.caption("本地演示 · 无需模型密钥")
        st.button("重新连接", key="reconnect", width="stretch")
        with st.expander("切换学习档案"):
            st.text_input("档案名称", key="profile_input", max_chars=40)
            st.caption("同一名称对应同一份本地档案；这不是登录或身份验证。")
            st.button(
                "切换档案",
                on_click=change_profile,
                width="stretch",
            )
        st.markdown(
            '<div class="sidebar-note">选择知识点 → 开始学习<br>完成练习 → 查看进步</div>',
            unsafe_allow_html=True,
        )
    top = st.columns([4, 2])
    top[0].caption("CodeMentor · 你的编程学习伙伴")
    top[1].caption(
        f"当前学生：{st.session_state['profile_name']} · " + ("已连接" if ready else "未连接")
    )
    if not ready:
        st.warning(error_message)
        st.caption("请先启动学习服务，再点击侧栏“重新连接”。你的当前界面记录会保留。")
    elif data is None:
        st.warning("学习概览暂时无法读取，可以稍后重新连接。")
    if st.session_state.get("notice"):
        st.info(st.session_state.pop("notice"))
    page = st.session_state["page"]
    if page == "学习首页":
        render_home(data, ready)
    elif page == "AI 辅导":
        render_chat(data, ready)
    elif page == "专项练习":
        render_quiz(ready, health.get("demo_mode", False))
    elif page == "学习画像":
        render_dashboard(data)
    else:
        render_documents(ready, health.get("rag_enabled", False))
    if st.session_state.pop("scroll_to_top", False):
        # Fixed trusted code only. Never interpolate user, model or database text.
        st.html(
            "<script>requestAnimationFrame(() => requestAnimationFrame(() => {"
            "document.querySelector('[data-testid=stMain]')?.scrollTo({top:0,behavior:'instant'});"
            "}));</script>",
            unsafe_allow_javascript=True,
        )


if __name__ == "__main__":
    main()
