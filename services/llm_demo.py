"""Deterministic offline LLM for the ``demo`` provider.

The demo client implements the same interface as :class:`LLMClient` but never
touches the network. It inspects the requested Pydantic schema and returns
plausible, topic-aware content so the full tutoring pipeline (graph, RAG,
persistence, dashboard) can be exercised end-to-end without an API key -- ideal
for first-run demos, CI and offline development.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from config.constants import QuizComposition
from config.curriculum import CURRICULUM, canonical_topic
from config.presentation import topic_label
from models.enums import ChatIntent, DifficultyLevel, QuestionType, StudentLevel
from services.demo_content_zh import CONTENT_ZH
from services.demo_quiz_variants_zh import MCQ_VARIANTS, NUMERICAL_VARIANTS
from utils.errors import LLMOutputParsingError

T = TypeVar("T", bound=BaseModel)

_GENERIC_REPLIES = [
    "That's a great question. In EduAgent, I break topics into simple ideas "
    "and build your understanding step by step.",
    "Let's learn that together. I'll start with the fundamentals, add examples, "
    "and then we can quiz what you've learned.",
]


class DemoLLMClient:
    """Schema-driven deterministic responses (never calls an external API)."""

    provider = "demo"

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    # ------------------------------------------------------------ generate --
    async def generate(self, system: str, user: str, *, temperature: float | None = None) -> str:
        return "你好，我是CodeMentor。当前为本地演示模式，可以先选择一个编程知识点，体验讲解、练习与学习画像。"

    # -------------------------------------------------------- generate_json --
    async def generate_json(
        self,
        system: str,
        user: str,
        *,
        model_class: type[T],
        temperature: float | None = None,
    ) -> T:
        builder = getattr(self, f"_demo{model_class.__name__.lower()}", None)
        if builder is None:
            raise LLMOutputParsingError(
                f"Demo provider has no generator for {model_class.__name__}"
            )
        try:
            return model_class.model_validate(builder(user))
        except ValidationError as exc:  # pragma: no cover - defensive
            raise LLMOutputParsingError(
                f"Demo output failed {model_class.__name__} validation: {exc}"
            ) from exc

    # -------------------------------------------------------------- helpers --
    @staticmethod
    def _first_sentence(text: str) -> str:
        """Return the first meaningful sentence of ``text``.

        Prompt-structure labels (``Student message:``, ``Assistant response:``,
        etc.) are skipped and the text is collapsed to a single line so a stored
        memory summary never embeds prompt scaffolding or multi-line fragments
        that later heuristics would misread as real content.
        """
        stripped = text.strip()
        if not stripped:
            return "the topic"
        for raw_line in stripped.splitlines():
            ln = raw_line.strip()
            if not ln:
                continue
            if re.match(r"^[A-Za-z ]+:$", ln):
                continue  # structural label, e.g. "Student message:"
            return ln.split(".", 1)[0].strip() or "the topic"
        return "the topic"

    @staticmethod
    def _prefix_value(lines: list[str], prefix: str) -> str:
        """Return the value after a ``prefix:`` line, or an empty string."""
        for ln in lines:
            if ln.lower().startswith(prefix):
                return ln.split(":", 1)[1].strip().strip("\"' —:-")
        return ""

    @staticmethod
    def _student_message(text: str) -> str | None:
        """Extract the student's raw message from an assembled prompt.

        Prompts are built as ``Student message:\\n<body>`` followed by profile
        and memory context blocks. Keyword heuristics must run against the
        *body* only — the context can contain words like "quiz" from history
        and would otherwise skew classification.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln.lower().startswith("student message:"):
                if i + 1 < len(lines):
                    return lines[i + 1]
                return None
        return None

    def _topic(self, text: str) -> str:
        """Extract a short topic string from the full assembled prompt.

        Strategy:
        1. Look for an explicit ``Topic: X`` line (embedded by teacher/quiz/evaluation).
        2. Look for the ``Student message:`` body (planner).
        3. Look for a ``Weak concepts: X`` line (recommendation).
        4. Fall back to the last non-trivial line.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # 1) Explicit "Topic: X" line
        topic = self._prefix_value(lines, "topic:")
        if topic:
            return canonical_topic(topic) or topic[:60]
        # 2) The student-message body (line after "Student message:")
        body = self._student_message(text)
        if body:
            return canonical_topic(body) or body[:60]
        for line in lines:
            if line.startswith('{"student_id":'):
                return json.loads(line).get("topic", "general")
        # 3) "Weak concepts:" line value (recommendation context) — the value is
        #    "none" / "none identified" when there are no weak areas, so any
        #    value starting with "none" must be treated as absent.
        weak = self._prefix_value(lines, "weak concepts:")
        if weak and not weak.lower().startswith("none"):
            return weak.split(",")[0].strip()[:60]
        # 4) Fallback: last non-trivial line
        if not lines:
            return "general"
        candidate = lines[-1].strip("\"' —:-")
        candidate = re.sub(r"\s{2,}", " ", candidate)
        if len(candidate) > 60:
            candidate = candidate[:60].rsplit(" ", 1)[0]
        return candidate or "general"

    def _intent(self, user: str) -> ChatIntent:
        message = (self._student_message(user) or user).lower()
        if any(
            w in message
            for w in ("quiz", "test me", "practice", "exercise", "mcq", "练习", "测验", "出题")
        ):
            return ChatIntent.PRACTICE
        if any(
            w in message
            for w in (
                "?",
                "what is",
                "explain",
                "why",
                "how",
                "answer",
                "？",
                "为什么",
                "如何",
                "怎么",
            )
        ):
            return ChatIntent.ANSWER_QUESTION
        return ChatIntent.LEARN

    # ------------------------------------------------------------ builders --
    def _demoplanneroutput(self, user: str) -> dict:
        topic = self._topic(user)
        intent = self._intent(user)
        return {
            "intent": intent.value,
            "topic": topic,
            "learning_goal": f"理解{topic_label(topic)}的基础概念并通过练习检查。",
            "level": StudentLevel.BEGINNER.value,
            "prerequisites": [],
        }

    def _demoexplanation(self, user: str) -> dict:
        topic = self._topic(user)
        if topic in CURRICULUM:
            item = CONTENT_ZH[topic]
            return {
                "topic": topic,
                "summary": item["summary"],
                "explanation": item["explanation"],
                "examples": [
                    CURRICULUM[topic]["example"].replace(
                        "# a becomes Fibonacci(n)", "# a是Fibonacci(n)"
                    )
                ],
                "analogies": [item["analogy"]],
                "recap": item["summary"],
            }
        return {
            "topic": topic,
            "summary": f"一起从基础理解{topic_label(topic)}。",
            "explanation": (
                f"学习{topic_label(topic)}时，可以先把问题拆成几个小步骤，理解每一步的作用，再用具体例子检查。当前演示内容用于体验学习流程。"
            ),
            "examples": [
                f"先用一个简单例子说明{topic_label(topic)}。",
                "再逐步解释例子中的每一个操作。",
            ],
            "analogies": [
                "像学习骑车一样，先把基础动作做好，再增加难度。",
            ],
            "recap": ("记住核心概念，自己尝试示例，再通过一份小练习检查理解。"),
        }

    def _demoquiz(self, user: str) -> dict:
        topic = self._topic(user)
        questions: list[dict] = []
        qindex = 0

        def question(qtype: QuestionType) -> dict:
            nonlocal qindex
            qindex += 1
            base = {
                "id": f"demoq{qindex}",
                "type": qtype.value,
                "topic": topic,
                "points": 1,
                "explanation": f"请结合{topic_label(topic)}的基础概念复习。",
            }
            if qtype == QuestionType.MCQ:
                base.update(
                    {
                        "question": f"关于{topic_label(topic)}，哪一项说法正确？",
                        "options": [
                            "先理解基础概念，再结合例子应用",
                            "不需要任何基础概念",
                            "只背结论就足够",
                            "完全不需要检查结果",
                        ],
                        "correct_answer": "先理解基础概念，再结合例子应用",
                    }
                )
            elif qtype == QuestionType.NUMERICAL:
                base.update(
                    {
                        "question": ("计算2 + 3的结果。"),
                        "options": [],
                        "correct_answer": "5",
                    }
                )
            else:
                base.update(
                    {
                        "question": (f"用自己的话解释{topic_label(topic)}的一个核心概念，并举例。"),
                        "options": [],
                        "correct_answer": ("回答应包括核心概念及一个具体例子。"),
                    }
                )
            if topic in CURRICULUM:
                item = CONTENT_ZH[topic]
                base["explanation"] = item["summary"]
                if qtype == QuestionType.MCQ:
                    base.update(
                        question=item["mcq"],
                        options=item["options"],
                        correct_answer=item["options"][0],
                    )
                elif qtype == QuestionType.NUMERICAL:
                    base.update(
                        question=item["numerical"], correct_answer=CURRICULUM[topic]["answer"]
                    )
                else:
                    base.update(
                        question=f"解释{topic_label(topic)}，并写一个简短代码示例。",
                        correct_answer=item["summary"],
                    )
            if topic in MCQ_VARIANTS:
                if qtype == QuestionType.MCQ and qindex > 1:
                    prompt, options, correct, explanation = MCQ_VARIANTS[topic][qindex - 2]
                    base.update(
                        question=prompt,
                        options=options,
                        correct_answer=correct,
                        explanation=explanation,
                    )
                elif qtype == QuestionType.NUMERICAL and qindex == 7:
                    prompt, correct, explanation = NUMERICAL_VARIANTS[topic]
                    base.update(question=prompt, correct_answer=correct, explanation=explanation)
            return base

        for _ in range(QuizComposition.MCQ_COUNT):
            questions.append(question(QuestionType.MCQ))
        for _ in range(QuizComposition.NUMERICAL_COUNT):
            questions.append(question(QuestionType.NUMERICAL))
        questions.append(question(QuestionType.SUBJECTIVE))

        return {
            "id": "demo_quiz",
            "topic": topic,
            "difficulty": DifficultyLevel.BEGINNER.value,
            "questions": questions,
        }

    def _demosubjectivegrade(self, user: str) -> dict:
        topic = self._topic(user)
        return {
            "score": 0.8,
            "feedback": (
                "已收到你的回答。演示模式对非空主观答案固定评分为0.8，请对照参考答案补充核心概念和代码例子；这不是对真实答案质量的判断。"
            ),
            "weak_concepts": [topic],
            "revision_topics": [topic],
            "next_lesson": f"尝试把{topic_label(topic)}用于一个具体例子。",
        }

    def _demomemoryextraction(self, user: str) -> dict:
        topic = self._topic(user)
        return {
            "facts": [
                {
                    "fact_type": "concept",
                    "content": f"[topic:{topic}] 学习了{topic_label(topic)}。",
                    "importance": 0.7,
                },
                {
                    "fact_type": "conversation_summary",
                    "content": self._first_sentence(user),
                    "importance": 0.5,
                },
            ]
        }

    def _demorecommendation(self, user: str) -> dict:
        topic = self._topic(user)
        return {
            "next_topic": topic,
            "rationale": (f"你已经开始学习{topic_label(topic)}，可以通过一个具体例子继续巩固。"),
            "revision_schedule": [
                f"第1天：复习{topic_label(topic)}的要点和示例",
                "第3天：完成一次针对性练习",
                "第7天：用一个应用问题检查理解",
            ],
            "daily_plan": [
                f"复习{topic_label(topic)}15分钟",
                "完成一个代码示例",
                "用自己的话总结核心概念",
            ],
            "weak_areas": [topic],
            "suggested_quiz_topics": [topic],
        }

    @property
    def settings(self) -> Any:
        return self._settings
