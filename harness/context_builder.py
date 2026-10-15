"""SQL-filtered, priority-trimmed context passed to the real agent prompts."""

import json
from functools import lru_cache

from sqlalchemy import case, func, or_, select

from config.curriculum import CURRICULUM, canonical_topic
from config.settings import HarnessSettings
from database.models import ConceptScoreRow, Conversation, MemoryEntry, Student, WrongAnswer
from database.models import Quiz as QuizRow
from harness.errors import ContextBudgetError


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Exact cl100k_base encoding count, not the configured model's billing count."""
    return len(_encoding().encode(text, disallowed_special=()))


def render_context(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ContextBuilder:
    def __init__(self, database, settings: HarnessSettings):
        self.db = database
        self.settings = settings

    def build(self, state: dict, node: str) -> dict:
        student_id = state["student_id"]
        topic = canonical_topic(state.get("topic") or state.get("user_input")) or (
            state.get("topic") or "general"
        )
        with self.db.session() as session:
            if state.get("quiz_id") and not state.get("topic"):
                quiz_topic = session.execute(
                    select(QuizRow.topic).where(
                        QuizRow.id == state["quiz_id"], QuizRow.student_id == student_id
                    )
                ).scalar_one_or_none()
                topic = canonical_topic(quiz_topic) or quiz_topic or topic
            profile = session.execute(
                select(
                    Student.name, Student.level, Student.learning_style, Student.updated_at
                ).where(Student.id == student_id)
            ).one()
            aliases = [topic.lower(), *CURRICULUM.get(topic, {}).get("aliases", ())]
            topic_scores = session.execute(
                select(ConceptScoreRow.proficiency, ConceptScoreRow.attempts)
                .where(ConceptScoreRow.student_id == student_id)
                .where(func.lower(ConceptScoreRow.concept).in_(aliases))
                .order_by(ConceptScoreRow.id.desc())
                .limit(1)
            ).all()
            related = func.lower(MemoryEntry.content).contains(topic.lower(), autoescape=True)
            memories = session.execute(
                select(
                    MemoryEntry.fact_type, func.substr(MemoryEntry.content, 1, 512).label("content")
                )
                .where(MemoryEntry.student_id == student_id)
                .where(or_(related, MemoryEntry.fact_type.in_(("preference", "profile"))))
                .order_by(case((related, 0), else_=1), MemoryEntry.id.desc())
                .limit(8)
            ).all()
            mistakes = session.execute(
                select(
                    WrongAnswer.question,
                    WrongAnswer.submitted,
                    WrongAnswer.explanation,
                    WrongAnswer.created_at,
                )
                .where(WrongAnswer.student_id == student_id)
                .where(func.lower(WrongAnswer.topic) == topic.lower())
                .order_by(WrongAnswer.id.desc())
                .limit(self.settings.recent_mistakes)
            ).all()
            records = session.execute(
                select(
                    Conversation.role, func.substr(Conversation.content, 1, 512).label("content")
                )
                .where(Conversation.student_id == student_id)
                .where(
                    or_(
                        Conversation.metadata_json["topic"].as_string() == topic,
                        func.lower(Conversation.content).contains(topic.lower(), autoescape=True),
                    )
                )
                .order_by(Conversation.id.desc())
                .limit(self.settings.recent_records)
            ).all()
        mastery = (
            {"score": topic_scores[0].proficiency, "attempts": topic_scores[0].attempts}
            if topic_scores
            else {"score": None, "attempts": 0}
        )
        payload = {
            "student_id": student_id,
            "topic": topic,
            "mastery": mastery,
            "goal": state.get("learning_goal") or "understand and practise the current topic",
            "profile": {
                "name": profile.name,
                "level": profile.level,
                "style": profile.learning_style,
            },
            "state": {
                "node": node,
                "quiz_id": state.get("quiz_id"),
                "latest_percent": (state.get("evaluation") or {}).get("percent"),
            },
            "mistakes": [dict(row._mapping) for row in mistakes],
            "memory": [dict(row._mapping) for row in memories],
            "recent": [dict(row._mapping) for row in records],
        }
        candidate_text = render_context(payload)
        before = count_tokens(candidate_text)
        removed = {"recent": 0, "memory": 0, "mistakes": 0}
        # Discard oldest optional items. Topic, mastery, goal and current state stay intact.
        for section in ("recent", "memory", "mistakes"):
            while (
                payload[section]
                and count_tokens(render_context(payload)) > self.settings.context_tokens
            ):
                payload[section].pop()
                removed[section] += 1
        text = render_context(payload)
        tokens = count_tokens(text)
        if tokens > self.settings.context_tokens:
            raise ContextBudgetError(
                "Required topic, mastery and goal exceed the context budget",
                detail={"required_tokens": tokens, "budget": self.settings.context_tokens},
            )
        return {
            "payload": payload,
            "text": text,
            "candidate_text": candidate_text,  # inspection only; never added to graph/LLM state
            "metrics": {
                "tokenizer": "cl100k_base",
                "candidate_tokens": before,
                "context_tokens": tokens,
                "budget": self.settings.context_tokens,
                "removed": removed,
            },
        }
