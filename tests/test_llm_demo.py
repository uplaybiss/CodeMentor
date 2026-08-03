"""Tests for the offline demo LLM client."""

from __future__ import annotations

import pytest

from config.settings import LLMSettings
from models.enums import ChatIntent, QuestionType
from models.schemas import Explanation, Quiz, Recommendation
from services.llm_demo import DemoLLMClient
from utils.errors import LLMOutputParsingError

# Reuse the demo schema builders' expectations: the demo client validates output
# against the requested Pydantic model, so we import the concrete models too.


@pytest.fixture
def client() -> DemoLLMClient:
    return DemoLLMClient(LLMSettings(provider="demo"))


class TestGenerate:
    async def test_generate_returns_text(self, client):
        text = await client.generate("system", "Student: What is gravity?")
        assert isinstance(text, str)
        assert text

    async def test_generate_mentions_demo_mode(self, client):
        text = await client.generate("s", "Hello")
        assert "demo mode" in text


class TestPlanner:
    async def test_planner_output(self, client):
        from agents.planner import PlannerOutput

        out = await client.generate_json(
            "system", "Topic: Physics\nStudent message: teach me physics", model_class=PlannerOutput
        )
        assert out.topic
        assert out.intent in ChatIntent

    def test_intent_quiz(self, client):
        assert client._intent("give me a quiz on fractions") == ChatIntent.PRACTICE

    def test_intent_question(self, client):
        assert client._intent("what is the capital of France?") == ChatIntent.ANSWER_QUESTION

    def test_intent_learn(self, client):
        assert client._intent("tell me about the water cycle") == ChatIntent.LEARN

    def test_intent_ignores_memory_context(self, client):
        # The assembled prompt carries profile + memory context that can contain
        # the word "quiz". Classification must use the student message body only,
        # otherwise every later turn gets misrouted to PRACTICE.
        prompt = (
            "Student message:\nwhat does photosynthesis do?\n\n"
            "Student profile:\n- level: beginner\n- topics learned: photosynthesis\n\n"
            "Memory:\n- (conversation_summary) Student message:\nQuiz result: 97\n"
        )
        assert client._intent(prompt) == ChatIntent.ANSWER_QUESTION

    def test_intent_practice_from_message_not_context(self, client):
        prompt = (
            "Student message:\ngive me a practice quiz on fractions\n\n"
            "Memory:\n- (concept) learnt fractions\n"
        )
        assert client._intent(prompt) == ChatIntent.PRACTICE


class TestTopicExtraction:
    def test_explicit_topic_line(self, client):
        text = "Topic: Photosynthesis\nRequested difficulty: beginner\nStudent level: beginner"
        assert client._topic(text) == "Photosynthesis"

    def test_student_message_body(self, client):
        text = "Student message:\nHow do rockets work?"
        assert client._topic(text) == "How do rockets work?"

    def test_weak_concepts_line(self, client):
        text = "Weak concepts: Algebra, Fractions\nNext: whatever"
        assert client._topic(text) == "Algebra"

    def test_fallback_last_line(self, client):
        assert client._topic("just some text about mitochondria") == "just some text about mitochondria"

    def test_empty(self, client):
        assert client._topic("") == "general"


class TestExplanation:
    async def test_builds_valid_explanation(self, client):
        expl = await client.generate_json(
            "s", "Topic: Gravity", model_class=Explanation
        )
        assert expl.topic == "Gravity"
        assert expl.explanation
        assert expl.examples
        assert expl.analogies


class TestQuiz:
    async def test_builds_valid_quiz_with_composition(self, client):
        from config.constants import QuizComposition

        quiz = await client.generate_json("s", "Topic: Fractions", model_class=Quiz)
        assert len(quiz.questions) == QuizComposition.TOTAL
        types = [q.type for q in quiz.questions]
        assert types.count(QuestionType.MCQ) == QuizComposition.MCQ_COUNT
        assert types.count(QuestionType.NUMERICAL) == QuizComposition.NUMERICAL_COUNT
        assert types.count(QuestionType.SUBJECTIVE) == QuizComposition.SUBJECTIVE_COUNT

    async def test_mcq_has_options_and_answer(self, client):
        quiz = await client.generate_json("s", "Topic: History", model_class=Quiz)
        mcq = next(q for q in quiz.questions if q.type == QuestionType.MCQ)
        assert len(mcq.options) >= 2
        assert mcq.correct_answer in mcq.options


class TestSubjectiveGrade:
    async def test_builds_valid_grade(self, client):
        from agents.evaluation import SubjectiveGrade

        grade = await client.generate_json(
            "s", "Topic: Biology", model_class=SubjectiveGrade
        )
        assert 0.0 <= grade.score <= 1.0
        assert grade.feedback


class TestMemoryExtraction:
    async def test_builds_facts(self, client):
        from agents.memory import MemoryExtraction

        out = await client.generate_json(
            "s", "Student message: I like biology", model_class=MemoryExtraction
        )
        assert out.facts


class TestRecommendation:
    async def test_builds_valid_recommendation(self, client):
        rec = await client.generate_json(
            "s", "Weak concepts: none", model_class=Recommendation
        )
        assert rec.next_topic
        assert rec.revision_schedule


class TestUnsupportedModel:
    async def test_unknown_model_raises(self, client):
        from pydantic import BaseModel

        class Weird(BaseModel):
            x: int

        with pytest.raises(LLMOutputParsingError):
            await client.generate_json("s", "whatever", model_class=Weird)
