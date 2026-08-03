"""Unit tests for each agent node using the deterministic demo LLM."""

from __future__ import annotations

import pytest

from agents.evaluation import EvaluationAgent, _is_correct
from agents.memory import MemoryAgent
from agents.planner import PlannerAgent, PlannerOutput
from agents.quiz import QuizAgent
from agents.recommendation import RecommendationAgent
from agents.teacher import TeacherAgent
from config.constants import Routes
from models.enums import ChatIntent, QuestionType
from models.schemas import Quiz, QuizQuestion


class TestPlannerAgent:
    async def test_routes_practice_to_quiz(self, ctx):
        agent = PlannerAgent(ctx)
        out = await agent.run(
            {"student_id": "s1", "user_input": "give me a quiz on fractions", "messages": []}
        )
        assert out["next_agent"] == Routes.QUIZ
        assert out["intent"] == ChatIntent.PRACTICE.value

    async def test_routes_question_to_answer(self, ctx):
        agent = PlannerAgent(ctx)
        out = await agent.run(
            {"student_id": "s1", "user_input": "what is the capital of France?"}
        )
        assert out["next_agent"] == Routes.ANSWER_QUESTION

    async def test_routes_learn_to_teach(self, ctx):
        agent = PlannerAgent(ctx)
        out = await agent.run(
            {"student_id": "s1", "user_input": "teach me the water cycle"}
        )
        assert out["next_agent"] == Routes.TEACH

    def test_route_mapping(self):
        assert PlannerAgent.route(ChatIntent.PRACTICE) == Routes.QUIZ
        assert PlannerAgent.route(ChatIntent.ANSWER_QUESTION) == Routes.ANSWER_QUESTION
        assert PlannerAgent.route(ChatIntent.LEARN) == Routes.TEACH
        assert PlannerAgent.route(ChatIntent.GENERAL) == Routes.END
        assert PlannerAgent.route(ChatIntent.UPLOAD) == Routes.END

    def test_planner_output_model(self):
        out = PlannerOutput(intent=ChatIntent.LEARN, topic="Math", learning_goal="Basics")
        assert out.level.value == "beginner"


class TestTeacherAgent:
    async def test_run_returns_lesson_and_persists(self, ctx):
        agent = TeacherAgent(ctx)
        out = await agent.run(
            {
                "student_id": "s2",
                "topic": "Gravity",
                "user_input": "explain gravity",
                "messages": [],
            }
        )
        lesson = out["lesson"]
        assert lesson["topic"] == "Gravity"
        assert lesson["explanation"]
        assert any(m["role"] == "assistant" for m in out["messages"])
        # Lesson and topic history persisted.
        lessons = ctx.lesson_repo.list_lessons("s2")
        assert lessons and lessons[0].topic == "Gravity"
        assert "Gravity" in ctx.student_repo.get_or_create("s2").topics_learned


class TestQuizAgent:
    async def test_run_persists_quiz_with_canonical_ids(self, ctx):
        agent = QuizAgent(ctx)
        out = await agent.run(
            {"student_id": "s3", "topic": "Fractions", "user_input": "quiz me", "messages": []}
        )
        quiz_dict = out["quiz"]
        quiz = Quiz(**quiz_dict)
        assert quiz.id.startswith("quiz_")
        assert len(quiz.questions) >= 1
        # The persisted quiz must be reloadable with matching question ids so the
        # evaluation agent can grade a submission.
        reloaded = ctx.quiz_repo.get_quiz(quiz.id)
        assert reloaded is not None
        assert [q.id for q in reloaded.questions] == [
            f"{quiz.id}q{i}" for i in range(1, len(reloaded.questions) + 1)
        ]
        assert any(m["role"] == "assistant" for m in out["messages"])


class TestEvaluationAgent:
    def _seed_quiz(self, ctx) -> Quiz:
        ctx.student_repo.get_or_create("s4")
        quiz = Quiz(
            id="quiz_eval",
            topic="Fractions",
            difficulty="beginner",
            questions=[
                QuizQuestion(
                    id="q1", type=QuestionType.MCQ, topic="Fractions",
                    question="1/2 + 1/2?", options=["1", "2"], correct_answer="1",
                ),
                QuizQuestion(
                    id="q2", type=QuestionType.NUMERICAL, topic="Fractions",
                    question="1/4 + 1/4?", options=[], correct_answer="0.5",
                ),
                QuizQuestion(
                    id="q3", type=QuestionType.SUBJECTIVE, topic="Fractions",
                    question="Explain fractions?", options=[], correct_answer="Model answer",
                ),
            ],
        )
        ctx.quiz_repo.create_quiz(quiz, student_id="s4")
        return quiz

    async def test_is_correct(self, ctx):
        from models.schemas import QuizQuestion as Q

        mcq = Q(id="a", type=QuestionType.MCQ, topic="t", question="?", correct_answer="Paris")
        assert _is_correct(mcq, "Paris")
        assert _is_correct(mcq, "paris")  # case-insensitive
        assert not _is_correct(mcq, "London")
        assert not _is_correct(mcq, None)

        num = Q(id="b", type=QuestionType.NUMERICAL, topic="t", question="?", correct_answer="5")
        assert _is_correct(num, "5.0")
        assert not _is_correct(num, "abc")

    async def test_run_scores_objective_and_persists(self, ctx):
        quiz = self._seed_quiz(ctx)
        agent = EvaluationAgent(ctx)
        # q1 correct, q2 wrong, q3 answered (demo LLM grades 0.8).
        out = await agent.run(
            {
                "student_id": "s4",
                "quiz_id": quiz.id,
                "answers": [
                    {"question_id": f"{quiz.id}q1", "answer": "1"},
                    {"question_id": f"{quiz.id}q2", "answer": "99"},
                    {"question_id": f"{quiz.id}q3", "answer": "Fractions are parts of a whole."},
                ],
            }
        )
        evaluation = out["evaluation"]
        assert evaluation["max_score"] == quiz.total_points
        assert 0.0 <= evaluation["percent"] <= 100.0
        # q2 wrong -> Fractions lands in weak concepts.
        assert "Fractions" in evaluation["weak_concepts"]
        # Attempt and progress point persisted.
        assert ctx.quiz_repo.list_attempts("s4")
        assert ctx.progress_repo.latest_score("s4") is not None

    async def test_run_raises_for_unknown_quiz(self, ctx):
        agent = EvaluationAgent(ctx)
        from utils.errors import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await agent.run({"student_id": "s4", "quiz_id": "ghost"})


class TestMemoryAgent:
    async def test_noop_without_content(self, ctx):
        agent = MemoryAgent(ctx)
        out = await agent.run({"student_id": "s5", "messages": []})
        assert out["memory_updates"] == []

    async def test_extracts_and_persists_facts(self, ctx):
        agent = MemoryAgent(ctx)
        out = await agent.run(
            {
                "student_id": "s5",
                "user_input": "I love learning biology",
                "messages": [{"role": "assistant", "content": "Great! Let's study cells.", "node": "teacher"}],
            }
        )
        assert out["memory_updates"]
        facts = ctx.memory_repo.list_facts("s5")
        assert len(facts) >= 1
        assert ctx.conversation_repo.count("s5") >= 2  # user + assistant


class TestRecommendationAgent:
    async def test_run_returns_and_persists(self, ctx):
        agent = RecommendationAgent(ctx)
        out = await agent.run({"student_id": "s6", "messages": []})
        rec = out["recommendations"]
        assert rec["next_topic"]
        assert any(m["role"] == "assistant" for m in out["messages"])
        latest = ctx.memory_repo.latest_recommendation("s6")
        assert latest is not None and latest.next_topic == rec["next_topic"]
