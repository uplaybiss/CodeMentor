"""End-to-end graph execution tests (tutoring + evaluation)."""

from __future__ import annotations

from graphs.workflow import build_evaluation_graph, build_tutoring_graph
from models.enums import DifficultyLevel, QuestionType
from models.schemas import Quiz, QuizQuestion


def _initial_tutoring_state(student_id: str, message: str, *, with_quiz: bool = False) -> dict:
    return {
        "student_id": student_id,
        "user_input": message,
        "topic": None,
        "with_quiz": with_quiz,
        "intent": "",
        "learning_goal": "",
        "level": "",
        "messages": [],
        "memory_updates": [],
        "errors": [],
        "lesson": None,
        "quiz": None,
        "evaluation": None,
        "recommendations": None,
        "rag_context": None,
        "rag_sources": [],
        "next_agent": "",
    }


# --------------------------------------------------------------------------- #
# Tutoring graph
# --------------------------------------------------------------------------- #
class TestTutoringGraph:
    async def test_learn_path(self, ctx):
        graph = build_tutoring_graph(ctx)
        state = _initial_tutoring_state("g_s1", "teach me about photosynthesis")
        final = await graph.ainvoke(state)
        assert final["intent"] == "learn"
        assert final["lesson"] is not None
        assert final.get("recommendations") is not None
        assert final.get("messages")

    async def test_question_path(self, ctx):
        graph = build_tutoring_graph(ctx)
        state = _initial_tutoring_state("g_s2", "what is the capital of France?")
        final = await graph.ainvoke(state)
        # ANSWER_QUESTION goes rag → teacher → memory → recommendation
        assert final["intent"] == "answer_question"
        assert final.get("lesson") is not None

    async def test_practice_path(self, ctx):
        graph = build_tutoring_graph(ctx)
        state = _initial_tutoring_state("g_s3", "give me a quiz on fractions", with_quiz=False)
        final = await graph.ainvoke(state)
        assert final["intent"] == "practice"
        assert final.get("quiz") is not None

    async def test_with_quiz_flag(self, ctx):
        graph = build_tutoring_graph(ctx)
        state = _initial_tutoring_state("g_s4", "teach me algebra basics", with_quiz=True)
        final = await graph.ainvoke(state)
        # teacher → quiz → memory → recommendation
        assert final.get("lesson") is not None
        assert final.get("quiz") is not None


# --------------------------------------------------------------------------- #
# Evaluation graph
# --------------------------------------------------------------------------- #
class TestEvaluationGraph:
    def _seed_quiz(self, ctx) -> Quiz:
        ctx.student_repo.get_or_create("eg_s1")
        quiz = Quiz(
            id="quiz_eg",
            topic="Algebra",
            difficulty=DifficultyLevel.BEGINNER,
            questions=[
                QuizQuestion(
                    id="q1", type=QuestionType.MCQ, topic="Algebra",
                    question="x + 2 = 5, x = ?", options=["2", "3"], correct_answer="3",
                ),
                QuizQuestion(
                    id="q2", type=QuestionType.NUMERICAL, topic="Algebra",
                    question="2 * 4 = ?", options=[], correct_answer="8",
                ),
                QuizQuestion(
                    id="q3", type=QuestionType.SUBJECTIVE, topic="Algebra",
                    question="Explain what a variable is.", options=[], correct_answer="A symbol.",
                ),
            ],
        )
        ctx.quiz_repo.create_quiz(quiz, student_id="eg_s1")
        return quiz

    async def test_full_evaluation_pipeline(self, ctx):
        quiz = self._seed_quiz(ctx)
        graph = build_evaluation_graph(ctx)
        final = await graph.ainvoke(
            {
                "student_id": "eg_s1",
                "quiz_id": quiz.id,
                "answers": [
                    {"question_id": f"{quiz.id}q1", "answer": "3"},
                    {"question_id": f"{quiz.id}q2", "answer": "8"},
                    {"question_id": f"{quiz.id}q3", "answer": "A variable represents a value."},
                ],
                "messages": [],
                "memory_updates": [],
                "errors": [],
            }
        )
        evaluation = final.get("evaluation")
        assert evaluation is not None
        assert evaluation["percent"] > 0
        assert final.get("recommendations") is not None
        # History persisted.
        assert ctx.progress_repo.latest_score("eg_s1") is not None
