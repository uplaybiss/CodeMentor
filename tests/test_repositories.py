"""CRUD tests for every SQLAlchemy repository (real SQLite, temp file)."""

from __future__ import annotations

import pytest

from models.enums import DifficultyLevel, QuestionType, StudentLevel
from models.schemas import (
    EvaluationResult,
    Explanation,
    Quiz,
    QuizQuestion,
    Recommendation,
)
from utils.errors import ResourceNotFoundError


def _seed_student(container, student_id: str):
    """Create the student row so FK-dependent writes succeed (matching how the
    agents/routers always call ``get_or_create`` before writing)."""
    container.student_repo.get_or_create(student_id)


# --------------------------------------------------------------------------- #
# Student repository
# --------------------------------------------------------------------------- #
class TestStudentRepository:
    def test_create_and_get(self, container):
        repo = container.student_repo
        profile = repo.create(student_id="s1", name="Ada")
        assert profile.student_id == "s1"
        assert profile.name == "Ada"
        fetched = repo.get("s1")
        assert fetched is not None and fetched.student_id == "s1"

    def test_get_or_create_is_idempotent(self, container):
        repo = container.student_repo
        first = repo.get_or_create("s2")
        second = repo.get_or_create("s2")
        assert first.student_id == second.student_id

    def test_update_profile(self, container):
        repo = container.student_repo
        repo.create(student_id="s3")
        updated = repo.update_profile("s3", name="Grace", level="intermediate")
        assert updated.name == "Grace"
        assert updated.level == StudentLevel.INTERMEDIATE

    def test_update_missing_raises(self, container):
        with pytest.raises(ResourceNotFoundError):
            container.student_repo.update_profile("nope", name="X")

    def test_record_quiz_score_appends(self, container):
        repo = container.student_repo
        repo.create(student_id="s4")
        repo.record_quiz_score("s4", 75.0)
        repo.record_quiz_score("s4", 90.0)
        assert repo.get("s4").quiz_scores == [75.0, 90.0]

    def test_add_weak_concept_dedupes(self, container):
        repo = container.student_repo
        repo.create(student_id="s5")
        repo.add_weak_concept("s5", "Algebra")
        repo.add_weak_concept("s5", "Algebra")
        assert repo.get("s5").weak_concepts == ["Algebra"]

    def test_add_topic_learned_and_list(self, container):
        repo = container.student_repo
        repo.create(student_id="s6")
        repo.add_topic_learned("s6", "Biology")
        repo.add_topic_learned("s6", "Chemistry")
        assert repo.get("s6").topics_learned == ["Biology", "Chemistry"]
        # list_topics reads the topics table, newest first.
        assert repo.list_topics("s6") == ["Chemistry", "Biology"]

    def test_concept_scores_ema(self, container):
        repo = container.student_repo
        repo.create(student_id="s7")
        first = repo.upsert_concept_score("s7", "Fractions", 0.5)
        assert first.attempts == 1 and first.proficiency == 0.5
        second = repo.upsert_concept_score("s7", "Fractions", 1.0)
        assert second.attempts == 2
        assert second.proficiency == pytest.approx(0.8, abs=0.001)  # 0.5*(1-0.6)+1.0*0.6

    def test_weakest_concepts(self, container):
        repo = container.student_repo
        repo.create(student_id="s8")
        repo.upsert_concept_score("s8", "Hard", 0.2)
        repo.upsert_concept_score("s8", "Easy", 0.9)
        assert repo.weakest_concepts("s8") == ["Hard"]


# --------------------------------------------------------------------------- #
# Quiz repository
# --------------------------------------------------------------------------- #
def _make_quiz(quiz_id: str = "quiz_1") -> Quiz:
    return Quiz(
        id=quiz_id,
        topic="Fractions",
        difficulty=DifficultyLevel.BEGINNER,
        questions=[
            QuizQuestion(
                id="q1",
                type=QuestionType.MCQ,
                topic="Fractions",
                question="1/2 + 1/2 = ?",
                options=["1", "2", "3"],
                correct_answer="1",
            ),
            QuizQuestion(
                id="q2",
                type=QuestionType.NUMERICAL,
                topic="Fractions",
                question="1/4 + 1/4 = ?",
                options=[],
                correct_answer="0.5",
            ),
        ],
    )


class TestQuizRepository:
    def test_create_and_get_roundtrip(self, container):
        _seed_student(container, "s10")
        repo = container.quiz_repo
        repo.create_quiz(_make_quiz(), student_id="s10")
        quiz = repo.get_quiz("quiz_1")
        assert quiz is not None
        assert quiz.topic == "Fractions"
        assert len(quiz.questions) == 2
        # Repository reconstructs canonical question ids {quiz_id}q{idx}.
        assert [q.id for q in quiz.questions] == ["quiz_1q1", "quiz_1q2"]
        assert quiz.questions[0].correct_answer == "1"

    def test_get_missing_returns_none(self, container):
        assert container.quiz_repo.get_quiz("missing") is None

    def test_list_quizzes_newest_first(self, container):
        _seed_student(container, "s11")
        repo = container.quiz_repo
        repo.create_quiz(_make_quiz("q_old"), student_id="s11")
        repo.create_quiz(_make_quiz("q_new"), student_id="s11")
        ids = repo.list_quizzes("s11")
        assert ids == ["q_new", "q_old"]

    def test_save_and_get_attempt(self, container):
        _seed_student(container, "s12")
        repo = container.quiz_repo
        repo.create_quiz(_make_quiz("quiz_a"), student_id="s12")
        evaluation = EvaluationResult(
            score=1.0,
            max_score=2.0,
            percent=50.0,
            weak_concepts=["Fractions"],
            revision_topics=["Fractions"],
            next_lesson="Revise Fractions",
            feedback="Keep practising.",
        )
        repo.save_attempt(
            attempt_id="attempt_1",
            student_id="s12",
            quiz_id="quiz_a",
            answers={"q1": "1"},
            evaluation=evaluation,
        )
        attempt = repo.get_attempt("attempt_1")
        assert attempt is not None
        assert attempt["percent"] == 50.0
        assert attempt["weak_concepts"] == ["Fractions"]
        attempts = repo.list_attempts("s12")
        assert len(attempts) == 1

    def test_get_attempt_or_404(self, container):
        with pytest.raises(ResourceNotFoundError):
            container.quiz_repo.get_attempt_or_404("nope")


# --------------------------------------------------------------------------- #
# Conversation repository
# --------------------------------------------------------------------------- #
class TestConversationRepository:
    def test_add_and_list(self, container):
        _seed_student(container, "s20")
        repo = container.conversation_repo
        repo.add_message("s20", "user", "Hi")
        repo.add_message("s20", "assistant", "Hello!")
        messages = repo.list_messages("s20")
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert repo.count("s20") == 2


# --------------------------------------------------------------------------- #
# Lesson repository
# --------------------------------------------------------------------------- #
class TestLessonRepository:
    def test_save_and_list(self, container):
        _seed_student(container, "s30")
        repo = container.lesson_repo
        lesson = Explanation(
            topic="Gravity",
            summary="A short summary",
            explanation="The full explanation.",
            examples=["Example"],
            analogies=["Analogy"],
            recap="Recap here.",
        )
        repo.save_lesson("s30", lesson)
        lessons = repo.list_lessons("s30")
        assert len(lessons) == 1
        assert lessons[0].topic == "Gravity"
        assert lessons[0].examples == ["Example"]


# --------------------------------------------------------------------------- #
# Memory repository
# --------------------------------------------------------------------------- #
class TestMemoryRepository:
    def test_add_and_list_facts(self, container):
        _seed_student(container, "s40")
        repo = container.memory_repo
        repo.add_fact("s40", "concept", "Likes biology", 0.8)
        repo.add_fact("s40", "concept", "Weak at algebra", 0.6)
        facts = repo.list_facts("s40")
        assert len(facts) == 2
        assert all(f["fact_type"] == "concept" for f in facts)

    def test_list_facts_filtered(self, container):
        _seed_student(container, "s41")
        repo = container.memory_repo
        repo.add_fact("s41", "concept", "A")
        repo.add_fact("s41", "preference", "B")
        facts = repo.list_facts("s41", fact_type="preference")
        assert [f["content"] for f in facts] == ["B"]

    def test_save_and_latest_recommendation(self, container):
        _seed_student(container, "s42")
        repo = container.memory_repo
        rec = Recommendation(
            next_topic="Calculus",
            rationale="Next logical step.",
            revision_schedule=["Day 1"],
            daily_plan=["Practice"],
            weak_areas=["Algebra"],
            suggested_quiz_topics=["Calculus"],
        )
        repo.save_recommendation("s42", rec)
        latest = repo.latest_recommendation("s42")
        assert latest is not None
        assert latest.next_topic == "Calculus"
        assert repo.latest_recommendation("nobody") is None


# --------------------------------------------------------------------------- #
# Document repository
# --------------------------------------------------------------------------- #
class TestDocumentRepository:
    def test_create_get_list_count(self, container):
        _seed_student(container, "s50")
        repo = container.document_repo
        repo.create(document_id="doc_1", student_id="s50", filename="notes.pdf")
        record = repo.get("doc_1")
        assert record is not None and record.filename == "notes.pdf"
        assert repo.count_for_student("s50") == 1
        listed = repo.list_for_student("s50")
        assert [r.id for r in listed] == ["doc_1"]

    def test_update_status(self, container):
        _seed_student(container, "s51")
        repo = container.document_repo
        repo.create(document_id="doc_2", student_id="s51", filename="a.md")
        updated = repo.update_status("doc_2", "ready", chunk_count=5)
        assert updated.status == "ready"
        assert updated.chunk_count == 5

    def test_delete(self, container):
        _seed_student(container, "s52")
        repo = container.document_repo
        repo.create(document_id="doc_3", student_id="s52", filename="b.txt")
        repo.delete("doc_3")
        assert repo.get("doc_3") is None

    def test_get_or_404(self, container):
        with pytest.raises(ResourceNotFoundError):
            container.document_repo.get_or_404("missing")


# --------------------------------------------------------------------------- #
# Progress repository
# --------------------------------------------------------------------------- #
class TestProgressRepository:
    def test_add_and_list(self, container):
        _seed_student(container, "s60")
        repo = container.progress_repo
        repo.add_snapshot("s60", "Biology", 80.0)
        repo.add_snapshot("s60", "Biology", 90.0)
        snapshots = repo.list_snapshots("s60")
        assert len(snapshots) == 2
        assert snapshots[0].score == 80.0
        assert repo.count("s60") == 2

    def test_latest_score(self, container):
        _seed_student(container, "s61")
        repo = container.progress_repo
        repo.add_snapshot("s61", "Chemistry", 70.0)
        latest = repo.latest_score("s61")
        assert latest is not None and latest.score == 70.0
        assert repo.latest_score("nobody") is None
