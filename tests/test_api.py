"""API endpoint tests via the FastAPI TestClient."""

from __future__ import annotations

from models.enums import DifficultyLevel, QuestionType
from models.schemas import Quiz, QuizQuestion


class TestHealth:
    def test_health_ok(self, client, settings):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == settings.app.version
        assert data["demo_mode"] is True


class TestStudents:
    def test_get_creates(self, client):
        resp = client.get("/api/v1/students/api_s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["student_id"] == "api_s1"
        assert data["name"] == "Learner"

    def test_put_updates(self, client):
        resp = client.put(
            "/api/v1/students/api_s2",
            json={"name": "Linus", "level": "intermediate"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Linus"
        assert resp.json()["level"] == "intermediate"

    def test_put_partial_update(self, client):
        client.put("/api/v1/students/api_s3", json={"name": "Grace"})
        resp = client.get("/api/v1/students/api_s3")
        assert resp.json()["name"] == "Grace"
        assert resp.json()["level"] == "beginner"  # unchanged


class TestQuizzes:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/quizzes/quizless")
        assert resp.status_code == 200
        assert resp.json()["quiz_ids"] == []

    def test_get_missing(self, client):
        resp = client.get("/api/v1/quiz/ghost")
        assert resp.status_code == 404


class TestDashboard:
    def test_dashboard_for_new_student(self, client):
        resp = client.get("/api/v1/dashboard/dash_s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["student_id"] == "dash_s1"
        assert data["name"] == "Learner"
        assert data["total_attempts"] == 0
        assert data["total_documents"] == 0
        assert data["quiz_scores"] == []


class TestChat:
    async def test_chat_learn(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"student_id": "chat_s1", "message": "teach me about gravity", "topic": "gravity"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["student_id"] == "chat_s1"
        assert data["intent"] == "learn"
        assert data["messages"]
        assert data["lesson"]

    async def test_chat_with_quiz_flag(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "student_id": "chat_s2",
                "message": "explain photosynthesis",
                "topic": "photosynthesis",
                "with_quiz": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["lesson"]
        assert data["quiz"]


class TestSubmitQuiz:
    def _create_quiz(self, client, student_id: str) -> Quiz:
        quiz = Quiz(
            id="api_quiz_1",
            topic="Geometry",
            difficulty=DifficultyLevel.BEGINNER,
            questions=[
                QuizQuestion(
                    id="q1", type=QuestionType.MCQ, topic="Geometry",
                    question="Area of circle = ?", options=["pi*r^2", "2*pi*r"], correct_answer="pi*r^2",
                ),
                QuizQuestion(
                    id="q2", type=QuestionType.NUMERICAL, topic="Geometry",
                    question="3 * 4 = ?", options=[], correct_answer="12",
                ),
                QuizQuestion(
                    id="q3", type=QuestionType.SUBJECTIVE, topic="Geometry",
                    question="What is a triangle?", options=[], correct_answer="A polygon with 3 sides.",
                ),
            ],
        )
        from api.dependencies import get_container_dep

        container = client.app.dependency_overrides.get(get_container_dep)()
        container.student_repo.get_or_create(student_id)
        container.quiz_repo.create_quiz(quiz, student_id)
        # Use the repo-reconstructed quiz id so evaluation can load it.
        return container.quiz_repo.get_quiz(quiz.id) or quiz

    async def test_submit_and_receive_evaluation(self, client):
        quiz = self._create_quiz(client, "sub_s1")
        resp = client.post(
            "/api/v1/chat/submit-quiz",
            json={
                "quiz_id": quiz.id,
                "student_id": "sub_s1",
                "answers": [
                    {"question_id": f"{quiz.id}q1", "answer": "pi*r^2"},
                    {"question_id": f"{quiz.id}q2", "answer": "12"},
                    {"question_id": f"{quiz.id}q3", "answer": "It has 3 sides."},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["evaluation"] is not None
        assert data["evaluation"]["percent"] > 0
        assert data["recommendations"] is not None
