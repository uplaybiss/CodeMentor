"""Evaluation agent — grades a quiz submission and updates the student model.

Grading strategy
----------------
* MCQ / numerical questions are scored deterministically (exact match for MCQ,
  numeric tolerance for numerical) — fast, transparent, and testable.
* The subjective answer is graded by the LLM on a 0..1 scale; if the LLM is
  unavailable the turn still completes with a neutral grade and generic feedback.
* The final :class:`EvaluationResult` is *computed in code* (never left to the
  model's arithmetic), so scores are always consistent with the quiz.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel

from agents.context import AgentContext
from models.enums import QuestionType
from models.schemas import EvaluationResult, Quiz, QuizQuestion
from prompts import render_prompt
from utils.errors import LLMError, LLMOutputParsingError, ResourceNotFoundError
from utils.helpers import generate_id

logger = logging.getLogger(__name__)


class SubjectiveGrade(BaseModel):
    """LLM output for grading one subjective answer."""

    score: float  # 0.0 .. 1.0
    feedback: str
    weak_concepts: list[str] = []
    revision_topics: list[str] = []
    next_lesson: str = ""


def _is_correct(question: QuizQuestion, answer: str | None) -> bool:
    """Deterministic correctness check for objective questions."""
    if answer is None or not answer.strip():
        return False
    submitted = answer.strip()
    correct = question.correct_answer.strip()
    if question.type == QuestionType.NUMERICAL:
        try:
            return abs(float(submitted) - float(correct)) <= 1e-6 * max(
                1.0, abs(float(correct))
            )
        except ValueError:
            return False
    return submitted.lower() == correct.lower()


class EvaluationAgent:
    """Evaluation node: grades answers, persists the attempt and updates mastery."""

    name = "evaluation"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    # ------------------------------------------------------------------ run --
    async def run(self, state: dict) -> dict:
        student_id = state.get("student_id") or "anonymous"
        quiz_id = state.get("quiz_id")
        quiz = self._ctx.quiz_repo.get_quiz(quiz_id) if quiz_id else None
        if quiz is None:
            raise ResourceNotFoundError(f"Quiz {quiz_id!r} not found")

        answers = {
            item.get("question_id"): (item.get("answer") or "")
            for item in state.get("answers") or []
        }

        objective_earned = 0.0
        subjective_earned = 0.0
        max_score = float(quiz.total_points)
        per_topic: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        weak_by_topic: list[str] = []

        for question in quiz.questions:
            submitted = answers.get(question.id)
            if question.type == QuestionType.SUBJECTIVE:
                grade = await self._grade_subjective(question, submitted)
                earned = question.points * grade.score
                subjective_earned += earned
                correct = grade.score >= 0.6
                per_topic[question.topic].append((correct, earned))
                if not correct:
                    weak_by_topic.append(question.topic)
            else:
                correct = _is_correct(question, submitted)
                earned = float(question.points) if correct else 0.0
                objective_earned += earned
                per_topic[question.topic].append((correct, earned))
                if not correct:
                    weak_by_topic.append(question.topic)

        score = round(objective_earned + subjective_earned, 2)
        percent = round(score / max_score * 100, 1) if max_score else 0.0
        weak_concepts = list(dict.fromkeys(weak_by_topic))
        revision_topics = list(dict.fromkeys(weak_concepts))

        # Latest subjective feedback, if any, else a generic message.
        feedback_parts = [
            f"You scored {percent:.1f}% on this quiz."
        ]
        feedback_parts.append(
            f"Revise: {', '.join(revision_topics) or 'nothing — well done!'}"
        )

        evaluation = EvaluationResult(
            score=score,
            max_score=max_score,
            percent=percent,
            weak_concepts=weak_concepts,
            revision_topics=revision_topics,
            next_lesson=(
                f"Revise the fundamentals of {revision_topics[0]}" if revision_topics
                else f"Move on to an application of {quiz.topic}"
            ),
            feedback=" ".join(feedback_parts),
        )

        self._persist(student_id, quiz_id, quiz, answers, evaluation, per_topic)
        message = self._format_message(evaluation)
        logger.info(
            "Evaluation agent scored quiz %s at %.1f%%", quiz_id, percent
        )
        return {
            "evaluation": evaluation.model_dump(),
            "messages": [
                {"role": "assistant", "content": message, "node": self.name}
            ],
        }

    # ------------------------------------------------------- subjective -- --
    async def _grade_subjective(
        self, question: QuizQuestion, submitted: str | None
    ) -> SubjectiveGrade:
        """Grade one subjective answer; degrade gracefully if the LLM fails."""
        if submitted is None or not submitted.strip():
            return SubjectiveGrade(
                score=0.0,
                feedback="No answer was submitted for the written question.",
                weak_concepts=[question.topic],
                revision_topics=[question.topic],
                next_lesson=f"Revise {question.topic} and answer the written question.",
            )
        user_prompt = (
            f"Question: {question.question}\n\n"
            f"Model answer: {question.correct_answer}\n\n"
            f"Student answer: {submitted}\n\n"
            f"Topic: {question.topic}"
        )
        try:
            return await self._ctx.llm.generate_json(
                render_prompt("evaluation"),
                user_prompt,
                model_class=SubjectiveGrade,
            )
        except (LLMError, LLMOutputParsingError):
            logger.warning(
                "Subjective grading LLM call failed; grading %r as neutral",
                question.id,
            )
            return SubjectiveGrade(
                score=0.5 if submitted.strip() else 0.0,
                feedback="Your written answer was received but could not be "
                "auto-graded. Review the model answer and try again.",
                weak_concepts=[question.topic],
                revision_topics=[question.topic],
                next_lesson=f"Revise {question.topic}.",
            )

    # -------------------------------------------------------- persistence --
    def _persist(
        self,
        student_id: str,
        quiz_id: str,
        quiz: Quiz,
        answers: dict[str, str],
        evaluation: EvaluationResult,
        per_topic: dict[str, list[tuple[bool, float]]],
    ) -> None:
        """Write the attempt, score history, concept mastery and progress point."""
        self._ctx.quiz_repo.save_attempt(
            attempt_id=generate_id("attempt"),
            student_id=student_id,
            quiz_id=quiz_id,
            answers=answers,
            evaluation=evaluation,
        )
        self._ctx.student_repo.record_quiz_score(student_id, evaluation.percent)
        for concept in evaluation.weak_concepts:
            self._ctx.student_repo.add_weak_concept(student_id, concept)
        for topic, results in per_topic.items():
            if not results:
                continue
            correct_count = sum(1 for ok, _ in results if ok)
            proficiency = round(correct_count / len(results), 4)
            self._ctx.student_repo.upsert_concept_score(
                student_id, topic, proficiency
            )
        self._ctx.progress_repo.add_snapshot(
            student_id, quiz.topic, evaluation.percent
        )

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(evaluation: EvaluationResult) -> str:
        lines = [
            f"### Quiz result: {evaluation.percent:.1f}%",
            f"Score: {evaluation.score:g} / {evaluation.max_score:g}",
            f"Feedback: {evaluation.feedback}",
        ]
        if evaluation.weak_concepts:
            lines += ["", "**Weak areas:** " + ", ".join(evaluation.weak_concepts)]
        if evaluation.revision_topics:
            lines += ["", f"**Revise:** {', '.join(evaluation.revision_topics)}"]
        if evaluation.next_lesson:
            lines += ["", f"**Next lesson:** {evaluation.next_lesson}"]
        return "\n".join(lines)
