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

from pydantic import BaseModel, Field

from agents.base import call_tool, profile_prompt_section
from agents.context import AgentContext
from config.presentation import topic_label
from harness.errors import ToolExecutionError, ToolParameterError
from models.enums import QuestionType
from models.schemas import EvaluationResult, Quiz, QuizQuestion
from prompts import render_prompt
from utils.errors import LLMError, LLMOutputParsingError, ResourceNotFoundError
from utils.helpers import generate_id

logger = logging.getLogger(__name__)


class SubjectiveGrade(BaseModel):
    """LLM output for grading one subjective answer."""

    score: float = Field(ge=0.0, le=1.0)
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
            return abs(float(submitted) - float(correct)) <= 1e-6 * max(1.0, abs(float(correct)))
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
        quiz = (
            await call_tool(
                self._ctx, self.name, "quiz.get", student_id=student_id, quiz_id=quiz_id
            )
            if quiz_id
            else None
        )
        if quiz is None:
            raise ResourceNotFoundError(f"Quiz {quiz_id!r} not found")

        submitted_items = state.get("answers") or []
        valid_ids = {q.id for q in quiz.questions}
        submitted_ids = [item.get("question_id") for item in submitted_items]
        if len(set(submitted_ids)) != len(submitted_ids) or any(
            q not in valid_ids for q in submitted_ids
        ):
            raise ToolParameterError("Answers contain duplicate or unknown question IDs")
        answers = {item.get("question_id"): (item.get("answer") or "") for item in submitted_items}

        objective_earned = 0.0
        subjective_earned = 0.0
        max_score = float(quiz.total_points)
        per_topic: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        weak_by_topic: list[str] = []
        wrong_answers: list[dict] = []
        question_feedback: list[dict] = []

        for question in quiz.questions:
            submitted = answers.get(question.id)
            if question.type == QuestionType.SUBJECTIVE:
                grade = await self._grade_subjective(question, submitted, student_id)
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
            if not correct:
                wrong_answers.append(
                    {
                        "topic": question.topic,
                        "question": question.question,
                        "submitted": submitted or "",
                        "explanation": question.explanation or question.correct_answer,
                    }
                )

            question_feedback.append(
                {
                    "question_id": question.id,
                    "is_correct": correct,
                    "earned": earned,
                    "max_score": question.points,
                    "submitted": submitted or "",
                    "reference_answer": question.correct_answer,
                    "explanation": grade.feedback
                    if question.type == QuestionType.SUBJECTIVE
                    else question.explanation,
                }
            )

        score = round(objective_earned + subjective_earned, 2)
        percent = round(score / max_score * 100, 1) if max_score else 0.0
        weak_concepts = list(dict.fromkeys(weak_by_topic))
        revision_topics = list(dict.fromkeys(weak_concepts))

        # Latest subjective feedback, if any, else a generic message.
        feedback_parts = [f"本次测验得分为{percent:.1f}%。"]
        feedback_parts.append(
            f"建议复习：{'、'.join(topic_label(x) for x in revision_topics)}。"
            if revision_topics
            else "本次题目均达到正确标准，可以继续巩固和应用。"
        )

        evaluation = EvaluationResult(
            score=score,
            max_score=max_score,
            percent=percent,
            weak_concepts=weak_concepts,
            revision_topics=revision_topics,
            next_lesson=(
                f"复习{topic_label(revision_topics[0])}的基础概念"
                if revision_topics
                else f"尝试{topic_label(quiz.topic)}的应用练习"
            ),
            feedback=" ".join(feedback_parts),
        )

        saved = await call_tool(
            self._ctx,
            self.name,
            "evaluation.save",
            student_id=student_id,
            quiz=quiz,
            answers=answers,
            evaluation=evaluation,
            per_topic=dict(per_topic),
            wrong_answers=wrong_answers,
        )
        evaluation = EvaluationResult.model_validate(saved)
        message = self._format_message(evaluation)
        logger.info("Evaluation agent scored quiz %s at %.1f%%", quiz_id, percent)
        return {
            "topic": quiz.topic,
            "question_feedback": question_feedback,
            "evaluation": evaluation.model_dump(),
            "messages": [{"role": "assistant", "content": message, "node": self.name}],
        }

    # ------------------------------------------------------- subjective -- --
    async def _grade_subjective(
        self, question: QuizQuestion, submitted: str | None, student_id: str = "anonymous"
    ) -> SubjectiveGrade:
        """Grade one subjective answer; degrade gracefully if the LLM fails."""
        if submitted is None or not submitted.strip():
            return SubjectiveGrade(
                score=0.0,
                feedback="这道主观题没有填写答案，请结合参考答案补充理解。",
                weak_concepts=[question.topic],
                revision_topics=[question.topic],
                next_lesson=f"复习{topic_label(question.topic)}后再尝试回答。",
            )
        user_prompt = (
            f"Question: {question.question}\n\n"
            f"Model answer: {question.correct_answer}\n\n"
            f"Student answer: {submitted}\n\n"
            f"Topic: {question.topic}\n{profile_prompt_section(None)}"
        )
        try:
            return await call_tool(
                self._ctx,
                self.name,
                "llm.json",
                student_id=student_id,
                system=render_prompt("evaluation"),
                user=user_prompt,
                model_class=SubjectiveGrade,
            )
        except (LLMError, LLMOutputParsingError, ToolExecutionError):
            logger.warning(
                "Subjective grading LLM call failed; grading %r as neutral",
                question.id,
            )
            return SubjectiveGrade(
                score=0.5 if submitted.strip() else 0.0,
                feedback="已收到你的回答，但本次未能完成自动评分，暂按中性分数记录。请对照参考答案自查。",
                weak_concepts=[question.topic],
                revision_topics=[question.topic],
                next_lesson=f"复习{topic_label(question.topic)}。",
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
            self._ctx.student_repo.upsert_concept_score(student_id, topic, proficiency)
        self._ctx.progress_repo.add_snapshot(student_id, quiz.topic, evaluation.percent)

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(evaluation: EvaluationResult) -> str:
        lines = [
            f"### 测验结果：{evaluation.percent:.1f}%",
            f"得分：{evaluation.score:g} / {evaluation.max_score:g}",
            f"反馈：{evaluation.feedback}",
        ]
        if evaluation.weak_concepts:
            lines += [
                "",
                "**薄弱知识点：** " + "、".join(topic_label(x) for x in evaluation.weak_concepts),
            ]
        if evaluation.revision_topics:
            lines += [
                "",
                f"**建议复习：** {'、'.join(topic_label(x) for x in evaluation.revision_topics)}",
            ]
        if evaluation.next_lesson:
            lines += ["", f"**下一步：** {evaluation.next_lesson}"]
        return "\n".join(lines)
