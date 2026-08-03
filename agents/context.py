"""Shared dependency context injected into every agent node."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from database.repositories.conversation_repository import ConversationRepository
from database.repositories.document_repository import DocumentRepository
from database.repositories.lesson_repository import LessonRepository
from database.repositories.memory_repository import MemoryRepository
from database.repositories.progress_repository import ProgressRepository
from database.repositories.quiz_repository import QuizRepository
from database.repositories.student_repository import StudentRepository
from services.llm_client import LLMClient
from services.rag.service import RAGService


@dataclass
class AgentContext:
    """Everything an agent needs: LLM, RAG and repositories."""

    llm: LLMClient
    settings: Settings
    student_repo: StudentRepository
    quiz_repo: QuizRepository
    conversation_repo: ConversationRepository
    lesson_repo: LessonRepository
    memory_repo: MemoryRepository
    document_repo: DocumentRepository
    progress_repo: ProgressRepository
    rag: RAGService
