"""Data-access repositories, one file per aggregate root."""

from database.repositories.base_repository import BaseRepository
from database.repositories.conversation_repository import ConversationRepository
from database.repositories.document_repository import DocumentRepository
from database.repositories.lesson_repository import LessonRepository
from database.repositories.memory_repository import MemoryRepository
from database.repositories.progress_repository import ProgressRepository
from database.repositories.quiz_repository import QuizRepository
from database.repositories.student_repository import StudentRepository

__all__ = [
    "BaseRepository",
    "ConversationRepository",
    "DocumentRepository",
    "LessonRepository",
    "MemoryRepository",
    "ProgressRepository",
    "QuizRepository",
    "StudentRepository",
]
