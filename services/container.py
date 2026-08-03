"""Dependency wiring — builds repos, LLM, RAG and the agent context.

The ``Container`` is constructed once per process and shared across the API and
the graph. Tests should construct their own with an in-memory database to avoid
touching the file system.
"""

from __future__ import annotations

from functools import lru_cache

from agents.context import AgentContext
from config.settings import Settings, get_settings
from database.connection import Database
from database.repositories import (
    ConversationRepository,
    DocumentRepository,
    LessonRepository,
    MemoryRepository,
    ProgressRepository,
    QuizRepository,
    StudentRepository,
)
from services.llm_client import build_llm_client
from services.rag.service import RAGService


class Container:
    """Process-wide singletons: database, repos, LLM and RAG."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
        create_schema: bool = True,
    ) -> None:
        self.settings = settings or get_settings()

        if database is not None:
            self.db = database
        else:
            self.db = Database(self.settings.database)
            if create_schema and self.settings.database.create_on_startup:
                self.db.create_schema()

        self.student_repo = StudentRepository(self.db)
        self.quiz_repo = QuizRepository(self.db)
        self.conversation_repo = ConversationRepository(self.db)
        self.lesson_repo = LessonRepository(self.db)
        self.memory_repo = MemoryRepository(self.db)
        self.document_repo = DocumentRepository(self.db)
        self.progress_repo = ProgressRepository(self.db)

        self.llm = build_llm_client(self.settings.llm)
        self.rag = RAGService(self.settings.vector_store)

    def context(self) -> AgentContext:
        """Return the :class:`AgentContext` handed to every agent node."""
        return AgentContext(
            llm=self.llm,
            settings=self.settings,
            student_repo=self.student_repo,
            quiz_repo=self.quiz_repo,
            conversation_repo=self.conversation_repo,
            lesson_repo=self.lesson_repo,
            memory_repo=self.memory_repo,
            document_repo=self.document_repo,
            progress_repo=self.progress_repo,
            rag=self.rag,
        )

    def dispose(self) -> None:
        """Release DB connection pools (called at shutdown / in tests)."""
        self.db.dispose()


@lru_cache(maxsize=1)
def get_container(settings: Settings | None = None) -> Container:
    """Return the process-wide :class:`Container` (cached)."""
    return Container(settings)
