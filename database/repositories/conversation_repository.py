"""Persistence for the conversation transcript."""

from __future__ import annotations

from sqlalchemy import func, select

from database.models import Conversation
from database.repositories.base_repository import BaseRepository
from utils.helpers import utc_now_iso


class ConversationRepository(BaseRepository):
    """Append-only transcript of student <-> tutor messages."""

    def add_message(
        self, student_id: str, role: str, content: str, metadata_json: dict | None = None
    ) -> int:
        with self.session() as session:
            row = Conversation(
                student_id=student_id,
                role=role,
                content=content,
                metadata_json=metadata_json or {},
                created_at=utc_now_iso(),
            )
            session.add(row)
            session.flush()
            return row.id

    def list_messages(self, student_id: str, limit: int = 50) -> list[dict]:
        """Return the most recent ``limit`` messages, oldest first."""
        with self.session() as session:
            rows = session.execute(
                select(Conversation)
                .where(Conversation.student_id == student_id)
                .order_by(Conversation.id.desc())
                .limit(limit)
            ).scalars().all()
            rows = list(reversed(rows))
            return [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "metadata": r.metadata_json or {},
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def count(self, student_id: str) -> int:
        with self.session() as session:
            return session.execute(
                select(func.count()).select_from(Conversation).where(
                    Conversation.student_id == student_id
                )
            ).scalar_one()
