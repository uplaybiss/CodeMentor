"""RAG agent — retrieves relevant notes and injects them into the state."""

from __future__ import annotations

import logging

from agents.context import AgentContext

logger = logging.getLogger(__name__)


class RAGAgent:
    """Retrieval node: grounds later nodes in the student's uploaded notes."""

    name = "rag"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    async def run(self, state: dict) -> dict:
        """Run a semantic search and store the context for downstream nodes."""
        query = (state.get("topic") or state.get("user_input") or "").strip()
        if not query:
            return {
                "rag_context": {"retrieved_chunks": [], "used": False},
                "rag_sources": [],
            }

        context = self._ctx.rag.build_context(query)
        sources = list(dict.fromkeys(c.source for c in context.retrieved_chunks))
        logger.info(
            "RAG retrieved %d chunk(s) for %r (sources: %s)",
            len(context.retrieved_chunks),
            query[:60],
            sources or "none",
        )
        return {
            "rag_context": context.model_dump(),
            "rag_sources": sources,
        }
