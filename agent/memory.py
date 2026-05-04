"""
Memory Manager
==============
Provides short-term (conversation history) and long-term (RAG-backed)
memory for the agent, enabling retrieval-augmented generation.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Combines:
    - Short-term buffer: last N conversation turns kept in a deque.
    - Long-term RAG memory: delegates to the RAG retriever when available.
    """

    def __init__(self, config: Dict[str, Any]):
        rag_cfg = config.get("rag", {})
        self._rag_enabled: bool = rag_cfg.get("enabled", False)
        self._top_k: int = rag_cfg.get("top_k", 5)

        # Short-term circular buffer (stores (role, text) tuples)
        self._buffer: Deque[Tuple[str, str]] = deque(maxlen=20)

        # Lazy-loaded RAG retriever
        self._retriever: Optional[Any] = None
        if self._rag_enabled:
            self._init_retriever(rag_cfg)

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------

    def add_turn(self, role: str, text: str) -> None:
        """Append a conversation turn to the short-term buffer."""
        self._buffer.append((role, text))

    def get_history(self) -> List[Dict[str, str]]:
        """Return the short-term buffer as a list of {role, content} dicts."""
        return [{"role": role, "content": text} for role, text in self._buffer]

    def clear_history(self) -> None:
        """Clear the short-term conversation buffer."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Long-term / RAG memory
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant context from the knowledge base.

        Returns an empty list if RAG is disabled or the retriever is
        unavailable.
        """
        if not self._rag_enabled or self._retriever is None:
            return []
        k = top_k or self._top_k
        try:
            docs = self._retriever.retrieve(query, top_k=k)
            return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("RAG retrieval failed: %s", exc)
            return []

    def index_documents(self, docs: List[Any]) -> None:
        """
        Index new documents into the long-term knowledge base.

        Each element in *docs* may be a string, a LangChain Document, or
        any object with a ``page_content`` attribute.
        """
        if self._retriever is None:
            logger.warning("Retriever not initialised; skipping indexing.")
            return
        try:
            self._retriever.add_documents(docs)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to index documents: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_retriever(self, rag_cfg: Dict[str, Any]) -> None:
        try:
            from rag.retriever import RAGRetriever

            self._retriever = RAGRetriever(rag_cfg)
            logger.info("RAG retriever initialised successfully.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not initialise RAG retriever: %s", exc)
            self._retriever = None
