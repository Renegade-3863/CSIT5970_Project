"""
Knowledge Base Tool
===================
Searches the RAG vector store and returns relevant document excerpts.
The agent calls this tool when it needs domain-specific information from
the indexed knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from tools.base import BaseTool

logger = logging.getLogger(__name__)


class KnowledgeBaseTool(BaseTool):
    """
    Tool: knowledge_base
    Search the indexed knowledge base for relevant information.
    Input: a natural-language search query.
    Output: top-K relevant document excerpts as a formatted string.
    """

    name = "knowledge_base"
    description = (
        "Search the internal knowledge base for relevant technical information. "
        "Input should be a concise search query. "
        "Returns the most relevant document excerpts."
    )

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        rag_cfg = config.get("rag", {})
        self._top_k: int = rag_cfg.get("top_k", 5)
        self._threshold: float = rag_cfg.get("similarity_threshold", 0.4)
        self._retriever = None
        self._init_retriever(rag_cfg)

    def run(self, tool_input: str) -> str:
        if self._retriever is None:
            return "Knowledge base is not available. Please ensure RAG is configured."

        try:
            docs = self._retriever.retrieve(tool_input, top_k=self._top_k)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Knowledge base retrieval failed: %s", exc)
            return f"Retrieval error: {exc}"

        if not docs:
            return "No relevant documents found in the knowledge base."

        results: List[str] = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            source = ""
            if hasattr(doc, "metadata") and doc.metadata.get("source"):
                source = f" [Source: {doc.metadata['source']}]"
            results.append(f"[{i}]{source}\n{content.strip()}")

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_retriever(self, rag_cfg: Dict[str, Any]) -> None:
        try:
            from rag.retriever import RAGRetriever
            self._retriever = RAGRetriever(rag_cfg)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not initialise RAG retriever for tool: %s", exc)
