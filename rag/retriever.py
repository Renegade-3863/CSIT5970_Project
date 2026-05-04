"""
RAG Retriever
=============
High-level interface that wraps the VectorStore to provide
LangChain-compatible Document objects for seamless integration
with the agent's memory and tool stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Lightweight LangChain-compatible document container."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.page_content[:60].replace("\n", " ")
        return f"Document(content='{preview}...', metadata={self.metadata})"


class RAGRetriever:
    """
    Provides retrieve() and add_documents() methods that the MemoryManager
    and KnowledgeBaseTool use.

    Args:
        rag_config: The 'rag' section from agent_config.yaml.
    """

    def __init__(self, rag_config: Dict[str, Any]):
        self._top_k: int = rag_config.get("top_k", 5)
        self._threshold: float = rag_config.get("similarity_threshold", 0.4)

        from rag.vectorstore import VectorStore

        self._store = VectorStore(
            collection_name=rag_config.get("collection_name", "knowledge_base"),
            persist_directory=rag_config.get("persist_directory", "./data/chroma_db"),
            embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant documents for *query*.

        Args:
            query: Natural-language query string.
            top_k: Override the default top_k.
            filter_metadata: Optional ChromaDB metadata filter dict.

        Returns:
            List of Document objects sorted by descending similarity.
        """
        k = top_k or self._top_k
        results = self._store.query(
            query_text=query,
            top_k=k,
            where=filter_metadata,
        )

        docs: List[Document] = []
        for text, score, meta in results:
            if score >= self._threshold:
                docs.append(Document(page_content=text, metadata={**meta, "score": score}))

        logger.debug(
            "Retrieved %d/%d docs above threshold %.2f for query: %s",
            len(docs),
            len(results),
            self._threshold,
            query[:60],
        )
        return docs

    def add_documents(self, docs: List[Any]) -> None:
        """
        Add documents to the vector store.

        Accepts both Document instances and raw strings.
        """
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                texts.append(doc.page_content)
                metadatas.append(getattr(doc, "metadata", {}))
            else:
                texts.append(str(doc))
                metadatas.append({})
        self._store.add_documents(texts, metadatas=metadatas)

    def count(self) -> int:
        """Return the number of indexed document chunks."""
        return self._store.count()
