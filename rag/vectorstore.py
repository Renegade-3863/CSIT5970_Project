"""
Vector Store
============
Thin wrapper around ChromaDB that provides a unified interface for
adding, querying, and persisting document embeddings.

Embeddings are generated locally using a sentence-transformer model
(default: all-MiniLM-L6-v2) so no external embedding API is required.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages a ChromaDB collection for storing and retrieving document chunks.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory where the database files are stored.
        embedding_model: Name of the sentence-transformer model used for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Lazy initialisation (avoids import errors if ChromaDB not installed)
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        try:
            import chromadb
            from chromadb.utils import embedding_functions as ef

            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._embedding_fn = ef.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialised = True
            logger.info(
                "ChromaDB collection '%s' ready at %s",
                self.collection_name,
                self.persist_directory,
            )
        except ImportError as exc:
            logger.error("ChromaDB or sentence-transformers not installed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add text chunks to the vector store.

        Args:
            texts: List of text strings to embed and store.
            ids: Optional list of unique IDs (auto-generated if omitted).
            metadatas: Optional list of metadata dicts (one per text).
        """
        self._ensure_init()
        if not texts:
            return

        if ids is None:
            existing_count = self._collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # ChromaDB upsert in batches of 500
        batch_size = 500
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_ids = ids[start:start + batch_size]
            batch_meta = metadatas[start:start + batch_size]
            self._collection.upsert(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_meta,
            )
        logger.debug("Indexed %d document chunks.", len(texts))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query the vector store for the most similar documents.

        Returns:
            A list of (text, score, metadata) tuples sorted by relevance.
        """
        self._ensure_init()
        params: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, max(1, self._collection.count())),
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            params["where"] = where

        results = self._collection.query(**params)

        output: List[Tuple[str, float, Dict[str, Any]]] = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            # Convert cosine distance to similarity score (0=identical, 2=opposite)
            similarity = 1.0 - dist / 2.0
            output.append((doc, similarity, meta or {}))

        return output

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        self._ensure_init()
        return self._collection.count()

    def delete_collection(self) -> None:
        """Drop the entire collection (irreversible)."""
        self._ensure_init()
        self._client.delete_collection(self.collection_name)
        self._initialised = False
        self._collection = None
        logger.warning("Deleted ChromaDB collection '%s'.", self.collection_name)
