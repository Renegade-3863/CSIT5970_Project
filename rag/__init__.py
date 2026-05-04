"""
RAG (Retrieval-Augmented Generation) Package
============================================
Provides vector-store indexing and retrieval using ChromaDB and
sentence-transformer embeddings.
"""

from rag.vectorstore import VectorStore
from rag.indexer import DocumentIndexer
from rag.retriever import RAGRetriever

__all__ = ["VectorStore", "DocumentIndexer", "RAGRetriever"]
