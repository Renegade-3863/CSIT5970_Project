"""
Document Indexer
================
Handles loading, chunking, and indexing of documents into the
VectorStore.  Supports plain text, PDF, and Word documents.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default chunk parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class DocumentIndexer:
    """
    Loads documents from files or raw text, splits them into overlapping
    chunks, and indexes them into a VectorStore.
    """

    def __init__(
        self,
        vector_store,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_directory(self, directory: str) -> int:
        """
        Recursively index all supported files in *directory*.

        Returns the total number of chunks indexed.
        """
        total = 0
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    count = self.index_file(path)
                    total += count
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Skipping file %s: %s", path, exc)
        return total

    def index_file(self, file_path: str) -> int:
        """
        Index a single file.

        Returns the number of chunks created.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".txt":
            text = self._read_text(file_path)
        elif ext == ".pdf":
            text = self._read_pdf(file_path)
        elif ext in (".docx", ".doc"):
            text = self._read_docx(file_path)
        elif ext in (".md", ".rst"):
            text = self._read_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        source = os.path.relpath(file_path)
        chunks = self._split(text)
        metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
        self._store.add_documents(chunks, metadatas=metadatas)
        logger.info("Indexed %d chunks from %s", len(chunks), file_path)
        return len(chunks)

    def index_texts(
        self,
        texts: List[str],
        source_name: str = "manual",
    ) -> int:
        """
        Index a list of raw strings.

        Returns the number of chunks created.
        """
        all_chunks: List[str] = []
        all_meta: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            chunks = self._split(text)
            all_chunks.extend(chunks)
            all_meta.extend(
                {"source": source_name, "doc_idx": i, "chunk": j}
                for j, _ in enumerate(chunks)
            )
        self._store.add_documents(all_chunks, metadatas=all_meta)
        return len(all_chunks)

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF support: pip install pypdf") from exc
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def _read_docx(path: str) -> str:
        try:
            import docx
        except ImportError as exc:
            raise ImportError(
                "python-docx is required for .docx support: pip install python-docx"
            ) from exc
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)

    # ------------------------------------------------------------------
    # Text splitter
    # ------------------------------------------------------------------

    def _split(self, text: str) -> List[str]:
        """
        Split *text* into overlapping chunks of at most *chunk_size*
        characters with *chunk_overlap* characters of overlap.
        """
        text = text.strip()
        if not text:
            return []

        # First try sentence-aware splitting via LangChain if available
        try:
            return self._langchain_split(text)
        except Exception:  # pylint: disable=broad-except
            pass

        # Fallback: simple character-level sliding window
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start += self._chunk_size - self._chunk_overlap
        return chunks

    def _langchain_split(self, text: str) -> List[str]:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
