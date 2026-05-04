"""
Test suite for the RAG module.
Uses a temporary in-memory ChromaDB collection to avoid side effects.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# EvaluationMetrics tests (no external deps)
# ---------------------------------------------------------------------------

class TestEvaluationMetrics(unittest.TestCase):
    """Tests for benchmark/metrics.py – pure maths, no external deps."""

    def setUp(self):
        from benchmark.metrics import EvaluationMetrics
        self.metrics = EvaluationMetrics

    def test_mean_basic(self):
        self.assertAlmostEqual(self.metrics.mean([2.0, 4.0, 6.0]), 4.0)

    def test_mean_empty(self):
        self.assertEqual(self.metrics.mean([]), 0.0)

    def test_median_odd(self):
        self.assertEqual(self.metrics.median([1.0, 3.0, 5.0]), 3.0)

    def test_median_even(self):
        self.assertEqual(self.metrics.median([1.0, 3.0]), 2.0)

    def test_median_empty(self):
        self.assertEqual(self.metrics.median([]), 0.0)

    def test_std_basic(self):
        # std of [2, 4, 6] ≈ 2.0
        self.assertAlmostEqual(self.metrics.std([2.0, 4.0, 6.0]), 2.0)

    def test_std_single_element(self):
        self.assertEqual(self.metrics.std([5.0]), 0.0)

    def test_pass_rate_all_passing(self):
        self.assertEqual(self.metrics.pass_at_threshold([8.0, 9.0, 10.0], 7.0), 1.0)

    def test_pass_rate_none_passing(self):
        self.assertEqual(self.metrics.pass_at_threshold([1.0, 2.0, 3.0], 7.0), 0.0)

    def test_pass_rate_partial(self):
        rate = self.metrics.pass_at_threshold([5.0, 8.0, 9.0], 7.0)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_pass_rate_empty(self):
        self.assertEqual(self.metrics.pass_at_threshold([], 7.0), 0.0)

    def test_summarise_keys(self):
        summary = self.metrics.summarise([5.0, 7.0, 9.0])
        for key in ("mean", "median", "std", "pass_rate", "min", "max", "count"):
            self.assertIn(key, summary)

    def test_compare_modes(self):
        mode_scores = {
            "llm_only": [5.0, 6.0],
            "custom_agent": [8.0, 9.0],
        }
        comparison = self.metrics.compare_modes(mode_scores)
        self.assertIn("llm_only", comparison)
        self.assertIn("custom_agent", comparison)
        self.assertGreater(
            comparison["custom_agent"]["mean"],
            comparison["llm_only"]["mean"],
        )

    def test_relative_improvement(self):
        improvement = self.metrics.relative_improvement([5.0, 5.0], [7.5, 7.5])
        self.assertAlmostEqual(improvement, 50.0)

    def test_relative_improvement_zero_baseline(self):
        """Should return inf when baseline mean is 0 and improved > 0."""
        result = self.metrics.relative_improvement([0.0, 0.0], [5.0, 5.0])
        self.assertEqual(result, float("inf"))


# ---------------------------------------------------------------------------
# DocumentIndexer / VectorStore tests (require chromadb)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.environ.get("RUN_CHROMA_TESTS", "0") == "1",
    "Set RUN_CHROMA_TESTS=1 to run ChromaDB integration tests",
)
class TestVectorStore(unittest.TestCase):
    """Integration tests for the VectorStore (require chromadb installed)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from rag.vectorstore import VectorStore
        self.store = VectorStore(
            collection_name="test_collection",
            persist_directory=self._tmpdir,
            embedding_model="all-MiniLM-L6-v2",
        )

    def test_add_and_query(self):
        """Documents added to the store should be retrievable."""
        self.store.add_documents(
            ["The sky is blue.", "Python is a programming language."],
            ids=["d1", "d2"],
        )
        results = self.store.query("colour of the sky", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("sky", results[0][0].lower())

    def test_count(self):
        """count() should reflect the number of indexed documents."""
        self.store.add_documents(["doc one", "doc two", "doc three"])
        self.assertEqual(self.store.count(), 3)

    def test_empty_texts_noop(self):
        """Adding empty list should be a no-op."""
        before = self.store.count()
        self.store.add_documents([])
        self.assertEqual(self.store.count(), before)


@unittest.skipUnless(
    os.environ.get("RUN_CHROMA_TESTS", "0") == "1",
    "Set RUN_CHROMA_TESTS=1 to run ChromaDB integration tests",
)
class TestDocumentIndexer(unittest.TestCase):
    """Integration tests for DocumentIndexer."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from rag.vectorstore import VectorStore
        from rag.indexer import DocumentIndexer
        self.store = VectorStore(
            collection_name="indexer_test",
            persist_directory=self._tmpdir,
        )
        self.indexer = DocumentIndexer(self.store, chunk_size=200, chunk_overlap=20)

    def test_index_texts(self):
        """index_texts() should return positive chunk count."""
        texts = ["Short document one.", "Short document two."]
        count = self.indexer.index_texts(texts, source_name="test")
        self.assertGreater(count, 0)

    def test_index_text_file(self):
        """index_file() should successfully index a .txt file."""
        txt_path = os.path.join(self._tmpdir, "sample.txt")
        with open(txt_path, "w") as fh:
            fh.write("This is a sample text document for testing purposes.")
        count = self.indexer.index_file(txt_path)
        self.assertGreater(count, 0)

    def test_unsupported_extension(self):
        """Unsupported file extensions should raise ValueError."""
        fake_path = os.path.join(self._tmpdir, "file.xyz")
        open(fake_path, "w").close()
        with self.assertRaises(ValueError):
            self.indexer.index_file(fake_path)


if __name__ == "__main__":
    unittest.main()
