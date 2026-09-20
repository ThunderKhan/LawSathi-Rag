import sys
import sys
import types
import unittest
from unittest.mock import MagicMock

for mod in ["chromadb", "chromadb.config", "sentence_transformers", "rank_bm25", "openai"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from src.rag_pipelines.hybrid_rag import HybridRAG


class TestDocumentLevelDiversification(unittest.TestCase):
    def _build_rag(self, max_chunks_per_document=2):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["A1", "A2", "A3", "B1", "C1"]
        rag.document_ids = ["case-a", "case-a", "case-a", "case-b", "case-c"]
        rag.alpha = 0.7
        rag.max_chunks_per_document = max_chunks_per_document
        rag._retrieve_bm25 = MagicMock(
            return_value={0: 1.00, 1: 0.95, 2: 0.90, 3: 0.89, 4: 0.88}
        )
        rag._retrieve_dense = MagicMock(
            return_value={0: 1.00, 1: 0.95, 2: 0.90, 3: 0.89, 4: 0.88}
        )
        return rag

    def test_top_k_is_diversified_by_document(self):
        rag = self._build_rag(2)
        self.assertEqual(rag.retrieve("relevant", k=4), ["A1", "A2", "B1", "C1"])

    def test_high_scoring_document_is_not_unlimited(self):
        rag = self._build_rag(1)
        self.assertEqual(rag.retrieve("relevant", k=3), ["A1", "B1", "C1"])

    def test_disabling_diversification_preserves_score_order(self):
        rag = self._build_rag(None)
        self.assertEqual(rag.retrieve("relevant", k=3), ["A1", "A2", "A3"])

if __name__ == "__main__":
    unittest.main()
