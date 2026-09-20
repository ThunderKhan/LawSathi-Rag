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

from src.rag_pipelines.citation_graph import CitationGraph, extract_case_citations
from src.rag_pipelines.hybrid_rag import HybridRAG


class TestCitationGraphExpansion(unittest.TestCase):
    def test_citations_are_canonicalized(self):
        self.assertEqual(
            extract_case_citations("(2010) 1 SCC 100 and AIR 1980 SC 123"),
            {"scc:2010:1:100", "air:1980:sc:123"},
        )

    def test_graph_resolves_directly_citing_documents(self):
        graph = CitationGraph()
        graph.add_document("case-a", {"scc:2010:1:100"})
        graph.add_document("case-b", {"scc:2015:2:200"})

        self.assertEqual(
            graph.documents_linked_to({"scc:2010:1:100"}),
            {"case-a"},
        )

    def test_hybrid_retrieval_can_expand_to_linked_authority(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["Unrelated holding.", "Case A cites (2010) 1 SCC 100."]
        rag.document_ids = ["case-b", "case-a"]
        rag.alpha = 0.7
        rag.citation_expansion = True
        rag.citation_expansion_limit = 1
        rag.citation_expansion_boost = 0.05
        rag.citation_graph = CitationGraph()
        rag.citation_graph.add_document("case-a", {"scc:2010:1:100"})
        rag.citation_graph.add_document("case-b", set())
        rag._retrieve_bm25 = MagicMock(return_value={"Unrelated holding.": 1.0})
        rag._retrieve_dense = MagicMock(return_value={"Unrelated holding.": 1.0})

        self.assertEqual(
            rag.retrieve("What did (2010) 1 SCC 100 hold?", k=2),
            ["Case A cites (2010) 1 SCC 100.", "Unrelated holding."],
        )

    def test_citation_expansion_can_be_disabled(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["Unrelated holding.", "Case A cites (2010) 1 SCC 100."]
        rag.document_ids = ["case-b", "case-a"]
        rag.alpha = 0.7
        rag.citation_expansion = False
        rag.citation_graph = CitationGraph()
        rag._retrieve_bm25 = MagicMock(return_value={"Unrelated holding.": 1.0})
        rag._retrieve_dense = MagicMock(return_value={"Unrelated holding.": 1.0})

        self.assertEqual(
            rag.retrieve("What did (2010) 1 SCC 100 hold?", k=1),
            ["Unrelated holding."],
        )

if __name__ == "__main__":
    unittest.main()
