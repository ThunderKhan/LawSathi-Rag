import sys
import types
import unittest

import numpy as np

fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing.cleaners import normalize_statutory_citations
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.hybrid_rag import HybridRAG


class FakeEncoder:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 2))


class FakeCollection:
    def add(self, **kwargs):
        return None


class TestStatutoryCitationAliases(unittest.TestCase):
    def test_common_section_aliases_share_one_canonical_form(self):
        aliases = [
            "Section 302",
            "Sections 302",
            "Sec. 302",
            "Sec 302",
            "S. 302",
            "§ 302",
            "§302",
            "SECTION 302.",
        ]

        for alias in aliases:
            self.assertEqual(normalize_statutory_citations(alias), "section 302")

    def test_subsection_and_trailing_punctuation_are_normalized(self):
        self.assertEqual(
            normalize_statutory_citations("Section 302(1)(a),"),
            "section 302(1)(a)",
        )
        self.assertEqual(
            normalize_statutory_citations("§302(1)(a);"),
            "section 302(1)(a)",
        )

    def test_naive_bm25_retrieves_with_an_alias(self):
        rag = NaiveRAG.__new__(NaiveRAG)
        rag.chunks = []
        rag.bm25 = None
        rag.index_documents(
            [
                "Section 302. Punishment for the offence.",
                "Section 304. A different offence.",
            ]
        )

        results = rag.retrieve("§ 302", k=1)

        self.assertEqual(results, ["Section 302. Punishment for the offence."])

    def test_hybrid_bm25_retrieves_with_an_alias(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = []
        rag.bm25 = None
        rag.encoder = FakeEncoder()
        rag.collection = FakeCollection()
        rag.index_documents(
            [
                "Section 302. Punishment for the offence.",
                "Section 304. A different offence.",
            ]
        )

        results = rag._retrieve_bm25("Sec. 302", limit=1)

        self.assertEqual(
            list(results),
            ["Section 302. Punishment for the offence."],
        )


if __name__ == "__main__":
    unittest.main()
