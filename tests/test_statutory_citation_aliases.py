import unittest

import src.rag_pipelines.hybrid_rag as hybrid_rag_module
import src.rag_pipelines.naive_rag as naive_rag_module
from src.preprocessing.cleaners import normalize_statutory_citations


class FakeBM25:
    def __init__(self, tokenized_documents):
        self.tokenized_documents = tokenized_documents

    def get_scores(self, tokenized_query):
        query_tokens = set(tokenized_query)
        return [
            float(len(query_tokens.intersection(document_tokens)))
            for document_tokens in self.tokenized_documents
        ]


class FakeEmbeddings:
    def tolist(self):
        return [[0.0, 0.0] for _ in range(len(self))]

    def __len__(self):
        return 2


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return FakeEmbeddings()


class FakeCollection:
    def add(self, **kwargs):
        return None


class TestStatutoryCitationAliases(unittest.TestCase):
    def setUp(self):
        self.original_naive_bm25 = naive_rag_module.BM25Okapi
        self.original_hybrid_bm25 = hybrid_rag_module.BM25Okapi
        naive_rag_module.BM25Okapi = FakeBM25
        hybrid_rag_module.BM25Okapi = FakeBM25

    def tearDown(self):
        naive_rag_module.BM25Okapi = self.original_naive_bm25
        hybrid_rag_module.BM25Okapi = self.original_hybrid_bm25

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
        rag = naive_rag_module.NaiveRAG.__new__(naive_rag_module.NaiveRAG)
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
        rag = hybrid_rag_module.HybridRAG.__new__(hybrid_rag_module.HybridRAG)
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
