import unittest
from unittest.mock import MagicMock

from src.rag_pipelines.hybrid_rag import HybridRAG
from src.rag_pipelines.legal_temporal_filter import (
    extract_temporal_metadata,
    matches_temporal_constraints,
)


class TestTemporalVersionAwareRetrieval(unittest.TestCase):
    def test_metadata_is_normalized_and_preserved(self):
        metadata = extract_temporal_metadata(
            {
                "judgment_date": "2020-06-15",
                "statute_version": "IPC-1860",
            },
            "case-2020",
        )
        self.assertEqual(metadata["document_id"], "case-2020")
        self.assertEqual(metadata["decision_date"], "2020-06-15")
        self.assertEqual(metadata["decision_year"], 2020)
        self.assertEqual(metadata["statute_version"], "IPC-1860")

    def test_temporal_constraints_reject_out_of_range_records(self):
        self.assertTrue(
            matches_temporal_constraints(
                {"decision_date": "2019-05-01", "decision_year": 2019, "statute_version": "v1"},
                decision_year=2019,
                as_of="2020-01-01",
                statute_version="v1",
            )
        )
        self.assertFalse(
            matches_temporal_constraints(
                {"decision_date": "2021-05-01", "decision_year": 2021, "statute_version": "v2"},
                decision_year=2019,
                as_of="2020-01-01",
                statute_version="v1",
            )
        )

    def test_as_of_honors_effective_and_repeal_dates(self):
        metadata = {
            "decision_date": "2018-01-01",
            "effective_date": "2019-01-01",
            "repeal_date": "2022-01-01",
        }
        self.assertTrue(matches_temporal_constraints(metadata, as_of="2020-01-01"))
        self.assertFalse(matches_temporal_constraints(metadata, as_of="2018-12-31"))
        self.assertFalse(matches_temporal_constraints(metadata, as_of="2022-02-01"))

    def test_hybrid_retrieval_filters_temporal_metadata(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["Historical case", "Current case"]
        rag.legal_metadata = [
            {"decision_year": 2010, "decision_date": "2010-01-01", "statute_version": "v1"},
            {"decision_year": 2020, "decision_date": "2020-01-01", "statute_version": "v2"},
        ]
        rag.alpha = 0.7
        rag._retrieve_bm25 = MagicMock(return_value={1: 1.0, 0: 0.9})
        rag._retrieve_dense = MagicMock(return_value={1: 1.0, 0: 0.9})

        self.assertEqual(
            rag.retrieve("legal issue", k=2, decision_year=2010),
            ["Historical case"],
        )

    def test_allow_out_of_range_bypasses_metadata_filter(self):
        rag = HybridRAG.__new__(HybridRAG)
        rag.chunks = ["Historical case", "Current case"]
        rag.legal_metadata = [
            {"decision_year": 2010},
            {"decision_year": 2020},
        ]
        rag.alpha = 0.7
        rag._retrieve_bm25 = MagicMock(return_value={1: 1.0, 0: 0.9})
        rag._retrieve_dense = MagicMock(return_value={1: 1.0, 0: 0.9})

        self.assertEqual(
            rag.retrieve("legal issue", k=2, decision_year=2010, allow_out_of_range=True),
            ["Current case", "Historical case"],
        )

if __name__ == "__main__":
    unittest.main()
