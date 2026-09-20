import re
from collections import defaultdict
from typing import Iterable


_SCC_CITATION = re.compile(
    r"(?<!\w)(?:\(|\[)?(\d{4})(?:\)|\])?\s+(\d+)\s+SCC\s+(\d+)(?!\w)",
    re.IGNORECASE,
)
_AIR_CITATION = re.compile(
    r"(?<!\w)AIR\s+(\d{4})\s+([A-Za-z]+)\s+(\d+)(?!\w)",
    re.IGNORECASE,
)


def extract_case_citations(text: str) -> set[str]:
    """Return canonical SCC and AIR citation signatures found in text."""
    if not text:
        return set()

    citations: set[str] = set()
    for year, volume, page in _SCC_CITATION.findall(text):
        citations.add(f"scc:{year}:{volume}:{page}")
    for year, court, page in _AIR_CITATION.findall(text):
        citations.add(f"air:{year}:{court.lower()}:{page}")
    return citations


class CitationGraph:
    """Small in-memory graph linking indexed documents to cited authorities."""

    def __init__(self) -> None:
        self._document_to_citations: dict[str, set[str]] = defaultdict(set)
        self._citation_to_documents: dict[str, set[str]] = defaultdict(set)

    def add_document(self, document_id: str, citations: Iterable[str]) -> None:
        normalized = {citation.strip().lower() for citation in citations if citation and citation.strip()}
        self._document_to_citations[document_id].update(normalized)
        for citation in normalized:
            self._citation_to_documents[citation].add(document_id)

    def documents_linked_to(self, citations: Iterable[str]) -> set[str]:
        """Return documents that directly cite any supplied authority."""
        linked: set[str] = set()
        for citation in citations:
            linked.update(self._citation_to_documents.get(citation.strip().lower(), set()))
        return linked
