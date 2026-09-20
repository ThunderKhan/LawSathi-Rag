import os
import sys
import time
import json
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Sequence

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.utils import config
from src.utils.helpers import save_jsonl
from src.rag_pipelines.naive_rag import NaiveRAG
from src.rag_pipelines.citation_graph import CitationGraph, extract_case_citations

logger = logging.getLogger(__name__)

def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Perform Min-Max normalization over a dictionary of string-to-score values."""
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    diff = max_val - min_val
    if diff == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / diff for k, v in scores.items()}

class HybridRAG(NaiveRAG):
    """Hybrid RAG pipeline combining BM25 lexical search and Dense vector search."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embed_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.7,
        citation_expansion: bool = False,
        citation_expansion_limit: int = 2,
        citation_expansion_boost: float = 0.05,
    ):
        """Initialize HybridRAG with optional citation-graph expansion."""
        super().__init__(model_name=model_name)
        if citation_expansion_limit < 1:
            raise ValueError("citation_expansion_limit must be at least 1")
        if citation_expansion_boost < 0:
            raise ValueError("citation_expansion_boost must be non-negative")
        self.alpha = alpha
        self.citation_expansion = citation_expansion
        self.citation_expansion_limit = citation_expansion_limit
        self.citation_expansion_boost = citation_expansion_boost
        self.document_ids: List[str] = []
        self.citation_graph = CitationGraph()
        try:
            logger.info(f"HybridRAG: Loading dense encoder {embed_model} on CPU...")
            self.encoder = SentenceTransformer(embed_model)
            logger.info("HybridRAG: Initializing ChromaDB Ephemeral client...")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
            self.collection = self.client.get_or_create_collection("hybrid_chunks")
        except Exception as e:
            logger.error(f"Failed to initialize HybridRAG components: {e}")
            sys.exit(1)

    def index_documents(
        self,
        chunks: List[str],
        document_ids: Optional[List[str]] = None,
        cited_authorities: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        """Index chunks and optional document-level citation relationships."""
        if document_ids is not None and len(document_ids) != len(chunks):
            raise ValueError("document_ids must have the same length as chunks")
        if cited_authorities is not None and len(cited_authorities) != len(chunks):
            raise ValueError("cited_authorities must have the same length as chunks")

        self.chunks = chunks
        self.document_ids = (
            list(document_ids)
            if document_ids is not None
            else [f"chunk:{index}" for index in range(len(chunks))]
        )
        self.citation_graph = CitationGraph()

        for index, chunk in enumerate(chunks):
            citations = (
                set(cited_authorities[index])
                if cited_authorities is not None
                else extract_case_citations(chunk)
            )
            self.citation_graph.add_document(self.document_ids[index], citations)

        try:
            tokenized_chunks = [chunk.split() for chunk in chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            embeddings = self.encoder.encode(chunks, show_progress_bar=True)
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings.tolist()
            )
        except Exception as e:
            logger.error(f"Failed to build hybrid index: {e}")

    def _retrieve_bm25(self, query: str, limit: int) -> Dict[str, float]:
        """Get top lexical matching scores."""
        tokenized = query.split()
        scores = self.bm25.get_scores(tokenized)
        top_idx = np.argsort(scores)[-limit:][::-1]
        return {self.chunks[i]: float(scores[i]) for i in top_idx}

    def _retrieve_dense(self, query: str, limit: int) -> Dict[str, float]:
        """Get top semantic matching similarities (1 - distance)."""
        q_emb = self.encoder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=limit)
        if not res or "documents" not in res or not res["documents"] or not res["documents"][0]:
            return {}
        docs = res["documents"][0]
        dists = res["distances"][0] if "distances" in res and res["distances"] else [0.0]*len(docs)
        return {doc: 1.0 - float(dist) for doc, dist in zip(docs, dists)}

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Perform hybrid retrieval with optional citation-graph expansion."""
        if not self.chunks:
            return []
        try:
            limit = min(10, len(self.chunks))
            bm25_res = self._retrieve_bm25(query, limit)
            dense_res = self._retrieve_dense(query, limit)
            norm_bm25 = min_max_normalize(bm25_res)
            norm_dense = min_max_normalize(dense_res)
            combined = {}
            for chunk in set(bm25_res.keys()).union(dense_res.keys()):
                b_score = norm_bm25.get(chunk, 0.0)
                d_score = norm_dense.get(chunk, 0.0)
                combined[chunk] = self.alpha * d_score + (1.0 - self.alpha) * b_score

            if self.citation_expansion:
                query_citations = extract_case_citations(query)
                if query_citations and self.document_ids:
                    linked_documents = self.citation_graph.documents_linked_to(query_citations)
                    linked_chunks = []
                    for index, document_id in enumerate(self.document_ids):
                        if document_id in linked_documents and self.chunks[index] not in combined:
                            linked_chunks.append(self.chunks[index])
                    expansion_score = max(combined.values(), default=0.0) + self.citation_expansion_boost
                    for chunk in linked_chunks[: self.citation_expansion_limit]:
                        combined[chunk] = expansion_score

            sorted_chunks = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
            return sorted_chunks[:k]
        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            return []

    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate answer using configured API (NVIDIA NIM, OpenAI, or Ollama)."""
        context_text = "\n\n".join(contexts)
        prompt = (
            "Answer the following legal question based only on the provided context. "
            "If the answer is not in the context, say 'I cannot answer from the provided context.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        if config.USE_LOCAL_MODEL:
            import requests
            url = f"{config.API_BASE_URL}/api/generate"
            payload = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                raise
        else:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.API_BASE_URL,
                api_key=config.OPENAI_API_KEY
            )
            try:
                response = client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"API error: {e}")
                raise

def run_main() -> None:
    """Validate HybridRAG pipeline on 3 test records."""
    test_path = project_root / "data" / "test.jsonl"
    if not test_path.exists():
        logger.error(f"Test split not found at {test_path}")
        sys.exit(1)
    if config.USE_LOCAL_MODEL:
        try:
            requests.get(config.OLLAMA_BASE_URL, timeout=5.0)
        except Exception:
            logger.error("Ollama server is offline. Please run it before execution.")
            sys.exit(1)
    records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            records.append(json.loads(line))
    chunks = list(set([c for r in records for c in r.get("context_chunks", [])]))
    rag = HybridRAG()
    rag.index_documents(chunks)
    results = [rag.answer(r["question"]) for r in records]
    out_path = project_root / "results" / "predictions" / "hybrid_rag.jsonl"
    save_jsonl(out_path, results)
    logger.info("Successfully executed verification run for HybridRAG.")

if __name__ == "__main__":
    run_main()
