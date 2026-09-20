import os
import sys
import time
import json
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any

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
from src.rag_pipelines.legal_temporal_filter import matches_temporal_constraints

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

    def __init__(self, model_name: str = "gpt-4o-mini", embed_model: str = "all-MiniLM-L6-v2", alpha: float = 0.7):
        """Initialize HybridRAG pipeline with optional legal temporal metadata filtering."""
        super().__init__(model_name=model_name)
        self.alpha = alpha
        self.legal_metadata: List[Dict[str, Any]] = []
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
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index chunks and their optional legal temporal/version metadata."""
        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError("metadata must have the same length as chunks")
        self.chunks = chunks
        self.legal_metadata = [dict(item) for item in metadata] if metadata is not None else [{} for _ in chunks]
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

    def _retrieve_bm25(self, query: str, limit: int) -> Dict[int, float]:
        """Get top lexical matching scores keyed by chunk index."""
        tokenized = query.split()
        scores = self.bm25.get_scores(tokenized)
        top_idx = np.argsort(scores)[-limit:][::-1]
        return {int(i): float(scores[i]) for i in top_idx}

    def _retrieve_dense(self, query: str, limit: int) -> Dict[int, float]:
        """Get top semantic matching similarities keyed by chunk index."""
        q_emb = self.encoder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=limit)
        if not res or "documents" not in res or not res["documents"] or not res["documents"][0]:
            return {}
        dists = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(res["documents"][0])
        ids = res.get("ids", [[]])[0] if res.get("ids") else []
        if ids:
            return {
                int(chunk_id.rsplit("_", 1)[-1]): 1.0 - float(distance)
                for chunk_id, distance in zip(ids, dists)
            }
        return {
            self.chunks.index(document): 1.0 - float(distance)
            for document, distance in zip(res["documents"][0], dists)
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        decision_year: Optional[int] = None,
        as_of: Optional[str] = None,
        statute_version: Optional[str] = None,
        allow_out_of_range: bool = False,
    ) -> List[str]:
        """Perform hybrid retrieval with optional legal temporal/version constraints."""
        if not self.chunks:
            return []
        try:
            constrained = any(
                value is not None
                for value in (decision_year, as_of, statute_version)
            )
            limit = len(self.chunks) if constrained and not allow_out_of_range else min(10, len(self.chunks))
            bm25_res = self._retrieve_bm25(query, limit)
            dense_res = self._retrieve_dense(query, limit)
            norm_bm25 = min_max_normalize(bm25_res)
            norm_dense = min_max_normalize(dense_res)
            combined = {}
            for chunk_index in set(bm25_res.keys()).union(dense_res.keys()):
                b_score = norm_bm25.get(chunk_index, 0.0)
                d_score = norm_dense.get(chunk_index, 0.0)
                combined[chunk_index] = self.alpha * d_score + (1.0 - self.alpha) * b_score

            if constrained and not allow_out_of_range:
                combined = {
                    index: score
                    for index, score in combined.items()
                    if matches_temporal_constraints(
                        self.legal_metadata[index],
                        decision_year=decision_year,
                        as_of=as_of,
                        statute_version=statute_version,
                    )
                }

            sorted_indices = sorted(combined.keys(), key=lambda index: combined[index], reverse=True)
            return [self.chunks[index] for index in sorted_indices[:k]]
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
