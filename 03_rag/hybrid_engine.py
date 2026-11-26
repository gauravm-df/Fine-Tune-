"""
Phase 4: Hybrid RAG Engine with Reciprocal Rank Fusion (RRF).
Merges Dense Vector Similarity + Sparse BM25 Keyword Search.
Provides strict source attribution and document citations.
"""

from pathlib import Path
from typing import List, Dict, Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "03_rag"))
from document_loader import document_loader
from bm25_retriever import BM25Retriever
from vector_store import vector_store

class HybridRAGEngine:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.vector_store = vector_store
        self.indexed_docs: List[Dict[str, Any]] = []

    def ingest_folder_or_files(self, paths: List[Path]):
        """Ingests documents across paths and indexes into both sparse & dense engines."""
        all_chunks = []
        for p in paths:
            if p.is_dir():
                for sub in p.glob("*"):
                    if sub.suffix.lower() in [".txt", ".md", ".csv", ".docx"]:
                        chunks = document_loader.load_file(sub)
                        all_chunks.extend(chunks)
            elif p.is_file():
                chunks = document_loader.load_file(p)
                all_chunks.extend(chunks)

        self.indexed_docs = all_chunks
        self.bm25.index_documents(all_chunks)
        self.vector_store.index_documents(all_chunks)
        print(f"Hybrid RAG Engine Indexed {len(all_chunks)} chunks from {len(paths)} source targets.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes Hybrid Search with Reciprocal Rank Fusion."""
        if not self.indexed_docs:
            return []

        bm25_res = self.bm25.search(query, top_k=top_k * 2)
        dense_res = self.vector_store.search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion scoring
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, (doc, _) in enumerate(bm25_res):
            cid = doc["chunk_id"]
            chunk_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        for rank, (doc, _) in enumerate(dense_res):
            cid = doc["chunk_id"]
            chunk_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Sort by fused score
        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []

        for cid, score in sorted_cids[:top_k]:
            doc = chunk_map[cid]
            final_results.append({
                "chunk_id": cid,
                "source": doc["source"],
                "text": doc["text"],
                "rrf_score": round(score, 5),
                "citation": f"[Source: {doc['source']}]"
            })

        return final_results

hybrid_rag = HybridRAGEngine()

