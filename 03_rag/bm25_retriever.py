"""
Phase 4: Sparse BM25 Keyword Search Engine (BM25Okapi).
Pure high-performance Python implementation with zero third-party C-extensions.
"""

import math
import re
from typing import List, Dict, Any, Tuple

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.corpus = documents
        self.doc_lengths = []
        self.doc_freqs = []
        df: Dict[str, int] = {}
        n_docs = len(documents)

        if n_docs == 0:
            return

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self.doc_lengths.append(len(tokens))
            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / n_docs if n_docs > 0 else 0

        # Calculate standard Lucene/Okapi IDF
        self.idf = {}
        for t, count in df.items():
            self.idf[t] = math.log(1 + (n_docs - count + 0.5) / (count + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        tokens = self._tokenize(query)
        if not tokens or not self.corpus:
            return []

        scores: List[Tuple[int, float]] = []

        for idx, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            score = 0.0
            for t in tokens:
                if t in freqs:
                    tf = freqs[t]
                    numerator = self.idf.get(t, 0.0) * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += (numerator / denominator)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.corpus[idx], float(score)))
        return results

