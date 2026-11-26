"""
Phase 4: High-Performance Batched Neural Embedding Model & Persistent Vector Database.
Features:
1. True 896-Dimensional Transformer Contextual Embeddings with GPU/MPS & Batched Inference.
2. Persistent SQLite Vector Database with duplicate-avoidance cache.
3. Fast batch processing: 100s of chunks in seconds.
"""

import struct
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "01_data" / "vector_store.sqlite"
DEFAULT_EMBED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

class NeuralEmbeddingModel:
    def __init__(self, model_id: str = DEFAULT_EMBED_MODEL):
        self.model_id = model_id
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Neural Embedding Model: {model_id} on {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch-generates high-precision normalized 896-dim neural vectors."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.extend(normalized.cpu().tolist())

        return all_embeddings

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class PersistentVectorDB:
    def __init__(self, db_path: Path = VECTOR_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vec_source ON document_vectors(source_file);")
            conn.commit()

    def get_existing_chunk_ids(self) -> Set[str]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT chunk_id FROM document_vectors;")
            return {r[0] for r in cur.fetchall()}

    def save_vectors_batch(self, records: List[Dict[str, Any]]):
        if not records:
            return
        with self._get_connection() as conn:
            cur = conn.cursor()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            data = []
            for r in records:
                blob = struct.pack(f"{len(r['embedding'])}f", *r['embedding'])
                data.append((
                    r["chunk_id"],
                    r["source"],
                    r.get("file_path", ""),
                    r["text"],
                    len(r["embedding"]),
                    blob,
                    now
                ))
            cur.executemany("""
                INSERT OR REPLACE INTO document_vectors 
                (chunk_id, source_file, file_path, chunk_text, embedding_dim, embedding_blob, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, data)
            conn.commit()

    def get_all_vectors(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT chunk_id, source_file, file_path, chunk_text, embedding_dim, embedding_blob FROM document_vectors;")
            rows = cur.fetchall()

        records = []
        for r in rows:
            dim = r["embedding_dim"]
            blob = r["embedding_blob"]
            vec = list(struct.unpack(f"{dim}f", blob))
            records.append({
                "chunk_id": r["chunk_id"],
                "source": r["source_file"],
                "file_path": r["file_path"],
                "text": r["chunk_text"],
                "vector": vec
            })
        return records

    def count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_vectors;")
            return cur.fetchone()[0]

class ProductionVectorStore:
    def __init__(self):
        self.db = PersistentVectorDB()
        self._embedder = None

    @property
    def embedder(self) -> NeuralEmbeddingModel:
        if self._embedder is None:
            self._embedder = NeuralEmbeddingModel()
        return self._embedder

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Batched indexing with cache check: only embeds chunks not already stored."""
        existing_ids = self.db.get_existing_chunk_ids()
        new_docs = [d for d in documents if d["chunk_id"] not in existing_ids]

        if not new_docs:
            print(f"All {len(documents)} chunks already cached in Vector DB ({self.db.db_path.name}). Total: {self.db.count()}.")
            return

        print(f"Generating Neural Embeddings for {len(new_docs)} new chunks (cached: {len(documents) - len(new_docs)})...")
        texts = [d["text"] for d in new_docs]
        embeddings = self.embedder.embed_texts(texts, batch_size=32)

        records_to_save = []
        for doc, emb in zip(new_docs, embeddings):
            records_to_save.append({
                "chunk_id": doc["chunk_id"],
                "source": doc["source"],
                "file_path": doc.get("file_path", ""),
                "text": doc["text"],
                "embedding": emb
            })

        self.db.save_vectors_batch(records_to_save)
        print(f"Saved {len(records_to_save)} new vectors to {self.db.db_path.name}. Total: {self.db.count()}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        stored = self.db.get_all_vectors()
        if not stored:
            return []

        q_vec = self.embedder.embed_text(query)
        scores = []

        for item in stored:
            d_vec = item["vector"]
            sim = sum(q * d for q, d in zip(q_vec, d_vec))
            scores.append((item, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

vector_store = ProductionVectorStore()
