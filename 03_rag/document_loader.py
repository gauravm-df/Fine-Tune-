"""
Phase 3: Multi-Format Document Ingestion & Intelligent Chunking Pipeline.
Supports TXT, MD, CSV, JSON, and DOCX formats with metadata preservation & sliding overlap.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Loads a document and extracts chunks with full source metadata."""
        if not file_path.exists():
            return []

        ext = file_path.suffix.lower()
        if ext == ".jsonl":
            return self._parse_jsonl(file_path, limit=100)
        elif ext == ".csv":
            content = self._parse_csv(file_path)
        elif ext == ".docx":
            content = self._parse_docx(file_path)
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                return []

        return self.create_chunks(content, source=file_path.name, file_path=str(file_path))

    def _parse_jsonl(self, path: Path, limit: int = 2000) -> List[Dict[str, Any]]:
        """Specialized parser for WhatsApp conversation JSONL archives."""
        chunks = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if len(chunks) >= limit:
                    break
                try:
                    data = json.loads(line.strip())
                    user_text = ""
                    asst_text = ""
                    if "messages" in data and isinstance(data["messages"], list):
                        for m in data["messages"]:
                            if m.get("role") == "user":
                                user_text = m.get("content", "")
                            elif m.get("role") == "assistant":
                                asst_text = m.get("content", "")
                    elif "text" in data:
                        t = data["text"]
                        u = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', t, re.DOTALL)
                        a = re.search(r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)', t, re.DOTALL)
                        if u:
                            user_text = u.group(1).strip()
                        if a:
                            asst_text = a.group(1).strip()

                    if user_text and asst_text:
                        chat_text = f"User asked: {user_text}\nAssistant replied: {asst_text}"
                        chunks.append({
                            "chunk_id": f"whatsapp_{i}",
                            "source": "WhatsApp_Chat_Archive",
                            "file_path": str(path),
                            "text": chat_text,
                            "length": len(chat_text)
                        })
                except Exception:
                    continue
        return chunks

    def _parse_csv(self, path: Path) -> str:
        lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                lines.append(" | ".join(row))
        return "\n".join(lines)

    def _parse_docx(self, path: Path) -> str:
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception:
            # Fallback if docx module not in current venv
            return f"[DOCX Content from {path.name}]"

    def create_chunks(self, text: str, source: str, file_path: str) -> List[Dict[str, Any]]:
        """Splits raw text into overlapping semantic passages."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        chunk_idx = 0

        for p in paragraphs:
            clean_p = p.strip()
            if len(clean_p) < 20:
                continue

            # If paragraph fits in chunk size
            if len(clean_p) <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{source}_{chunk_idx}",
                    "source": source,
                    "file_path": file_path,
                    "text": clean_p,
                    "length": len(clean_p)
                })
                chunk_idx += 1
            else:
                # Sliding window with overlap
                start = 0
                while start < len(clean_p):
                    end = start + self.chunk_size
                    segment = clean_p[start:end].strip()
                    if segment:
                        chunks.append({
                            "chunk_id": f"{source}_{chunk_idx}",
                            "source": source,
                            "file_path": file_path,
                            "text": segment,
                            "length": len(segment)
                        })
                        chunk_idx += 1
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks

document_loader = DocumentLoader()

