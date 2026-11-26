"""
Phase 9: Grounded Context Builder and Anti-Hallucination Assembler.
Connects Intent Router, KYC SQL Database, and Hybrid RAG into a validated prompt context.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_kyc_db"))
sys.path.append(str(PROJECT_ROOT / "03_rag"))
sys.path.append(str(PROJECT_ROOT / "06_orchestration"))

from kyc_service import kyc_service
from hybrid_engine import hybrid_rag
from intent_router import intent_router
from prompt_templates import get_system_prompt

class ContextBuilder:
    def __init__(self):
        self.router = intent_router
        self.kyc = kyc_service
        self.rag = hybrid_rag

    def build_context(
        self,
        query: str,
        actor_role: str = "officer",
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        route, meta = self.router.classify_intent(query)
        system_prompt = get_system_prompt(route)

        sql_record = None
        rag_chunks = []
        context_blocks = []

        # 1. Handle KYC Route
        if route in ["EXACT_KYC", "HYBRID"]:
            entities = meta.get("entities", {})
            identifier = None

            if "person_id" in entities:
                identifier = entities["person_id"]
            elif "aadhaar" in entities:
                identifier = entities["aadhaar"]
            elif "pan" in entities:
                identifier = entities["pan"]
            else:
                identifier = query

            kyc_res = self.kyc.lookup_record(identifier, actor_role=actor_role)
            if kyc_res["found"]:
                sql_record = kyc_res["record"]
                context_blocks.append(f"VERIFIED DATABASE RECORD:\n{kyc_res['formatted_summary']}")
            else:
                context_blocks.append(f"DATABASE LOOKUP STATUS: {kyc_res['message']}")

        # 2. Handle RAG Route (Document Knowledge & WhatsApp Chat Retrieval)
        if route in ["DOC_RAG", "HYBRID", "GENERAL_CHAT"]:
            top_n = 2 if route == "GENERAL_CHAT" else 3
            rag_chunks = self.rag.search(query, top_k=top_n)
            if rag_chunks:
                header = "RELEVANT CHAT HISTORY EXCERPTS:" if route == "GENERAL_CHAT" else "RELEVANT KNOWLEDGE EXCERPTS:"
                context_blocks.append(header)
                for c in rag_chunks:
                    context_blocks.append(f"{c['citation']} {c['text']}")

        # Assemble full grounding prompt
        grounding_text = "\n\n".join(context_blocks) if context_blocks else ""

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}\n"
        )

        if grounding_text:
            full_prompt += f"\nGROUNDING KNOWLEDGE:\n{grounding_text}<|im_end|>\n"
        else:
            full_prompt += "<|im_end|>\n"

        # Attach history
        if chat_history:
            for turn in chat_history[-3:]:
                full_prompt += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"

        full_prompt += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        return {
            "route": route,
            "system_prompt": system_prompt,
            "grounding_text": grounding_text,
            "sql_record": sql_record,
            "rag_chunks": rag_chunks,
            "full_formatted_prompt": full_prompt
        }

context_builder = ContextBuilder()

