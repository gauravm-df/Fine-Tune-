"""
Phase 8: Intent Classification and Query Routing Service.
Routes user queries to:
1. EXACT_KYC: Deterministic SQL lookup from SQLite citizens table.
2. DOC_RAG: Hybrid Vector + BM25 document search with citations.
3. GENERAL_CHAT: Direct fine-tuned LLM conversational companion.
4. HYBRID: Combined KYC lookup + document policy.
"""

import re
from typing import Dict, Any, Tuple

class IntentRouter:
    def __init__(self):
        # Keyword triggers for KYC
        self.kyc_patterns = [
            re.compile(r'\b(p\d{5})\b', re.IGNORECASE),                         # Person ID: P00001
            re.compile(r'\b(\d{12})\b'),                                       # 12-digit Aadhaar
            re.compile(r'\b([A-Z]{5}\d{4}[A-Z])\b'),                           # 10-char PAN
            re.compile(r'\b(aadhaar|pan|kyc|father|mother|parents|citizen)\b', re.IGNORECASE),
            re.compile(r'\b(who is|kavya|arjun|rohan|sharma|verma|patel)\b', re.IGNORECASE)
        ]

        # Keyword triggers for Document RAG
        self.rag_patterns = [
            re.compile(r'\b(policy|guide|document|manual|architecture|phase|spec|requirements|pipeline|ingest|retrieval)\b', re.IGNORECASE),
            re.compile(r'\b(how to deploy|runbook|overview|diagram|rules|license)\b', re.IGNORECASE)
        ]

    def classify_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        clean_q = query.strip()

        is_kyc = False
        kyc_entities = {}

        # 1. Check for explicit citizen identifiers
        pid_match = re.search(r'\b(P\d{4,6})\b', clean_q, re.IGNORECASE)
        if pid_match:
            is_kyc = True
            kyc_entities["person_id"] = pid_match.group(1).upper()

        aadhaar_match = re.search(r'\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b', clean_q)
        if aadhaar_match:
            is_kyc = True
            kyc_entities["aadhaar"] = aadhaar_match.group(1).replace(" ", "")

        pan_match = re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', clean_q)
        if pan_match:
            is_kyc = True
            kyc_entities["pan"] = pan_match.group(1).upper()

        # Check keyword matches
        kyc_hits = sum(1 for pat in self.kyc_patterns if pat.search(clean_q))
        rag_hits = sum(1 for pat in self.rag_patterns if pat.search(clean_q))

        if kyc_entities or kyc_hits >= 2:
            is_kyc = True

        is_rag = (rag_hits >= 1)

        # Decide route
        if is_kyc and is_rag:
            route = "HYBRID"
        elif is_kyc:
            route = "EXACT_KYC"
        elif is_rag:
            route = "DOC_RAG"
        else:
            route = "GENERAL_CHAT"

        return route, {
            "query": clean_q,
            "route": route,
            "entities": kyc_entities,
            "kyc_confidence": round(kyc_hits / max(len(self.kyc_patterns), 1), 2),
            "rag_confidence": round(rag_hits / max(len(self.rag_patterns), 1), 2)
        }

intent_router = IntentRouter()

