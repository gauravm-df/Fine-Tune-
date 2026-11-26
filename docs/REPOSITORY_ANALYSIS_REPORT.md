# Repository Analysis & Baseline Architecture Report
**Project:** Hybrid RAG + Fine-Tuned Foundation LLM + Exact KYC System
**Target Environment:** macOS (Apple Silicon MPS) & Linux (CUDA / Colab)
**Date:** November 25, 2026

---

## 1. Executive Summary & Objective

This analysis audits the existing assets in `/Users/DF/Documents/llm` and establishes the migration and integration blueprint for **Existing_LLM_Project**. 

The fundamental goal is to build an enterprise-grade hybrid AI system that combines:
1. **Existing Foundation LLM** (Qwen 2.5 / TinyLlama) fine-tuned with **LoRA** strictly on conversational WhatsApp data.
2. **Exact KYC Database Engine** (SQLite/PostgreSQL) as the sole, authoritative, zero-hallucination source of truth for citizen records.
3. **Hybrid RAG Engine** (Dense vector embeddings + Sparse BM25 keyword search + Reciprocal Rank Fusion) for policy/document knowledge retrieval.
4. **Intelligent Query Router & Context Builder** that enforces RBAC, system prompt isolation, and anti-hallucination context assembly.

---

## 2. Inventory of Existing Components & Reuse Strategy

| Existing Asset | Current State | Audit Finding | Reuse & Integration Strategy |
| :--- | :--- | :--- | :--- |
| `01_data/kyc_index.sqlite` | 15.9 MB, 100,000 rows | Contains clean `citizens` table (`person_id`, `name`, `father_name`, `mother_name`, `age`, `aadhaar_number`, `pan_number`). | **REUSE AS-IS for Phase 5.** Connect via parameterized SQL engine with audit logging. Never fine-tune into model weights. |
| `01_data/train_whatsapp_sample.jsonl` | 17.8 MB | 25,000 raw ChatML WhatsApp dialogues across 10 categories. | **REUSE with PII Sanitization (Phase 2).** Clean phone numbers, strip raw names, format strictly for conversational fine-tuning. |
| `02_config/config.py` | Initial config | Base model set to `Qwen/Qwen2.5-0.5B-Instruct`, LoRA rank $r=16$, alpha $32$. | **REUSE & EXTEND.** Centralize hardware parameters, RAG chunk sizes, and DB paths. |
| `03_finetune/train_lora_local.py` | Local LoRA script | Trains LoRA adapter on MPS. | **UPGRADE.** Ensure dataset contains ONLY cleaned WhatsApp data, excluding raw KYC identities. |
| `04_merge/merge_adapters.py` | Merge utility | Fuses LoRA adapter weights using `merge_and_unload()`. | **REUSE AS-IS.** For zero-overhead standalone deployment. |
| `*.docx` Guides & Manuals | 5 Docx files (~200 KB) | Comprehensive technical guides and policy documentation. | **INGESTION ASSET for Phase 3.** Use as ground-truth documents to build the Hybrid RAG knowledge base. |
| Virtual Environment (`venv`) | Python 3.13 / MPS | PyTorch, Transformers, Datasets, PEFT 0.22, Streamlit, FastAPI installed. | **REUSE.** No redundant virtual environment needed. |

---

## 3. Strict Compliance & Boundary Audit

### A. The "Zero-KYC in LLM Weights" Rule
- **Previous Risk:** In early test scripts, KYC question-answer pairs (`"Who is the father of Arjun Sharma?"`) were mixed into the fine-tuning training file.
- **Risk Identified:** LLMs are probabilistic. Baking exact citizen identities into model weights leads to hallucinations, PII leakage, and violates DPDP/GDPR "Right to be Forgotten".
- **Remediation:** Strict separation.
  - `finetune_train.jsonl` will **strictly contain conversational dialogues** (WhatsApp chats).
  - All citizen queries will route **deterministically to the SQL database**.

### B. Hardware & Latency Assessment
- Device: Apple Silicon Unified Memory (`mps`).
- Base Model: `Qwen/Qwen2.5-0.5B-Instruct` requires ~1.2 GB RAM in bfloat16.
- SQLite query latency: `< 2 milliseconds` for 100,000 records with indexed columns.
- Embedding Generation: Lightweight Hugging Face sentence transformers (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`) run in `< 15ms` per chunk.

---

## 4. Architectural Gaps to Complete in Phases 2–12

1. **Phase 2 (Dataset Cleaning):** Add PII sanitizer to strip phone numbers, real emails, and format balanced conversational splits.
2. **Phase 3 & 4 (Hybrid RAG):** Implement multi-format document loader (DOCX, PDF, TXT), semantic chunker with 10% overlap, dense embeddings, BM25 sparse search, and Reciprocal Rank Fusion (RRF).
3. **Phase 5 (KYC Engine):** Build `db_manager.py` with indexed lookups, parameterized queries to prevent SQL injection, and audit trail logging.
4. **Phase 8 & 9 (Orchestrator):** Implement intent classifier (`KYC_LOOKUP`, `DOC_RAG`, `CONVERSATIONAL`, `HYBRID`) and strict grounded context builder with citation formatting.
5. **Phase 10 & 11 (API, UI & Automated Tests):** Unified FastAPI backend with OpenAPI schema, Streamlit frontend, and automated test suite verifying anti-hallucination and security.

