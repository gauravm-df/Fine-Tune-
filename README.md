# Hybrid RAG + Fine-Tuned LLM + Exact KYC System

A production-ready hybrid AI system for Apple Silicon (**MPS**), **CUDA**, and **CPU** that combines:
1. **Fine-Tuned Conversational Foundation LLM (PEFT / LoRA & QLoRA)** — supports models such as **Qwen 2.5** (`Qwen/Qwen2.5-0.5B-Instruct` by default) and **TinyLlama**, fine-tuned strictly on PII-sanitized WhatsApp dialogues.
2. **Deterministic Exact KYC Database (`02_kyc_db/`)** — parameterized SQLite lookups (`01_data/kyc_index.sqlite`) with role-based PII masking (`officer`/`admin` vs `guest`) and audit logging so citizen records are never hallucinated or baked into model weights.
3. **Hybrid RAG Engine (`03_rag/`)** — dense vector embeddings + sparse BM25 keyword retrieval fused via Reciprocal Rank Fusion (RRF) for policy and technical document Q&A with citations.
4. **Intent Router & Context Builder (`06_orchestration/`)** — routes queries across `EXACT_KYC`, `DOC_RAG`, `GENERAL_CHAT`, and `HYBRID` pipelines.

---

## 📁 Project Architecture

```
Fine-Tune-Cum-RAG/
├── 01_data/
│   ├── clean_conversational_train.jsonl  # PII-sanitized WhatsApp training split
│   ├── clean_conversational_val.jsonl    # PII-sanitized WhatsApp validation split
│   ├── kyc_index.sqlite                  # Authoritative SQLite citizen KYC database
│   ├── vector_store.sqlite               # Persisted RAG vector store
│   ├── prepare_finetune_data.py          # Formats & splits JSONL datasets
│   ├── finetune_train.jsonl              # Legacy training split
│   └── finetune_val.jsonl                # Legacy validation split
├── 02_config/
│   └── config.py                         # Central config (base model, LoRA rank/alpha, device)
├── 02_kyc_db/
│   ├── db_manager.py                     # Parameterized SQLite engine & audit logging
│   └── kyc_service.py                    # RBAC & PII field masking (Aadhaar/PAN)
├── 03_finetune/
│   ├── train_lora_local.py               # Legacy local Apple Silicon MPS / CPU LoRA script
│   └── train_qlora_colab.py              # 4-bit GPU QLoRA script (Colab / NVIDIA)
├── 03_rag/
│   ├── document_loader.py                # Multi-format loader & chunker
│   ├── bm25_retriever.py                 # Sparse BM25 keyword retriever
│   ├── vector_store.py                   # Dense embedding vector store
│   └── hybrid_engine.py                  # Hybrid Vector + BM25 + RRF fusion engine
├── 04_merge/
│   └── merge_adapters.py                 # Fuses LoRA adapter weights into base model
├── 04_model/
│   └── base_loader.py                    # Unified model loader (merged / LoRA / base fallback)
├── 05_finetune/
│   ├── preprocess_whatsapp.py            # PII sanitizer & conversational split generator
│   └── train_lora.py                     # Conversational LoRA fine-tuning (WhatsApp data only)
├── 05_inference/
│   ├── cli_chat.py                       # Legacy interactive terminal chat
│   ├── app_streamlit.py                  # Legacy standalone Streamlit chat UI
│   └── server_fastapi.py                 # Legacy standalone FastAPI server
├── 06_orchestration/
│   ├── intent_router.py                  # Routes EXACT_KYC, DOC_RAG, GENERAL_CHAT, HYBRID
│   ├── context_builder.py                # Assembles grounded prompts & citations
│   └── prompt_templates.py               # System & role prompt templates
├── 07_serving/
│   ├── app_ui.py                         # Hybrid RAG + KYC + LLM Streamlit Web UI
│   └── api_service.py                    # Unified Production FastAPI Gateway (port 8000)
├── docs/
│   ├── ARCHITECTURE_DIAGRAMS.md          # System architecture & flow diagrams
│   ├── REPOSITORY_ANALYSIS_REPORT.md     # Asset audit & zero-KYC compliance rules
│   └── SYSTEM_RUNBOOK.md                 # Operations guide & curl API examples
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## 🚀 Quick Start Guide

### Step 1: Environment Setup
```bash
cd Fine-Tune-Cum-RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Alternatively, run `./setup_env.sh` to create the virtual environment and install dependencies.)*

### Step 2: Preprocess & Run Conversational LoRA Fine-Tuning
Sanitize conversational dialogues (keeping raw citizen KYC records strictly out of model weights) and train the LoRA adapter locally on Apple Silicon MPS / CUDA / CPU:
```bash
python3 05_finetune/preprocess_whatsapp.py
python3 05_finetune/train_lora.py
```
*Conversational LoRA adapters are saved to `05_finetune/lora_adapter`.*

*(For legacy fine-tuning scripts, see `03_finetune/train_lora_local.py` or `03_finetune/train_qlora_colab.py` for 4-bit NVIDIA/Colab QLoRA.)*

### Step 3: (Optional) Merge Adapters
To fuse adapter weights into a standalone model without runtime LoRA overhead:
```bash
python3 04_merge/merge_adapters.py
```
*Merged weights are saved to `04_merge/merged_model` and automatically picked up by `04_model/base_loader.py`.*

### Step 4: Serve & Run Inference

- **Streamlit Hybrid Web UI (Recommended — Port 8501):**
  ```bash
  streamlit run 07_serving/app_ui.py
  ```
  Displays assistant responses alongside verified SQL records and retrieved RAG citations.

- **Unified FastAPI Gateway (Port 8000):**
  ```bash
  python3 07_serving/api_service.py
  ```
  - Swagger Docs: `http://localhost:8000/docs`
  - Health Check: `http://localhost:8000/health`
  - Direct KYC Lookup: `GET http://localhost:8000/v1/kyc/{identifier}`
  - Unified Chat: `POST http://localhost:8000/v1/chat`

- **Legacy Standalone Inference (`05_inference/`):**
  ```bash
  python3 05_inference/cli_chat.py
  streamlit run 05_inference/app_streamlit.py
  python3 05_inference/server_fastapi.py
  ```

---

## 📚 Additional Documentation

- [System Runbook & API Examples](docs/SYSTEM_RUNBOOK.md)
- [Architecture Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)
- [Repository Analysis & Compliance Report](docs/REPOSITORY_ANALYSIS_REPORT.md)
