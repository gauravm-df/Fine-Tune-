"""
Phase 10: Unified Production FastAPI Gateway.
Integrates Intent Router, KYC Database, Hybrid RAG, and LLM Inference.
Provides full REST API with role-based authorization and health checks.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_kyc_db"))
sys.path.append(str(PROJECT_ROOT / "03_rag"))
sys.path.append(str(PROJECT_ROOT / "04_model"))
sys.path.append(str(PROJECT_ROOT / "06_orchestration"))

from kyc_service import kyc_service
from hybrid_engine import hybrid_rag
from context_builder import context_builder
from base_loader import model_manager

app = FastAPI(
    title="Hybrid RAG + Fine-Tuned LLM + Exact KYC API",
    description="Enterprise API combining deterministic SQL KYC retrieval, Hybrid Document RAG, and fine-tuned LLM conversational generation.",
    version="2.0.0"
)

# Global model state
tokenizer = None
model = None
device = "cpu"

class ChatRequest(BaseModel):
    query: str
    actor_role: Optional[str] = "officer"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 150
    chat_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    route: str
    response: str
    verified_sql_record: Optional[Dict[str, Any]] = None
    citations: Optional[List[str]] = []
    grounding_used: bool

@app.on_event("startup")
def startup_event():
    global tokenizer, model, device
    print("Initializing Hybrid API Service...")
    # Ingest project documents on startup
    doc_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "REPOSITORY_ANALYSIS_REPORT.md",
        PROJECT_ROOT / "docs" / "ARCHITECTURE_DIAGRAMS.md",
        PROJECT_ROOT / "01_data" / "clean_conversational_train.jsonl"
    ]
    hybrid_rag.ingest_folder_or_files([p for p in doc_paths if p.exists()])

    try:
        tokenizer, model, device = model_manager.load_inference_pipeline()
        print(f"Model successfully initialized on {device.upper()}")
    except Exception as e:
        print(f"Note: Model will load on-demand or running in fallback mode: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Hybrid RAG + KYC + LLM Gateway",
        "device": device,
        "model_initialized": (model is not None),
        "kyc_database_rows": 100000,
        "rag_indexed_chunks": len(hybrid_rag.indexed_docs)
    }

@app.get("/v1/kyc/{identifier}")
def direct_kyc_lookup(identifier: str, actor_role: str = "officer"):
    """Authoritative direct SQL lookup with RBAC."""
    result = kyc_service.lookup_record(identifier, actor_role=actor_role)
    if not result["found"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/v1/chat", response_model=ChatResponse)
def unified_chat(req: ChatRequest):
    """Main routing and generation endpoint."""
    ctx = context_builder.build_context(
        query=req.query,
        actor_role=req.actor_role,
        chat_history=req.chat_history
    )

    route = ctx["route"]
    sql_record = ctx["sql_record"]
    citations = [c["citation"] for c in ctx["rag_chunks"]]

    # Generation via LLM
    final_reply = ""

    if model is not None and tokenizer is not None:
        try:
            inputs = tokenizer(ctx["full_formatted_prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.15,
                    do_sample=(req.temperature > 0),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            raw_out = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            final_reply = raw_out.split("<|im_end|>")[0].strip()
        except Exception as e:
            final_reply = f"[Inference Error: {e}]"

    # Fallback to direct verified grounding if model generation is empty
    if not final_reply:
        if route == "EXACT_KYC" and sql_record:
            final_reply = (
                f"Verified Citizen Record for {sql_record.get('name')} (ID: {sql_record.get('person_id')}):\n"
                f"• Father: {sql_record.get('father_name')}, Mother: {sql_record.get('mother_name')}\n"
                f"• Age: {sql_record.get('age')}\n"
                f"• Aadhaar: {sql_record.get('aadhaar_number')}, PAN: {sql_record.get('pan_number')}"
            )
        elif route == "DOC_RAG" and ctx["rag_chunks"]:
            top = ctx["rag_chunks"][0]
            final_reply = f"{top['citation']}: {top['text'][:250]}..."
        else:
            final_reply = "I am here to assist with verified Citizen KYC lookups, technical documentation, or conversational chat."

    return ChatResponse(
        route=route,
        response=final_reply,
        verified_sql_record=sql_record,
        citations=citations,
        grounding_used=bool(ctx["grounding_text"])
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=False)

