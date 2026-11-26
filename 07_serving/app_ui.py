"""
Phase 10: Streamlit Interactive Web Application.
Multi-role, interactive UI with route visualization, verified SQL inspection, and citations.
"""

from typing import Dict, Any
from pathlib import Path
import sys
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_kyc_db"))
sys.path.append(str(PROJECT_ROOT / "03_rag"))
sys.path.append(str(PROJECT_ROOT / "04_model"))
sys.path.append(str(PROJECT_ROOT / "06_orchestration"))

from context_builder import context_builder
from base_loader import model_manager
from hybrid_engine import hybrid_rag

st.set_page_config(
    page_title="Hybrid RAG + Fine-Tuned LLM",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Hybrid RAG + Fine-Tuned LLM")
st.caption("Deterministic SQL Citizen Verification • Hybrid Vector/BM25 Knowledge • Fine-Tuned Conversational Persona")

# Sidebar Controls
st.sidebar.header("⚙️ Model Settings")
role = "officer"
temperature = st.sidebar.slider("Sampling Temperature", 0.0, 1.0, 0.2, 0.05)
max_new_tokens = st.sidebar.slider("Max Generation Tokens", 32, 256, 128, 16)

# Load pipeline
@st.cache_resource
def load_all():
    doc_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "REPOSITORY_ANALYSIS_REPORT.md",
        PROJECT_ROOT / "docs" / "ARCHITECTURE_DIAGRAMS.md",
        PROJECT_ROOT / "01_data" / "clean_conversational_train.jsonl"
    ]
    hybrid_rag.ingest_folder_or_files([p for p in doc_paths if p.exists()])
    tok, mod, dev = model_manager.load_inference_pipeline()
    return tok, mod, dev

tokenizer, model, device = load_all()
st.sidebar.success(f"Device Acceleration: **{device.upper()}**")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am your Hybrid AI Assistant. I can help with verified Citizen KYC lookups, technical policy documents, or friendly chat. What can I do for you today?"}
    ]

# Render Chat History
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if user_query := st.chat_input("Ask a citizen KYC question, policy query, or casual chat..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 1. Orchestrate & Build Context
    ctx = context_builder.build_context(user_query, actor_role=role)
    sql_record = ctx["sql_record"]
    rag_chunks = ctx["rag_chunks"]

    # 2. Generate Reply
    with st.chat_message("assistant"):
        reply = ""
        if model is not None and tokenizer is not None:
            inputs = tokenizer(ctx["full_formatted_prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.15,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            reply = raw.split("<|im_end|>")[0].strip()

        if not reply and sql_record:
            reply = f"Retrieved record for **{sql_record.get('name')}** (ID: `{sql_record.get('person_id')}`)."

        st.markdown(reply)

        # Expanders for Grounding Data
        if sql_record:
            with st.expander("🔍 Authoritative Database Record (Zero Hallucination)"):
                st.json(sql_record)

        if rag_chunks:
            with st.expander("📚 Retrieved Knowledge Citations"):
                for c in rag_chunks:
                    st.markdown(f"**{c['citation']}** (RRF Score: `{c['rrf_score']}`)")
                    st.text(c['text'])

    st.session_state["messages"].append({"role": "assistant", "content": reply})

