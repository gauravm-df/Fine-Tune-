"""
Streamlit Web Interface for Fine-Tuned Existing Foundation Model.
"""

import sys
from pathlib import Path
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_config"))
import config

st.set_page_config(
    page_title="Existing LLM (Fine-Tuned) Chat",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Fine-Tuned Foundation Model Chat")
st.caption(f"Base: {config.BASE_MODEL_NAME} | Fine-Tuned via LoRA on WhatsApp Dataset")

@st.cache_resource
def load_model():
    merged_dir = config.MERGED_DIR
    adapter_dir = config.OUTPUT_DIR
    device = config.DEVICE
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    if merged_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(str(merged_dir), torch_dtype=dtype, trust_remote_code=True).to(device)
    elif adapter_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)
        mod = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)

    mod.eval()
    return tok, mod, device

tokenizer, model, device = load_model()
st.sidebar.title("Settings")
st.sidebar.info(f"Compute Device: **{device.upper()}**")
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
max_toks = st.sidebar.slider("Max Tokens", 32, 256, 128, 16)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "Hello! I am your fine-tuned foundation model. Ask me anything!"}
    ]

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    messages = [
        {"role": "system", "content": "You are a friendly, natural English WhatsApp conversational companion."},
        {"role": "user", "content": prompt}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_toks,
            temperature=temp,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=(temp > 0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

