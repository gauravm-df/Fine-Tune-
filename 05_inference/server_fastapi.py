"""
FastAPI Server for Existing Fine-Tuned Model Serving.
Provides OpenAI-compatible Chat Completions endpoint.
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_config"))
import config

app = FastAPI(title="Existing Fine-Tuned LLM Service", version="1.0.0")

tokenizer = None
model = None

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.85

@app.on_event("startup")
def startup_event():
    global tokenizer, model
    device = config.DEVICE
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    if config.MERGED_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(config.MERGED_DIR), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(config.MERGED_DIR), torch_dtype=dtype, trust_remote_code=True).to(device)
    elif config.OUTPUT_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(config.OUTPUT_DIR), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)
        model = PeftModel.from_pretrained(base, str(config.OUTPUT_DIR))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

@app.get("/health")
def health():
    return {"status": "ok", "base_model": config.BASE_MODEL_NAME, "device": config.DEVICE}

@app.post("/v1/chat")
def chat(req: ChatRequest):
    global tokenizer, model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    messages = [
        {"role": "system", "content": "You are a friendly, natural English WhatsApp conversational companion."},
        {"role": "user", "content": req.prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(config.DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=(req.temperature > 0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=8000, reload=False)

