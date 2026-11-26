"""
Interactive CLI Chat for Existing Fine-Tuned Model (LoRA or Merged).
Automatically detects whether to load the merged weights or dynamic LoRA adapter.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_config"))
import config

def load_model_and_tokenizer():
    merged_dir = config.MERGED_DIR
    adapter_dir = config.OUTPUT_DIR

    device = config.DEVICE
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    if merged_dir.exists():
        print(f"Loading standalone merged weights from {merged_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir),
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
    elif adapter_dir.exists():
        print(f"Loading base model ({config.BASE_MODEL_NAME}) + LoRA adapter ({adapter_dir})...")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        print(f"Loading base model directly ({config.BASE_MODEL_NAME}) for testing...")
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)

    model.eval()
    return tokenizer, model, device

def main():
    tokenizer, model, device = load_model_and_tokenizer()

    print("=" * 65)
    print(f"EXISTING LLM CHAT TERMINAL (Running on {device})")
    print("Type 'exit' or 'quit' to end chat.")
    print("=" * 65)

    system_message = "You are a friendly, natural English WhatsApp conversational companion."

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    temperature=0.2,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )

            reply = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            print(f"AI: {reply}")
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

if __name__ == "__main__":
    main()

