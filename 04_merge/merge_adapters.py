"""
LoRA Adapter Merge Script.
Fuses the fine-tuned LoRA adapter weights directly back into the base model weights.
Generates a standalone model in safetensors format with ZERO runtime inference overhead.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_config"))
import config

def merge():
    adapter_dir = config.OUTPUT_DIR
    merged_dir = config.MERGED_DIR

    if not adapter_dir.exists():
        print(f"Error: LoRA adapter directory {adapter_dir} does not exist. Please train first.")
        return

    print("=" * 65)
    print("MERGING LORA ADAPTER WEIGHTS WITH BASE MODEL")
    print(f"Base Model: {config.BASE_MODEL_NAME}")
    print(f"Adapter:    {adapter_dir}")
    print(f"Output:     {merged_dir}")
    print("=" * 65)

    print("Loading base model in float16/bfloat16...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging adapter weights into base model layers (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to {merged_dir}...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print("Merge complete! The standalone model is ready for zero-overhead inference.")

if __name__ == "__main__":
    merge()

