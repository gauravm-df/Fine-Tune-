"""
Phase 7: Conversational LoRA Fine-Tuning Pipeline.
STRICT COMPLIANCE: Trains exclusively on clean, PII-sanitized WhatsApp dialogues.
Zero raw citizen KYC records are used in this pipeline.
"""

import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "01_data"
TRAIN_FILE = DATA_DIR / "clean_conversational_train.jsonl"
VAL_FILE = DATA_DIR / "clean_conversational_val.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "05_finetune" / "lora_adapter"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def run_conversational_lora():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    print("=" * 65)
    print("PHASE 7: CONVERSATIONAL LORA FINE-TUNING (WHATSAPP DIALOGUES ONLY)")
    print(f"Base Foundation Model: {BASE_MODEL}")
    print(f"Compute Acceleration:  {device.upper()}")
    print(f"Training Data Split:   {TRAIN_FILE}")
    print(f"Adapter Output Target: {OUTPUT_DIR}")
    print("=" * 65)

    if not TRAIN_FILE.exists():
        print(f"Error: Training file {TRAIN_FILE} not found. Run Phase 2 preprocessing first.")
        return

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading clean dataset...")
    dataset = load_dataset("json", data_files={
        "train": str(TRAIN_FILE),
        "val": str(VAL_FILE)
    })

    def tokenize_func(ex):
        if "text" in ex:
            return tokenizer(ex["text"], truncation=True, max_length=512)
        return tokenizer(tokenizer.apply_chat_template(ex["messages"], tokenize=False), truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_func, batched=False, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=500,
        logging_steps=25,
        save_steps=250,
        save_total_limit=2,
        bf16=(device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported())),
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("\nStarting LoRA Fine-Tuning...")
    trainer.train()

    print(f"\nSaving LoRA adapter weights to: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Fine-Tuning Complete! Persona adapter ready.")

if __name__ == "__main__":
    run_conversational_lora()

