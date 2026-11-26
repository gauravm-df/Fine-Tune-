"""
Local PEFT/LoRA Fine-Tuning Script for Existing Pre-Trained Foundation Models.
Optimized for Apple Silicon Mac (MPS), CUDA, and CPU without needing 4-bit CUDA quantization.
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
sys.path.append(str(PROJECT_ROOT / "02_config"))
import config

def run_lora_training():
    print("=" * 65)
    print("STAGE: PEFT / LoRA FINE-TUNING ON EXISTING FOUNDATION MODEL")
    print(f"Base Model: {config.BASE_MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Adapter Output: {config.OUTPUT_DIR}")
    print("=" * 65)

    print("Loading base model tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model weights...")
    dtype = torch.bfloat16 if config.DEVICE in ["mps", "cuda"] else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True
    ).to(config.DEVICE)

    # Configure LoRA Adapter
    lora_cfg = LoraConfig(
        r=config.LORA_CONFIG["r"],
        lora_alpha=config.LORA_CONFIG["lora_alpha"],
        target_modules=config.LORA_CONFIG["target_modules"],
        lora_dropout=config.LORA_CONFIG["lora_dropout"],
        bias=config.LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading datasets from {config.TRAIN_FILE}...")
    dataset = load_dataset("json", data_files={
        "train": str(config.TRAIN_FILE),
        "val": str(config.VAL_FILE)
    })

    def tokenize_function(example):
        # Supports ChatML or raw text
        if "text" in example:
            text = example["text"]
        elif "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        else:
            text = example.get("prompt", "") + example.get("response", "")
        return tokenizer(text, truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_function, batched=False, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=config.TRAINING_ARGS["output_dir"],
        per_device_train_batch_size=config.TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps=config.TRAINING_ARGS["gradient_accumulation_steps"],
        learning_rate=config.TRAINING_ARGS["learning_rate"],
        lr_scheduler_type=config.TRAINING_ARGS["lr_scheduler_type"],
        warmup_ratio=config.TRAINING_ARGS["warmup_ratio"],
        max_steps=config.TRAINING_ARGS["max_steps"],
        logging_steps=config.TRAINING_ARGS["logging_steps"],
        save_steps=config.TRAINING_ARGS["save_steps"],
        save_total_limit=config.TRAINING_ARGS["save_total_limit"],
        bf16=(config.DEVICE == "mps" or (config.DEVICE == "cuda" and torch.cuda.is_bf16_supported())),
        fp16=False,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=data_collator
    )

    print("\nStarting LoRA Fine-Tuning...")
    trainer.train()

    print(f"\nSaving fine-tuned LoRA adapters to {config.OUTPUT_DIR}...")
    model.save_pretrained(str(config.OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.OUTPUT_DIR))
    print("LoRA Fine-Tuning Completed Successfully!")

if __name__ == "__main__":
    run_lora_training()

