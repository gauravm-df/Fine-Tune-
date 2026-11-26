"""
Google Colab / Cloud GPU QLoRA Fine-Tuning Script.
Uses 4-bit BitsAndBytes quantization for fine-tuning larger 3B, 7B, or 8B foundation models.
"""

from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Or "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./lora_adapter_qlora"

def run_qlora_training():
    print(f"Loading {BASE_MODEL_NAME} with 4-bit NF4 Quantization for GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": "finetune_train.jsonl"})

    def tokenize_func(ex):
        if "text" in ex:
            return tokenizer(ex["text"], truncation=True, max_length=512)
        return tokenizer(tokenizer.apply_chat_template(ex["messages"], tokenize=False), truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_func, batched=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=1000,
        logging_steps=20,
        save_steps=250,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("QLoRA Training Complete! Adapter saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_qlora_training()

