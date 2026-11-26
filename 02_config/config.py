"""
Central Configuration for Existing Foundation Model Fine-Tuning (PEFT/LoRA).
Supports lightweight models suitable for local Apple Silicon Mac (MPS) as well as Colab GPU.
"""

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Recommended Base Models:
# 1. Qwen/Qwen2.5-0.5B-Instruct  (Extremely fast on MacBook Air/Pro, 0.5B params, Native ChatML)
# 2. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params, very popular for fast edge tuning)
# 3. Qwen/Qwen2.5-1.5B-Instruct  (Higher quality, runs comfortably with 8GB-16GB RAM)
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Paths
DATA_DIR = PROJECT_ROOT / "01_data"
OUTPUT_DIR = PROJECT_ROOT / "03_finetune" / "lora_adapter"
MERGED_DIR = PROJECT_ROOT / "04_merge" / "merged_model"

TRAIN_FILE = DATA_DIR / "finetune_train.jsonl"
VAL_FILE = DATA_DIR / "finetune_val.jsonl"

# Hardware detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# LoRA Hyperparameters (Parameter-Efficient Fine-Tuning)
LORA_CONFIG = {
    "r": 16,                         # Rank of update matrices
    "lora_alpha": 32,                # Scaling factor
    "target_modules": [              # Attention projection layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Hyperparameters
TRAINING_ARGS = {
    "output_dir": str(OUTPUT_DIR),
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "max_steps": 500,                # Set to 500-1500 for fast local runs, or num_train_epochs=1
    "logging_steps": 25,
    "save_steps": 250,
    "save_total_limit": 2,
    "fp16": False,                   # False on Mac MPS (uses bfloat16 or float32)
    "bf16": (DEVICE == "mps" or (DEVICE == "cuda" and torch.cuda.is_bf16_supported())),
    "report_to": "none"
}

