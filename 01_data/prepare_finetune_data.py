"""
Dual-Domain Data Preparation Script for Existing LLM Fine-Tuning.
Blends:
1. WhatsApp Conversational Chat Dataset (14 Lakh corpus sample)
2. Citizen Identity & KYC Dataset (people_100k_records.csv)
Produces balanced multi-task training and validation splits in ChatML format.
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "01_data"

def prepare_blended_dataset(
    whatsapp_path: Path = DATA_DIR / "train_whatsapp_sample.jsonl",
    people_path: Path = DATA_DIR / "train_ready_100k.jsonl",
    output_train: Path = DATA_DIR / "finetune_train.jsonl",
    output_val: Path = DATA_DIR / "finetune_val.jsonl",
    whatsapp_samples: int = 25000,
    people_samples: int = 25000
):
    all_samples = []

    # 1. Load WhatsApp chat samples
    if whatsapp_path.exists():
        print(f"Loading up to {whatsapp_samples} samples from WhatsApp dataset: {whatsapp_path.name}...")
        count = 0
        with open(whatsapp_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= whatsapp_samples:
                    break
                try:
                    all_samples.append(json.loads(line.strip()))
                    count += 1
                except Exception:
                    continue
        print(f"Loaded {count} WhatsApp conversation samples.")

    # 2. Load Citizen Identity & KYC samples
    if people_path.exists():
        print(f"Loading up to {people_samples} samples from People 100K KYC dataset: {people_path.name}...")
        count = 0
        with open(people_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= people_samples:
                    break
                try:
                    all_samples.append(json.loads(line.strip()))
                    count += 1
                except Exception:
                    continue
        print(f"Loaded {count} Citizen Identity records.")

    print(f"Total blended dataset size: {len(all_samples)} samples.")
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.95)
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]

    with open(output_train, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_val, "w", encoding="utf-8") as f:
        for r in val_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated:")
    print(f"  -> Training Split:   {len(train_data)} samples in {output_train.name}")
    print(f"  -> Validation Split: {len(val_data)} samples in {output_val.name}")

if __name__ == "__main__":
    prepare_blended_dataset()
