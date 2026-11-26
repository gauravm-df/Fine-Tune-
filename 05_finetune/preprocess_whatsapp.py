"""
Phase 2: Conversational Dataset Cleaning, PII Sanitization & Normalization.
STRICT COMPLIANCE: Uses ONLY authorized conversational dialogues (WhatsApp).
Zero raw KYC citizen records are permitted in fine-tuning data.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "01_data"

# PII Regex Patterns
REGEX_PHONE = re.compile(r'(\+?91[\-\s]?)?[6789]\d{9}')
REGEX_EMAIL = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
REGEX_AADHAAR = re.compile(r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b')
REGEX_PAN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
REGEX_CREDIT_CARD = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

def sanitize_pii(text: str) -> str:
    """Masks all sensitive personal identifiable information."""
    if not isinstance(text, str):
        return ""
    text = REGEX_PHONE.sub("[PHONE_REDACTED]", text)
    text = REGEX_EMAIL.sub("[EMAIL_REDACTED]", text)
    text = REGEX_AADHAAR.sub("[AADHAAR_REDACTED]", text)
    text = REGEX_PAN.sub("[PAN_REDACTED]", text)
    text = REGEX_CREDIT_CARD.sub("[CARD_REDACTED]", text)
    return text.strip()

def normalize_text(text: str) -> str:
    """Normalizes whitespace, unicode characters, and common chat artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")
    return text.strip()

def process_conversations(
    source_path: Path,
    output_train: Path,
    output_val: Path,
    limit: int = 50000,
    val_ratio: float = 0.05
) -> Tuple[int, int, int]:
    print(f"Reading conversational dataset from: {source_path}")
    seen_hashes = set()
    cleaned_records: List[Dict[str, Any]] = []

    with open(source_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(cleaned_records) >= limit:
                break
            try:
                data = json.loads(line.strip())
            except Exception:
                continue

            # Extract user & assistant messages
            user_msg = ""
            assistant_msg = ""

            if "messages" in data and isinstance(data["messages"], list):
                for m in data["messages"]:
                    if m.get("role") == "user":
                        user_msg = m.get("content", "")
                    elif m.get("role") == "assistant":
                        assistant_msg = m.get("content", "")
            elif "text" in data:
                # Parse ChatML format
                text = data["text"]
                user_match = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', text, re.DOTALL)
                asst_match = re.search(r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)', text, re.DOTALL)
                if user_match:
                    user_msg = user_match.group(1).strip()
                if asst_match:
                    assistant_msg = asst_match.group(1).strip()

            if not user_msg or not assistant_msg:
                continue

            # Check for PII and sanitize
            clean_user = normalize_text(sanitize_pii(user_msg))
            clean_asst = normalize_text(sanitize_pii(assistant_msg))

            # Deduplication
            content_hash = hashlib.sha256(f"{clean_user}|{clean_asst}".encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            record = {
                "messages": [
                    {"role": "system", "content": "You are a friendly, natural English WhatsApp conversational companion."},
                    {"role": "user", "content": clean_user},
                    {"role": "assistant", "content": clean_asst}
                ],
                "text": (
                    f"<|im_start|>system\nYou are a friendly, natural English WhatsApp conversational companion.<|im_end|>\n"
                    f"<|im_start|>user\n{clean_user}<|im_end|>\n"
                    f"<|im_start|>assistant\n{clean_asst}<|im_end|>"
                )
            }
            cleaned_records.append(record)

    total_valid = len(cleaned_records)
    split_point = int(total_valid * (1.0 - val_ratio))
    train_split = cleaned_records[:split_point]
    val_split = cleaned_records[split_point:]

    with open(output_train, "w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_val, "w", encoding="utf-8") as f:
        for r in val_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"PII Sanitization & Deduplication Complete:")
    print(f"  • Total Unique Conversational Dialogues: {total_valid}")
    print(f"  • Clean Train Split: {len(train_split)} records -> {output_train.name}")
    print(f"  • Clean Val Split:   {len(val_split)} records -> {output_val.name}")
    return total_valid, len(train_split), len(val_split)

if __name__ == "__main__":
    src = DATA_DIR / "train_whatsapp_sample.jsonl"
    out_tr = DATA_DIR / "clean_conversational_train.jsonl"
    out_vl = DATA_DIR / "clean_conversational_val.jsonl"
    process_conversations(src, out_tr, out_vl, limit=30000)

