"""
Phase 6: Foundation Model Loader & Hardware Acceleration Manager.
Loads pre-trained instruction-tuned foundation models with MPS / CUDA / CPU acceleration.
"""

from pathlib import Path
from typing import Tuple, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = PROJECT_ROOT / "05_finetune" / "lora_adapter"
MERGED_DIR = PROJECT_ROOT / "04_merge" / "merged_model"

class ModelManager:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = self._detect_device()
        self.dtype = torch.bfloat16 if self.device in ["mps", "cuda"] else torch.float32

    def _detect_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_inference_pipeline(self) -> Tuple[Any, Any, str]:
        """Loads merged weights, adapter weights, or base model with fallback."""
        print(f"Loading LLM on device: {self.device.upper()} ({self.dtype})...")

        if MERGED_DIR.exists():
            print(f"Using Standalone Merged Model: {MERGED_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(MERGED_DIR),
                torch_dtype=self.dtype,
                trust_remote_code=True
            ).to(self.device)
        elif ADAPTER_DIR.exists():
            print(f"Using Base Model ({self.model_name}) + LoRA Adapter ({ADAPTER_DIR})")
            tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True
            ).to(self.device)
            model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        else:
            print(f"Using Base Foundation Model directly: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True
            ).to(self.device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return tokenizer, model, self.device

model_manager = ModelManager()

