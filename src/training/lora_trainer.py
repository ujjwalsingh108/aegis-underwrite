"""
This module sets up supervised fine-tuning with PEFT/LoRA so the Qwen model learns
to output structured, policy-compliant underwriting memos instead of generic conversational
prose.

This script prepares training arguments, configures LoRA target modules for attention
projections, formats conversational training pairs, and runs lightweight training on 
CPU or GPU.
"""

from dataclasses import dataclass
from typing import Dict, List
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from src.common.config import settings

@dataclass
class UnderwritingTrainingSample:
    """Structured training sample for supervised fine-tuning of the Qwen model."""
    instruction: str
    input_context: str
    expected_memo_json: str

class UnderwritingLoRATrainer:
    def __init__(self):
        self.model_name = settings.base_model_name
        self.output_dir = settings.lora_output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) # decoding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    """This function returns a LoRA configuration that tells PEFT how to inject trainable adapters into a pretrained LLM."""
    def get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )

    def prepare_peft_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cuda",
            low_gpu_mem_usage=True
        )
        lora_config = self.get_lora_config()
        peft_model = get_peft_model(model, lora_config)
        return peft_model

    def format_prompt(self, sample: UnderwritingTrainingSample) -> str:
        return (
            f"<|im_start|>system\n{sample.instruction}<|im_end|>\n"
            f"<|im_start|>user\n{sample.input_context}<|im_end|>\n"
            f"<|im_start|>assistant\n{sample.expected_memo_json}<|im_end|>"
        )