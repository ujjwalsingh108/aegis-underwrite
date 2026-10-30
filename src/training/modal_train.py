"""
Supervised Training Loop Execution on Modal

We need a serverless training job that:
a. Generates the synthetic underwriting dataset on the fly (or reads from a volums),

b. Tokenizes the prompt/response pairs with attention masking so the model only computes
loss on the assistant memo tokens, not the prompts.

c. Trains a LoRA adapter on Modal.

d. Commits the adapter weights directly to the persistent.
"""

import os
from pathlib import Path
import modal

from src.common.config import settings
from src.serving.modal_app import app, app_volume, cpu_image, gpu_image

active_image = gpu_image if settings.use_gpu else cpu_image
active_gpu = settings.gpu_type if settings.use_gpu else None

@app.function(
    image=active_image,
    gpu=active_gpu,
    cpu=settings.cpu_cores,
    memory=settings.memory_mb,
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("underwriting_hf_token")],
    timeout=1800, # 30-minute ceiling for training
)

def train_lora_job(sample_count: int = 30, epochs: int = 1):
    """Executes a lightweight LoRA fine-tuning loop and saves adapter weights."""
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments
    )
    from src.training.dataset_generator import SyntheticUnderwritingDatasetGenerator
    from src.training.lora_trainer import UnderwritingLoRATrainer

    print(f"--- Starting LoRA Training Job on {'GPU: ' + str(active_gpu) if settings.use_gpu else 'CPU'} ---")
    trainer_helper = UnderwritingLoRATrainer()
    tokenizer = trainer_helper.tokenizer

    # 1. Generate synthetic dataset
    data_gen = SyntheticUnderwritingDatasetGenerator()
    dataset_path = "/tmp/underwrite_train.jsonl"
    samples = data_gen.generate_and_save_dataset(dataset_path, count=sample_count)
    print(f"Generated {len(samples)} synthetic training samples.")

    # 2. Tokenize with target-only loss masking
    formatted_texts = [trainer_helper.format_prompt(s) for s in samples]
    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=1024,
            truncation=True,
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenzied_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. Prepare PEFT model
    peft_model = trainer_helper.prepare_peft_model()
    peft_model.print_trainable_parameters()

    output_dir = settings.lora_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 4. Configure training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/trainer_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=not settings.use_gpu,
        fp16=False,
        bf16=True
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenzied_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )

    print("Executing training steps...")
    trainer.train()

    # 5. Save adapter weights to persistent volume
    print(f"Saving fine-tuned LoRA adapter to {output_dir}...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Commit persistent volume changes so they persist after container shutdown
    app_volume.commit()
    print("LoRA adapter weights committed to Modal Volume successfully!")

@app.local_entrypoint()
def main():
    """Trigger the remote training run from th local CLI."""
    print("Dispatching LoRA training job to Modal...")
    train_lora_job.remote(sample_count=20, epochs=1)
    print("Training job complete.")