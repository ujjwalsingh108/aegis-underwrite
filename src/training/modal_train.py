"""
src/training/modal_train.py

Supervised Training Loop Execution on Modal

This script serves as the serverless orchestrator for the LoRA fine-tuning job.
In a production architecture, dataset generation (Evol-Instruct) and the actual 
training loop are decoupled. 

This job:
a. Verifies the existence of the pre-generated Train/Eval JSONL datasets on the network volume.
b. Triggers the highly optimized UnderwritingLoRATrainer (FlashAttention-2, bfloat16).
c. Commits the resulting adapter weights to the persistent Modal volume for inference.
"""

import os
from pathlib import Path
import modal

from src.common.config import settings
from src.serving.modal_app import app, app_volume, gpu_image

@app.function(
    image=gpu_image,
    gpu=settings.gpu_type,
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("underwriting_hf_token")],
    timeout=7200, # 2-hour ceiling for production training runs
)
def train_lora_job():
    """Executes the GPU-bound LoRA fine-tuning loop and commits adapter weights."""
    from src.training.lora_trainer import UnderwritingLoRATrainer
    
    print(f"--- Starting Production LoRA Training Job on GPU: {settings.gpu_type} ---")
    
    # 1. Verify Dataset Existence
    train_path = "/vol/dataset/train.jsonl"
    eval_path = "/vol/dataset/eval.jsonl"
    
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            "Training datasets not found on volume. "
            "Please run the dataset_generator.py Evol-Instruct pipeline first."
        )
        
    print(f"Found training data at {train_path} and eval data at {eval_path}.")

    # 2. Initialize the optimized trainer
    trainer = UnderwritingLoRATrainer()

    # 3. Execute the Training Loop
    # (Tokenization, FlashAttention-2 setup, and loss masking are handled inside this method)
    trainer.train(train_path=train_path, eval_path=eval_path)

    # 4. Commit persistent volume changes
    # This is the most critical step in serverless training. If the container shuts down
    # before committing, all trained LoRA weights saved to /vol will be permanently lost.
    print("Committing LoRA adapter weights to Modal Volume...")
    app_volume.commit()
    print("Commit successful! The adapter is now available for the vLLM inference engine.")

@app.local_entrypoint()
def main():
    """Trigger the remote training run from the local CLI."""
    print("Dispatching production LoRA training job to Modal...")
    train_lora_job.remote()
    print("Training job complete and volume committed.")