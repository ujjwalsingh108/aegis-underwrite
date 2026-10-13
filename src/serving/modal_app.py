"""
Defines the Modal container environment, installs CUDA drivers, PyTorch, 
and vLLM, and sets up network volumes.

Modal application definition and container environment configuration for GPU
accelerated serverless workloads.
"""

import modal
from src.common.config import settings

# 1. Define the persistent network volume for caching weights and adapters
app_volume = modal.Volume.from_name(
    settings.modal_volume_name, create_if_missing=True
)

# 2. Build the GPU container image
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "huggingface_hub>=0.24.0",
        "hf_transfer>=0.1.8",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 3. Instantiate the Modal App
app = modal.App(name=settings.app_name)