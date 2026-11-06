"""
Defines the Modal container environment, installs CUDA drivers, PyTorch,
and vLLM, and sets up network volumes.

Modal application definition and container environment configuration for GPU
accelerated serverless workloads. All ML inference is strictly constrained to GPUs.
"""

import modal
from modal import asgi_app
from src.common.config import settings

app_volume = modal.Volume.from_name(settings.modal_volume_name, create_if_missing=True)

# 1. Ultra-Lean Image for Web Gateways
gateway_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.112.2",
        "starlette==0.38.2",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.0",
        "requests>=2.31.0",
        "python-multipart==0.0.9"  # REQUIRED for FastAPI UploadFile (PDFs)
    )
    .add_local_python_source("src")
)

# 2. Heavy ML Image strictly for GPU Inference Workers
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",         
        "outlines",
        "torch>=2.4.0",
        "transformers>=4.48.2,<4.54.0",
        "accelerate>=0.33.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "pypdf>=4.2.0",
        "pdfplumber>=0.11.0",
        "huggingface_hub>=0.24.0",
    )
    .add_local_python_source("src")
)

app = modal.App(name=settings.app_name)

@app.function(
    image=gateway_image,  # Explicitly assign the gateway image
    volumes={settings.lora_output_dir: app_volume},  # Fix NameError: use app_volume
    secrets=[modal.Secret.from_name("aegis-api-secret")]
)
@asgi_app()
def web_endpoint():
    from src.api.main import app as fastapi_app
    return fastapi_app