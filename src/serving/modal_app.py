"""
Defines the Modal container environment, installs CUDA drivers, PyTorch,
and vLLM, and sets up network volumes.

Modal application definition and container environment configuration for GPU
accelerated serverless workloads.

When using CPU, vllm won't run efficiently or may fail without CUDA. Instead,
we use Hugging Face transformers with torch configured to CPU inference, which
switches to vllm automatically when GPU is enabled.
"""

import modal
from src.common.config import settings

# 1. Define the persistent network volume for caching weights and adapters
app_volume = modal.Volume.from_name(settings.modal_volume_name, create_if_missing=True)

# 2. CPU runtime image (No CUDA required, builds quickly, completely free)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.48.2,<4.54.0",
        "accelerate>=0.33.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.0",
        "pypdf>=4.2.0",
        "pdfplumber>=0.11.0",
        "huggingface_hub>=0.24.0",
        "faker>=24.0.0",
    )
    .add_local_python_source("src")
)

# GPU vLLM image (Used when settings.use_gpu is True)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",         
        "outlines", # <--- Add outlines explicitly for JSON constraint mapping
        "torch>=2.4.0",
        "transformers>=4.48.2,<4.54.0",
        "accelerate>=0.33.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.0",
        "pypdf>=4.2.0",
        "pdfplumber>=0.11.0",
        "huggingface_hub>=0.24.0",
        "faker>=24.0.0",
    )
    .add_local_python_source("src")
)

# 3. Instantiate the Modal App
app = modal.App(name=settings.app_name)

@app.function(
    image=cpu_image,
    keep_warm=1, # Keep one CPU container alive for instant HTTP routing # type: ignore
)

@modal.asgi_app()
def api():
    """Mounts the FastAPI gateway to a public serverless endpoint."""
    from src.api.gateway import app as fastapi_app
    return fastapi_app

