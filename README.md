# Aegis-Underwrite Architecture

## Production Enterprise Challenge: _Real-Time Multimodal Voice & Document AI Underwriting Engine_

In enterprise AI architecture, the industry frontier has moved beyond basic text RAG into asynchronous streaming, specialized fine-tuning, serverless GPU infrastructure, and multi-tier latency optimization.

This challenge requires you to build, train, benchmark, and deploy an Autonomous Multimodal Loan & Policy Underwriting Engine completely on Modal.com using serverless GPU containers, vLLM, custom LoRA adaptation, and structured vision-language pipelines.

## System Architecture

```text
                       [Incoming Client / Webhook]
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │         Modal Fastify/ASGI Gateway        │
             │         (Scale-to-Zero Entrypoint)        │
             └─────────────────────┬─────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
 [Modal CPU Volume Worker]                    [Modal Serverless A10G/L4]
  Document OCR & Layout Parser                 Custom LoRA Domain Adapter
  • Surya / Docling / PDF-Extract              • DeepSeek-R1-Distill / Qwen2.5-7B
  • Bounding box financial tables              • vLLM Engine with PagedAttention
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Async Worker & Task Pipeline        │
             │   (Modal Shared Network Volumes + Queue)  │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │         Deterministic Guardrail &         │
             │            Verification Engine            │
             │  • Out-of-bounds ratio checks             │
             │  • Strict Pydantic JSON Schema Emission   │
             │  • Real-time WebSocket streaming audit    │
             └───────────────────────────────────────────┘

```

## Setting up virtual environment

- Create the virtual environment - `python -m venv .venv`

- Activate it in PowerShell - `.venv\Scripts\Activate.ps1`

## Installing dependencies in the pyproject.toml file

To install the current folder containing __pyproject.toml__, pass __.__ (the dot represents the current directory): `pip install -e .`

## Why we added `__init__.py` in the `src/` folder

- __Package Marker:__ In Python, placing `__init__.py` inside a folder explicitly marks it as a regular importable Python package rather than just an arbitrary file system directory.

- __Build Backend Recognition:__ Build tools like `hatchling` look for package identifiers when packing wheels. Because we specified `packages = ["src"]`, having `src/__init__.py` guarantees Hatchling treats `src` as the root distribution package without throwing packaging errors.

- __Clean Root-Level Imports:__ Once installed in editable mode (`pip install -e .`), any module or test in your repo can import cleanly from anywhere using absolute paths (for example, `from src.common.config import settings` or `from src.engine.pipeline import UnderwritingPipeline`) without needing fragile `sys.path.insert()` hacks.

## Significance of `config.py`

- `Decoupling Code from Parameters:` Hardcoding values like model names (`"Qwen/Qwen2.5-7B-Instruct"`), file paths, or numeric thresholds directly into deep pipeline files makes maintenance brittle. If you decide to evaluate a different model or change a threshold, you only change it in one place.

- `Environment Variable Injection:` Cloud and serverless platforms (like Modal, Docker, or Kubernetes) pass secrets and dynamic configurations via system environment variables. A proper config module reads these variables automatically.

- `Type Safety & Early Failure:` Using Pydantic for configuration guarantees that if an environment variable is malformed (e.g., passing `"abc"` for a numeric ratio threshold), the application will crash during startup with a descriptive error instead of failing hours into a long GPU workload.

- `Separation of Secrets:` Sensitive credentials (like API keys) reside in a local `.env` file (ignored by Git), while the config file defines how the application securely ingests them.