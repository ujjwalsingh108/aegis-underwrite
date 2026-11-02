from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppSettings(BaseSettings):
    """Application-wide configuration settings and risk thresholds."""

    # Application & Infrastructure Metadata
    app_name: str = "aegis-underwrite"
    modal_volume_name: str = "aegis-underwrite-volume"
    environment: str = Field(default="development", description="Current deployment environment (dev, staging, prod)")

    # Frontend Decoupling & Security
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        description="CORS allowed origins for the React/Next.js frontend"
    )
    auth_jwt_issuer: Optional[str] = Field(
        default=None, 
        description="URL of the OAuth2 provider (e.g., Auth0, Clerk) for RBAC validation"
    )

    # Compute & Hardware Selection
    use_gpu: bool = True
    gpu_type: str = "A10G"
    cpu_cores: float = 2.0
    memory_mb: int = 4096

    # Agentic Models & Modalities
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_model_name: str = Field(
        default="meta-llama/Llama-3.1-405B-Instruct", 
        description="Massive open-weights model used strictly for Evol-Instruct synthetic data generation"
    )
    vision_model_name: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct", 
        description="Vision-Language model for KYC document tampering and video extraction"
    )
    lora_output_dir: str = "/vol/lora_adapters"

    # External Microservices (GraphRAG & MCP)
    vector_db_url: Optional[str] = Field(
        default=None, 
        description="Endpoint for Qdrant/Milvus GraphRAG knowledge base"
    )

    # Business Underwriting Thresholds
    max_dti_threshold: float = 0.43
    min_dscr_threshold: float = 1.25
    min_liquidity_months: float = 6.0
    max_fraud_probability: float = 0.15

    # Configuration for loading environment variables
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.example"), # Looks for .env first, falls back to example
        env_file_encoding="utf-8",
        extra="allow",
    )

# Global singleton instance for clean imports across the project
settings = AppSettings()



# # Pydantic BaseSettings for env & config mapping
# """
# src/common/config.py

# Centralized configuration management using Pydantic Settings.
# Reads default parameters and automatically overrides them via environment variables or .env files.
# """

# from pathlib import Path
# from pydantic_settings import BaseSettings, SettingsConfigDict

# class AppSettings(BaseSettings):
#     """Application-wide configuration settings and risk thresholds."""

#     # Application & Infrastructure Metadata
#     app_name: str = "aegis-underwrite"
#     modal_volume_name: str = "aegis-underwrite-volume"

#     # Compute & Hardware Selection (Set USE_GPU=True once card is linked)
#     use_gpu: bool = True
#     gpu_type: str = "A10G"
#     cpu_cores: float = 2.0
#     memory_mb: int = 4096

#     # Model & Adapter Paths
#     # When running on CPU, we use a lightweight instruction model or fallback to 1.5/3B
#     base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
#     lora_output_dir: str = "/vol/lora_adapters"

#     # Business Underwriting Thresholds
#     max_dti_threshold: float = 0.43
#     min_dscr_threshold: float = 1.25
#     min_liquidity_months: float = 6.0

#     # Configuration for loading environment variables
#     model_config = SettingsConfigDict(
#         env_file=".env.example",
#         env_file_encoding="utf-8",
#         extra="allow",  # Allows unexpected environment variables without throwing errors
#     )

# # Global singleton instance for clean imports across the project
# settings = AppSettings()