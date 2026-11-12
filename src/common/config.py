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

    # Security & API Keys
    master_api_key: str = Field(
        default="dev-secret-key-change-in-prod", 
        description="Master API key for authenticating FastAPI requests"
    )

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
    memory_mb: int = 16384   # 16 GB Gigabytes of RAM - Sufficient for Base Engine (Qwen2.5-1.5B), Vision Engine (Qwen2-VL-2B-Instruct)

    # Agentic Models & Modalities
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vision_model_name: str = Field(
            default="Qwen/Qwen2-VL-2B-Instruct", 
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
    max_ltv_threshold: float = 0.80  # Maximum 80% Loan-to-Value

    # Configuration for loading environment variables
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.example"), # Looks for .env first, falls back to example
        env_file_encoding="utf-8",
        extra="allow",
    )

# Global singleton instance for clean imports across the project
settings = AppSettings()