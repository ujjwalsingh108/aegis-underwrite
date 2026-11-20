"""
Central, validated configuration for every Aegis component.

All environment variables are read here and nowhere else. Values are validated at import time,
so a malformed or insecure deployment fails fast at startup instead of mid-request.
"""

from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder values that must never be accepted outside local development.
_PLACEHOLDER_KEYS = {"MASTER_API_KEY", "UNDERWRITER_API_KEY", "AUDITOR_API_KEY", "", "changeme"}
_MIN_KEY_LENGTH = 16

Environment = Literal["development", "test", "staging", "production"]
_ENV_ALIASES = {"dev": "development", "local": "development", "prod": "production", "stage": "staging"}


class AppSettings(BaseSettings):
    """Application-wide configuration settings and risk thresholds."""

    # Application & Infrastructure Metadata
    app_name: str = "aegis-underwrite"
    modal_volume_name: str = "aegis-underwrite-volume"
    environment: Environment = Field(default="development", description="Deployment environment")
    log_level: str = Field(default="INFO", description="Root log level (DEBUG, INFO, WARNING, ...)")

    # Security & API Keys (RBAC)
    master_api_key: SecretStr = Field(default=SecretStr("MASTER_API_KEY"), description="Admin key")
    underwriter_api_key: SecretStr = Field(
        default=SecretStr("UNDERWRITER_API_KEY"), description="Underwriter key (may run underwriting)"
    )
    auditor_api_key: SecretStr = Field(
        default=SecretStr("AUDITOR_API_KEY"), description="Read-only compliance/audit key"
    )

    # Frontend Decoupling & Security
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"], description="CORS allowed origins (JSON list in env)"
    )
    auth_jwt_issuer: str | None = Field(
        default=None, description="Reserved: OAuth2 issuer URL for a future JWT-based get_current_user"
    )
    enable_docs: bool | None = Field(
        default=None, description="Expose /docs and /openapi.json. Defaults to off in production."
    )
    rate_limit_per_minute: int = Field(
        default=30, ge=1, description="Per-identity limit on underwriting calls (per container)"
    )
    max_upload_mb: float = Field(default=10.0, gt=0, le=50, description="Max PDF upload size")

    # Decision persistence (audit ledger)
    decision_store_path: str = Field(
        default="data/decisions.db", description="SQLite file holding the decision ledger"
    )

    # LLM advisory review
    llm_review_enabled: bool = Field(
        default=True, description="Ask the vLLM worker for an advisory second opinion"
    )
    llm_timeout_seconds: float = Field(default=300.0, gt=0, description="Incl. GPU cold start")

    # Compute & Hardware Selection
    use_gpu: bool = True
    gpu_type: str = "A10G"
    cpu_cores: float = 2.0
    memory_mb: int = 16384

    # Agentic Models & Modalities
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vision_model_name: str = Field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        description="Reserved: vision-language model for KYC document extraction (not wired yet)",
    )
    lora_output_dir: str = "/vol/lora_adapters"

    # Offline dataset generation (src/training/dataset_generator.py)
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"

    # External Microservices
    vector_db_url: str | None = Field(default=None, description="Reserved: GraphRAG knowledge base")

    # Business Underwriting Thresholds (policy - change only with credit-policy sign-off)
    max_dti_threshold: float = Field(default=0.43, gt=0, le=1)
    min_dscr_threshold: float = Field(default=1.25, gt=0)
    min_liquidity_months: float = Field(default=6.0, ge=0)
    max_fraud_probability: float = Field(default=0.15, ge=0, le=1)
    max_ltv_threshold: float = Field(default=0.80, gt=0, le=1.5)
    # Verified income below this fraction of stated income is a severe misrepresentation.
    severe_income_shortfall_ratio: float = Field(default=0.75, gt=0, le=1)
    # Verified income below this fraction of stated income raises an anomaly flag.
    income_mismatch_flag_ratio: float = Field(default=0.85, gt=0, le=1)
    # Document tampering score above this raises an anomaly flag.
    tampering_flag_threshold: float = Field(default=0.15, ge=0, le=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Unknown keys (e.g. HF_TOKEN in .env) are ignored instead of silently attached.
        extra="ignore",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def _normalise_environment(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            return _ENV_ALIASES.get(lowered, lowered)
        return value

    @model_validator(mode="after")
    def _validate_security(self) -> "AppSettings":
        keys = {
            "MASTER_API_KEY": self.master_api_key.get_secret_value(),
            "UNDERWRITER_API_KEY": self.underwriter_api_key.get_secret_value(),
            "AUDITOR_API_KEY": self.auditor_api_key.get_secret_value(),
        }
        # Duplicate keys would silently collapse roles in the key registry (privilege confusion).
        if len(set(keys.values())) != len(keys):
            raise ValueError("MASTER_API_KEY, UNDERWRITER_API_KEY and AUDITOR_API_KEY must differ.")

        if self.environment in ("staging", "production"):
            weak = [
                name
                for name, value in keys.items()
                if value in _PLACEHOLDER_KEYS or len(value) < _MIN_KEY_LENGTH
            ]
            if weak:
                raise ValueError(
                    f"Insecure API keys for environment '{self.environment}': {', '.join(weak)} "
                    f"must be set to random values of at least {_MIN_KEY_LENGTH} characters."
                )
            if "*" in self.allowed_origins:
                raise ValueError("ALLOWED_ORIGINS must not contain '*' outside development.")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def docs_enabled(self) -> bool:
        return self.enable_docs if self.enable_docs is not None else not self.is_production


# Global singleton instance for clean imports across the project
settings = AppSettings()
