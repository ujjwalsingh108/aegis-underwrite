# Pydantic BaseSettings for env & config mapping
"""
src/common/config.py

Centralized configuration management using Pydantic Settings.
Reads default parameters and automatically overrides them via environment variables or .env files.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """Application-wide configuration settings and risk thresholds."""

    # Application & Infrastructure Metadata
    app_name: str = "aegis-underwrite"
    modal_volume_name: str = "aegis-underwrite-volume"

    # Model & Adapter Paths
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_output_dir: str = "/vol/lora_adapters"

    # Business Underwriting Thresholds
    max_dti_threshold: float = 0.43
    min_dscr_threshold: float = 1.25
    min_liquidity_months: float = 6.0

    # Configuration for loading environment variables
    model_config = SettingsConfigDict(
        env_file=".env.example",
        env_file_encoding="utf-8",
        extra="allow",  # Allows unexpected environment variables without throwing errors
    )

# Global singleton instance for clean imports across the project
settings = AppSettings()