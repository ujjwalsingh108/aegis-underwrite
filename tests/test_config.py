"""Startup configuration guardrails."""

import pytest
from pydantic import ValidationError

from src.common.config import AppSettings

STRONG = {
    "master_api_key": "m" * 20,
    "underwriter_api_key": "u" * 20,
    "auditor_api_key": "a" * 20,
}


def _settings(**kwargs) -> AppSettings:
    return AppSettings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_production_rejects_placeholder_keys():
    placeholders = {
        "master_api_key": "MASTER_API_KEY",
        "underwriter_api_key": "UNDERWRITER_API_KEY",
        "auditor_api_key": "short",
    }
    with pytest.raises(ValidationError, match="Insecure API keys"):
        _settings(environment="production", **placeholders)


def test_production_accepts_strong_keys_and_disables_docs():
    settings = _settings(environment="prod", **STRONG)
    assert settings.environment == "production"
    assert settings.docs_enabled is False


def test_duplicate_keys_rejected_everywhere():
    with pytest.raises(ValidationError, match="must differ"):
        _settings(**{**STRONG, "auditor_api_key": STRONG["master_api_key"]})


def test_wildcard_cors_rejected_outside_development():
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        _settings(environment="staging", allowed_origins=["*"], **STRONG)


def test_secrets_are_not_exposed_in_repr():
    assert STRONG["master_api_key"] not in repr(_settings(**STRONG))
