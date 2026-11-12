"""
src/tests/test_api.py

API integration tests updated for the enterprise multimodal underwriting schema.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app
from src.common.config import settings

client = TestClient(app)

@pytest.fixture
def mock_llm_engine():
    with patch("src.api.main.UnderwritingLLMEngine") as mock:
        instance = mock.return_value
        
        # Mock Modal's .remote() method
        mock_remote = MagicMock()
        mock_remote.return_value = {
            "application_id": "APP-111",
            "decision": "APPROVE",
            "calculated_dti": 0.35,
            "calculated_dscr": 0.0,
            "liquidity_runway_months": 12,
            "breached_policies": [],
            "risk_factors": ["None"],
            "required_covenants": [],
            "audit_trail": []
        }
        instance.generate_memo.remote = mock_remote
        yield mock_remote

def test_underwrite_endpoint_success(mock_llm_engine):
    """Happy Path: Complete HTTP Request mapping to Modal Engine"""
    payload = {
        "applicant_id": "APP-111",
        "stated_monthly_income": 10000.0,
        "verified_monthly_income": 9500.0,
        "total_monthly_debt": 3000.0,
        "liquid_reserves": 50000.0,
        "anomaly_flags": []
    }
    # Note: If API key is strictly required via Depends(), attach it here
    headers = {"X-API-Key": settings.master_api_key} 
    
    response = client.post("/underwrite", json=payload, headers=headers)
    
    assert response.status_code == 200
    assert response.json()["decision"] == "APPROVE"
    mock_llm_engine.assert_called_once()

def test_underwrite_endpoint_missing_fields():
    """Negative: FastAPI automatically rejects payloads missing required Pydantic fields."""
    payload = {
        "applicant_id": "APP-111",
        # Missing income/debt fields
    }
    headers = {"X-API-Key": settings.master_api_key} 
    response = client.post("/underwrite", json=payload, headers=headers)
    assert response.status_code == 422