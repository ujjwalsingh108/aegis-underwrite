"""
API integration tests
"""

from fastapi.testclient import TestClient
from src.api.gateway import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_underwrite_endpoint_success():
    payload = {
        "financials": {
            "applicant_id": "app_api_test",
            "stated_monthly_income": 120000.0,
            "total_monthly_debt": 30000.0,
            "liquid_assets": 200000.0
        },
        "metrics": {
            "verified_monthly_income": 120000.0,
            "detected_monthly_liabilities": 30000.0,
            "verified_liquid_reserves": 200000.0,
            "income_variance_pct": 0.0,
            "anomaly_flags": []
        }
    }
    response = client.post("/underwrite", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["application_id"] == "app_api_test"
    assert data["decision"] == "APPROVE"
    assert data["calculated_dti"] == 0.25