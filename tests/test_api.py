"""
src/tests/test_api.py

API integration tests updated for the enterprise multimodal underwriting schema.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_underwrite_endpoint_success():
    """
    Tests a flawless application using the new ApplicantProfile and 
    multimodal ExtractedFinancialMetrics schemas.
    """
    payload = {
        "profile": {  # Shifted from "financials" to "profile"
            "applicant_id": "APP-TEST-IND-01",
            "entity_type": "INDIVIDUAL",
            "requested_loan_amount": 500000.0,
            "stated_monthly_income": 120000.0,
            "total_monthly_debt": 30000.0,
            "liquid_assets": 200000.0,
            "stated_collateral_value": 650000.0
        },
        "metrics": {
            "verified_monthly_income": 120000.0,
            "detected_monthly_liabilities": 30000.0,
            "verified_liquid_reserves": 200000.0,
            "verified_collateral_value": 650000.0,
            "pep_or_sanctions_hit": False,
            "ubo_verified": True,
            "fraud_metrics": {
                "kyc_confidence_score": 0.98,
                "document_tampering_score": 0.01,
                "synthetic_voice_probability": 0.02
            },
            "income_variance_pct": 0.0,
            "anomaly_flags": []
        }
    }
    
    response = client.post("/underwrite", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Assert core enterprise fields
    assert data["application_id"] == "APP-TEST-IND-01"
    assert data["decision"] == "APPROVE"
    
    # Assert deterministic quantitative logic
    assert data["calculated_dti"] == 0.25      # 30,000 / 120,000
    assert data["approved_loan_amount"] == 500000.0
    
    # Assert fraud aggregation logic (max of tampering, 1-kyc, or voice)
    # max(0.01, 1 - 0.98, 0.02) = 0.02
    assert data["overall_fraud_probability"] == 0.02

def test_underwrite_endpoint_fraud_rejection():
    """
    Tests the waterfall decision routing to ensure severe OFAC/Sanctions 
    hits instantly trigger a REJECT_FRAUD response and nullify the loan.
    """
    payload = {
        "profile": {
            "applicant_id": "APP-TEST-LLC-02",
            "entity_type": "LLC",
            "requested_loan_amount": 1000000.0,
            "stated_monthly_income": 200000.0,
            "total_monthly_debt": 50000.0,
            "liquid_assets": 500000.0
        },
        "metrics": {
            "verified_monthly_income": 200000.0,
            "detected_monthly_liabilities": 50000.0,
            "verified_liquid_reserves": 500000.0,
            "pep_or_sanctions_hit": True,  # <--- Fatal compliance hit
            "ubo_verified": True,
            "fraud_metrics": {
                "kyc_confidence_score": 0.95,
                "document_tampering_score": 0.01,
                "synthetic_voice_probability": 0.05
            },
            "income_variance_pct": 0.0,
            "anomaly_flags": []
        }
    }
    
    response = client.post("/underwrite", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["decision"] == "REJECT_FRAUD"
    assert data["approved_loan_amount"] == 0.0
    assert any("OFAC/Sanctions" in breach for breach in data["breached_policies"])