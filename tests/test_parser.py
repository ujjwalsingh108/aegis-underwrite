"""
src/tests/test_parser.py

Document Parser Verification with Synthetic Fixtures

Before wiring into an API gateway, we need automated tests verifying that the 
parser can read PDF bytes, extract tables, and correctly aggregate the new 
multimodal compliance and fraud metrics.
"""

import pytest
from src.vision.parser import FinancialDocumentParser

@pytest.fixture
def parser():
    return FinancialDocumentParser()

def test_aggregate_metrics_variance_calculation(parser):
    """Tests if income variance percentage is accurately calculated and rounded."""
    metrics = parser.aggregate_metrics(
        stated_monthly_income=10000.0,
        verified_income=8500.0,
        detected_liabilities=3000.0,
        verified_reserves=15000.0
    )
    assert metrics.income_variance_pct == -15.0
    assert any("Income mismatch" in flag for flag in metrics.anomaly_flags)

def test_aggregate_metrics_fraud_flags(parser):
    """Tests compliance triggers (Sanctions, Photoshop metadata)."""
    metrics = parser.aggregate_metrics(
        stated_monthly_income=5000.0,
        verified_income=5000.0,
        detected_liabilities=1000.0,
        verified_reserves=10000.0,
        pep_or_sanctions_hit=True,
        document_tampering_score=0.25
    )
    assert len(metrics.anomaly_flags) == 2
    assert "CRITICAL:" in metrics.anomaly_flags[1]
    assert "document tampering" in metrics.anomaly_flags[0]

def test_aggregate_metrics_zero_stated_income(parser):
    """Boundary: Ensures zero stated income bypasses variance division to prevent crash."""
    metrics = parser.aggregate_metrics(
        stated_monthly_income=0.0,
        verified_income=5000.0,
        detected_liabilities=1000.0,
        verified_reserves=10000.0
    )
    assert metrics.income_variance_pct == 0.0