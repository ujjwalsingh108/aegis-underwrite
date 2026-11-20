"""
src/tests/test_parser.py

Document parser and verification-metric aggregation.
"""

import pytest

from src.engine.verification import build_verified_metrics, income_variance_pct
from src.vision.parser import FinancialDocumentParser

VERIFIED = {"kyc_confidence_score": 0.95, "document_tampering_score": 0.01}


def test_parser_reads_pdf(tmp_path, blank_pdf_bytes):
    path = tmp_path / "doc.pdf"
    path.write_bytes(blank_pdf_bytes)
    parser = FinancialDocumentParser()
    assert parser.extract_text(str(path)) == ""
    assert parser.extract_tables(str(path)) == []


def test_aggregate_metrics_variance_calculation():
    """Tests if income variance percentage is accurately calculated and rounded."""
    metrics = build_verified_metrics(
        stated_monthly_income=10000.0,
        verified_income=8000.0,
        detected_liabilities=3000.0,
        verified_reserves=15000.0,
        **VERIFIED,
    )
    assert metrics.income_variance_pct == -20.0
    assert any("Income mismatch" in flag for flag in metrics.anomaly_flags)


def test_aggregate_metrics_fraud_flags():
    """Tests compliance triggers (Sanctions, Photoshop metadata)."""
    metrics = build_verified_metrics(
        stated_monthly_income=5000.0,
        verified_income=5000.0,
        detected_liabilities=1000.0,
        verified_reserves=10000.0,
        pep_or_sanctions_hit=True,
        kyc_confidence_score=0.95,
        document_tampering_score=0.25,
    )
    assert len(metrics.anomaly_flags) == 2
    assert "CRITICAL:" in metrics.anomaly_flags[1]
    assert "document tampering" in metrics.anomaly_flags[0]


def test_aggregate_metrics_zero_stated_income():
    """Boundary: Ensures zero stated income bypasses variance division to prevent crash."""
    assert income_variance_pct(0.0, 5000.0) == 0.0


def test_supplied_flags_are_kept_and_deduplicated():
    metrics = build_verified_metrics(
        stated_monthly_income=5000.0,
        verified_income=5000.0,
        detected_liabilities=1000.0,
        verified_reserves=10000.0,
        supplied_flags=["Missing tax transcript", "Missing tax transcript"],
        **VERIFIED,
    )
    assert metrics.anomaly_flags == ["Missing tax transcript"]


def test_verification_scores_are_required():
    with pytest.raises(TypeError):
        build_verified_metrics(  # type: ignore[call-arg]
            stated_monthly_income=1.0, verified_income=1.0,
            detected_liabilities=1.0, verified_reserves=1.0,
        )
