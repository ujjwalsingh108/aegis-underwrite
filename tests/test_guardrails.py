"""
Deterministic guardrails

Now the schemas, ratio math, and test foundation are solid,
we need to enforce strict guardrails so the pipeline never emits a
hallucinated or out-of-policy memo.

This ensures that any output produced by the engine satisfies regulatory and underwriting
policies before leaving the system.
"""

import pytest # type: ignore
from src.common.schemas import (
    ApplicantFinancials,
    ExtractedFinancialMetrics,
    UnderwritingDecision
)
from src.engine.pipeline import UnderwritingPipeline

def test_guardrail_rejects_severe_income_discrepancy():
    pipeline = UnderwritingPipeline()
    financials = ApplicantFinancials(
        applicant_id="app_tampered",
        stated_monthly_income=100000.0,
        total_monthly_debt=20000.0,
        liquid_assets=50000.0,
    )
    # Verified income is 50% below stated income (-50.0%)
    metrics = ExtractedFinancialMetrics(
        verified_monthly_income=50000.0,
        detected_monthly_liabilities=20000.0,
        verified_liquid_reserves=50000.0,
        income_variance_pct=-50.0,
        anomaly_flags=["Severe income shortfall detected"]
    )
    memo = pipeline.run_preliminary_audit(financials, metrics)
    assert memo.decision == UnderwritingDecision.REJECT
    assert "Severe income shortfall detected" in memo.risk_factors

def test_guardrail_flags_conditional_for_single_breach():
    pipeline = UnderwritingPipeline()
    financials = ApplicantFinancials(
        applicant_id="app_tight_liquidity",
        stated_monthly_income=100000.0,
        total_monthly_debt=25000.0, # DTI = 25% (healthy)
        liquid_assets=50000.0   # Runway = 2 months (below 6 month requirement)
    )
    metrics = ExtractedFinancialMetrics(
        verified_monthly_income=100000.0,
        detected_monthly_liabilities=25000.0,
        verified_liquid_reserves=50000.0,
        income_variance_pct=0.0,
        anomaly_flags=[],
    )

    memo = pipeline.run_preliminary_audit(financials, metrics)
    assert memo.decision == UnderwritingDecision.CONDITIONAL
    assert len(memo.covenants)
