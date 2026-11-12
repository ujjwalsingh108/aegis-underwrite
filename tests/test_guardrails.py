"""
src/tests/test_guardrails.py

Deterministic Guardrails

Now that the schemas, ratio math, and test foundations are mapped to the 
multimodal enterprise architecture, we enforce strict deterministic guardrails.
This ensures the pipeline never emits a hallucinated memo and always traps 
fatal regulatory/credit breaches before leaving the system.
"""

import pytest
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
    MultimodalFraudMetrics,
    UnderwritingDecision,
    EntityType
)
from src.engine.pipeline import UnderwritingPipeline

def test_guardrail_rejects_severe_income_discrepancy():
    """
    Ensures that a massive drop in verified income immediately triggers 
    a credit rejection and zeros out the approved loan amount.
    """
    pipeline = UnderwritingPipeline()
    profile = ApplicantProfile(
        applicant_id="APP-CREDIT-RISK-01",
        entity_type=EntityType.INDIVIDUAL,
        requested_loan_amount=500000.0,
        stated_monthly_income=100000.0,
        total_monthly_debt=20000.0,
        liquid_assets=500000.0,
    )
    
    # Verified income is 50% below stated income
    metrics = ExtractedFinancialMetrics(
        verified_monthly_income=50000.0,
        detected_monthly_liabilities=20000.0,
        verified_liquid_reserves=500000.0,
        income_variance_pct=-50.0,
        pep_or_sanctions_hit=False,
        ubo_verified=True,
        fraud_metrics=MultimodalFraudMetrics(
            kyc_confidence_score=0.99, 
            document_tampering_score=0.01
        ),
        anomaly_flags=["Severe income shortfall detected"]
    )
    
    memo = pipeline.run_preliminary_audit(profile, metrics)
    
    # Assert deterministic constraints
    assert memo.decision == UnderwritingDecision.REJECT_CREDIT
    assert memo.approved_loan_amount == 0.0
    assert any("Severe income variance" in risk for risk in memo.risk_factors)

def test_guardrail_flags_conditional_for_single_breach():
        """
        Ensures a healthy applicant with a single minor breach (tight liquidity runway)
        is routed to conditional approval with mandatory defensive covenants.
        """
        pipeline = UnderwritingPipeline()
        profile = ApplicantProfile(
            applicant_id="APP-TIGHT-CASH-02",
            entity_type=EntityType.LLC,           # <--- Back to LLC
            requested_loan_amount=250000.0,
            stated_monthly_income=0.0,            # (Not used for LLCs)
            total_monthly_debt=25000.0,               # (Not used for LLCs)
            liquid_assets=50000.0,                # Runway = 2 months (fails the 6.0 month min)
            annual_operating_income=1200000.0,    # <--- Added for DSCR math
            annual_debt_service=300000.0          # <--- DSCR = 4.0 (Healthy)
        )
        
        metrics = ExtractedFinancialMetrics(
            verified_monthly_income=0.0,          # (Not used for LLCs)
            detected_monthly_liabilities=25000.0,     # (Not used for LLCs)
            verified_liquid_reserves=50000.0,
            income_variance_pct=0.0,
            pep_or_sanctions_hit=False,
            ubo_verified=True,
            fraud_metrics=MultimodalFraudMetrics(
                kyc_confidence_score=0.95,
                document_tampering_score=0.02
            ),
            anomaly_flags=[],
        )
        
        memo = pipeline.run_preliminary_audit(profile, metrics)
        
        # Assert deterministic conditional logic
        assert memo.decision == UnderwritingDecision.CONDITIONAL_APPROVAL
        assert memo.approved_loan_amount == 250000.0
        assert len(memo.required_covenants) > 0
        # As an LLC, the pipeline should have dynamically added a UBO guarantee covenant
        assert any("Guarantee" in covenant for covenant in memo.required_covenants)

def test_guardrail_fatal_fraud_and_sanctions():
    """
    Ensures that an applicant with perfect financial ratios is still 
    instantly rejected if multimodal AI flags document tampering or OFAC hits.
    """
    pipeline = UnderwritingPipeline()
    profile = ApplicantProfile(
        applicant_id="APP-FRAUD-03",
        entity_type=EntityType.CORP,
        requested_loan_amount=1000000.0,
        stated_monthly_income=500000.0,  # Flawless ratios
        total_monthly_debt=10000.0,
        liquid_assets=9000000.0,
    )
    
    metrics = ExtractedFinancialMetrics(
        verified_monthly_income=500000.0,
        detected_monthly_liabilities=10000.0,
        verified_liquid_reserves=9000000.0,
        income_variance_pct=0.0,
        pep_or_sanctions_hit=True,       # FATAL: Sanctions list hit
        ubo_verified=False,
        fraud_metrics=MultimodalFraudMetrics(
            kyc_confidence_score=0.90,
            document_tampering_score=0.85 # FATAL: 85% chance of forged PDF
        ),
        anomaly_flags=[]
    )

    memo = pipeline.run_preliminary_audit(profile, metrics)
    
    # The mathematical engine must trap this before LLM evaluation
    assert memo.decision == UnderwritingDecision.REJECT_FRAUD
    assert memo.approved_loan_amount == 0.0
    assert memo.overall_fraud_probability == 0.85
    assert any("OFAC/Sanctions" in breach for breach in memo.breached_policies)