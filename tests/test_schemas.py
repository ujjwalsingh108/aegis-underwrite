"""
src/tests/test_schemas.py

Deterministic Guardrails for Data Contracts

This ensures Pydantic strictly rejects invalid inputs, missing required attributes, 
or hallucinated enumerations before they can enter the mathematical engine or vLLM.
"""

import pytest
from pydantic import ValidationError
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
    UnderwritingDecision,
    UnderwritingMemo,
    AuditTrailEntry,
    EntityType
)

def test_valid_applicant_profile():
    """Ensures a fully populated profile passes validation cleanly."""
    applicant = ApplicantProfile(
        applicant_id="APP-CORP-101",
        entity_type=EntityType.CORP,
        requested_loan_amount=2000000.0,
        stated_monthly_income=75000.0,
        total_monthly_debt=15000.0,
        liquid_assets=120000.0,
        stated_collateral_value=2500000.0
    )
    assert applicant.applicant_id == "APP-CORP-101"
    assert applicant.stated_monthly_income == 75000.0
    assert applicant.entity_type == "CORP"

def test_applicant_invalid_entity_raises_validation_error():
    """
    Ensures that hallucinated or unsupported entity structures (e.g., NON_PROFIT) 
    are trapped by the Pydantic Enum contract before processing.
    """
    with pytest.raises(ValidationError):
        ApplicantProfile(
            applicant_id="APP-ERR-02",
            entity_type="NON_PROFIT",  # Invalid: Not in EntityType Enum
            requested_loan_amount=50000.0,
            stated_monthly_income=5000.0,
            total_monthly_debt=1000.0,
            liquid_assets=5000.0,
        )

def test_underwriting_memo_emission():
    """
    Ensures the final output memo successfully serializes all new enterprise 
    fields including fraud scores, covenants, and audit trails.
    """
    memo = UnderwritingMemo(
        application_id="APP-001",
        decision=UnderwritingDecision.CONDITIONAL_APPROVAL,
        calculated_dti=0.25,
        calculated_dscr=1.85,
        calculated_ltv=0.75,
        liquidity_runway_months=12.0,
        overall_fraud_probability=0.01,
        approved_loan_amount=2000000.0,
        breached_policies=[],
        risk_factors=[],
        required_covenants=["Personal Guarantee required"],
        audit_trail=[
            AuditTrailEntry(
                step_name="Multimodal_Ingestion",
                agent_role="FastAPI_Gateway",
                status="SUCCESS",
                details="Payload and documents received"
            )
        ]
    )
    assert memo.decision == "CONDITIONAL_APPROVAL"
    assert memo.approved_loan_amount == 2000000.0
    assert len(memo.audit_trail) == 1
    assert memo.audit_trail[0].agent_role == "FastAPI_Gateway"