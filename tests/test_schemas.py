"""
This ensures data contracts reject invalid inputs, negative incomes, or missing required attributes
"""

import pytest  # type: ignore
from pydantic import ValidationError
from src.common.schemas import (
    ApplicantFinancials,
    ExtractedFinancialMetrics,
    UnderwritingDecision,
    UnderwritingMemo,
    AuditTrialEntry,
)


def test_valid_applicant_financials():
    applicant = ApplicantFinancials(
        applicant_id="corp_101",
        stated_monthly_income=75000.0,
        total_monthly_debt=15000.0,
        liquid_assets=120000.0,
        annual_operating_income=1000000.0,
        annual_debt_service=750000.0,
    )
    assert applicant.applicant_id == "corp_101"
    assert applicant.stated_monthly_income == 75000.0


def test_applicant_negative_income_raises_validation_error():
    with pytest.raises(ValidationError):
        ApplicantFinancials(
            applicant_id="corp_err",
            stated_monthly_income=-5000.0,
            total_monthly_debt=1000.0,
            liquid_assets=5000.0,
            annual_operating_income=100000.0,
            annual_debt_service=75000.0,
        )

def test_underwriting_memo_emission():
    memo = UnderwritingMemo(
        application_id="app_001",
        decision=UnderwritingDecision.APPROVE,
        calculated_dti=0.25,
        calculated_dscr=1.85,
        liquidity_runway_months=12.0,
        breached_policies=[],
        risk_factors=[],
        covenants=[],
        audit_trail=[
            AuditTrialEntry(
                step_name="Ingestion",
                status="SUCCESS",
                details="Payload received"
            )
        ]
    )
    assert memo.decision == "APPROVE"
    assert len(memo.audit_trail) == 1
