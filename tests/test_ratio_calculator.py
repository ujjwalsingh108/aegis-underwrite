"""
This verifies standard ratio mathematics, policy breach flags, and boundary edge cases(zero income, zero liabilities)
"""

import pytest # type: ignore
from src.common.schemas import ApplicantFinancials
from src.engine.ratio_calculator import (
    calculate_dti,
    calculate_dscr,
    calculate_liquidity_runway,
    evaluate_ratios
)

def test_calculate_dti_standard():
    assert calculate_dti(monthly_debt=4000.0, monthly_income=12000.0) == 0.3333

def test_calculate_dti_zero_income():
    assert calculate_dti(monthly_debt=1000.0, monthly_income=0.0) == float("inf")

def test_calculate_dscr_standard():
    assert calculate_dscr(net_operating_income=150000.0, total_debt_service=100000.0) == 1.50

def test_calculate_dscr_none_handling():
    assert calculate_dscr(net_operating_income=None, total_debt_service=100000.0) is None
    assert calculate_dscr(net_operating_income=150000.0, total_debt_service=None) is None

def test_calculate_liquidity_runway():
    assert calculate_liquidity_runway(liquid_assets=30000.0, monthly_obligations=5000.0) == 6.0
    assert calculate_liquidity_runway(liquid_assets=0.0, monthly_obligations=5000.0) == 0.0

def test_evaluate_ratio_pass():
    financials = ApplicantFinancials(
        applicant_id="app_valid",
        stated_monthly_income=150000.0,
        total_monthly_debt=30000.0,
        liquid_assets=250000.0,
        annual_operating_income=2400000.0,
        annual_debt_service=1000000.0
    )
    result = evaluate_ratios(financials)
    assert result["status"] == "PASS"
    assert len(result["breached_policies"]) == 0
    assert result["calculated_dti"] == 0.20

def test_evaluate_ratios_breaches():
    financials = ApplicantFinancials(
        applicant_id="app_risk",
        stated_monthly_income=50000.0,
        annual_operating_income=600000.0,
        annual_debt_service=420000.0,
        total_monthly_debt=35000.0, # 70% DTI -> exceeds default 43%
        liquid_assets=10000.0,       # < 1 month runway -> breaches 6 months threshold
    )
    result = evaluate_ratios(financials)
    assert result["status"] == "FAIL"
    assert len(result["breached_policies"]) >= 2