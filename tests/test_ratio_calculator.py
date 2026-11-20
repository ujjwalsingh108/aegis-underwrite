"""
src/tests/test_ratio_calculator.py

This verifies standard ratio mathematics, policy breach flags, and boundary edge 
cases (zero income, zero liabilities) mapping to the multimodal enterprise schema.
"""

from src.engine.ratio_calculator import evaluate_ratios
from src.engine.verification import build_verified_metrics


def test_evaluate_ratios_individual_dti(valid_individual_profile):
    """Happy Path: DTI calculation for INDIVIDUAL"""
    ratios = evaluate_ratios(valid_individual_profile)
    assert ratios["calculated_dti"] == 0.35  # 3500 / 10000
    assert ratios["calculated_ltv"] is None  # No collateral provided

def test_evaluate_ratios_llc_dscr_and_ltv(valid_llc_profile):
    """Happy Path: DSCR and LTV for LLC"""
    ratios = evaluate_ratios(valid_llc_profile)
    assert ratios["calculated_dscr"] == 3.3333  # 800000 / 240000 (rounded)
    assert ratios["calculated_ltv"] == 0.6667   # 1000000 / 1500000 (rounded)

def test_evaluate_ratios_zero_income_individual(valid_individual_profile):
    """Boundary: Individual with 0 income"""
    valid_individual_profile.stated_monthly_income = 0.0
    ratios = evaluate_ratios(valid_individual_profile)
    assert ratios["calculated_dti"] == float('inf')

def test_evaluate_ratios_zero_debt_llc(valid_llc_profile):
    """Boundary: LLC with 0 debt service has infinite coverage (previously an xfail defect)."""
    valid_llc_profile.annual_debt_service = 0.0
    ratios = evaluate_ratios(valid_llc_profile)
    assert ratios["calculated_dscr"] == float('inf')


def test_evaluate_ratios_uses_conservative_verified_figures(valid_individual_profile):
    """Verified income, the higher of stated/detected debt and the lower of stated/verified reserves."""
    metrics = build_verified_metrics(
        stated_monthly_income=10000.0,
        verified_income=8000.0,
        detected_liabilities=4000.0,
        verified_reserves=20000.0,
        kyc_confidence_score=0.99,
        document_tampering_score=0.0,
    )
    ratios = evaluate_ratios(valid_individual_profile, metrics)
    assert ratios["calculated_dti"] == 0.5             # 4000 / 8000
    assert ratios["liquidity_runway_months"] == 5.0    # 20000 / 4000
    assert ratios["status"] == "FAIL"


def test_missing_corporate_financials_flagged(valid_llc_profile):
    valid_llc_profile.annual_operating_income = None
    ratios = evaluate_ratios(valid_llc_profile)
    assert ratios["missing_financials"] is True