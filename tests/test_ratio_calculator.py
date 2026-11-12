"""
src/tests/test_ratio_calculator.py

This verifies standard ratio mathematics, policy breach flags, and boundary edge 
cases (zero income, zero liabilities) mapping to the multimodal enterprise schema.
"""

import pytest
from src.engine.ratio_calculator import evaluate_ratios

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

@pytest.mark.xfail(reason="Production Bug: Unhandled ZeroDivisionError on 0 debt for LLC")
def test_evaluate_ratios_zero_debt_llc(valid_llc_profile):
    """Boundary: LLC with 0 debt (Exposes Production Defect)"""
    valid_llc_profile.annual_debt_service = 0.0
    ratios = evaluate_ratios(valid_llc_profile)
    assert ratios["calculated_dscr"] == float('inf') # Will fail until production bug is fixed