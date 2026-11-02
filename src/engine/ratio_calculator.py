# Deterministic Financial Ratio Engine

"""
Architectural Intent - By decoupling quantitative calculation from language
models, the pipeline guarantees repeatable, non-hallucinatory ratio outputs.

The calculations handle edge cases deterministically:

    * Division by zero or negative earnings in Debt-to-Income (DTI) or Loan-to-Value (LTV) 
      calculations return infinity rather than an unhandled crash.

    * Zero annual debt service in Debt Service Coverage Ratio (DSCR) calculations cleanly 
      signals that the applicant carries no required debt service.

    * Zero burn rate or zero recurring debt in liquidity runway checks handles infinite coverage gracefully. 
"""

from typing import Any, Dict, List, Optional
from src.common.config import settings
from src.common.schemas import ApplicantProfile

def calculate_dti(monthly_debt: float, monthly_income: float) -> float:
    """
    Computes Debt-to-Income ratio.
    Formula: total monthly debt obligations / gross monthly income
    """
    if monthly_income <= 0:
        return float("inf")
    
    return round(monthly_debt / monthly_income, 4)

def calculate_dscr(
        net_operating_income: Optional[float], 
        total_debt_service: Optional[float]
) -> Optional[float]:
    """
    Computes Debt Service Coverage Ratio for corporate and business entities.
    Formula: annual net operating income / annual debt service obligations.
    Returns None if either component is missing.
    """
    if net_operating_income is None or total_debt_service is None:
        return None
    elif total_debt_service <= 0:
        return float("inf") if net_operating_income > 0 else 0.0

    return round(net_operating_income / total_debt_service, 4)

def calculate_liquidity_runway(
        liquid_assets: float, 
        monthly_obligations: float
) -> float:
    """
    Computes liquidity runway in months supported by cash reserves.
    Formula: accessible liquid assets / recurring monthly obligations.
    """
    if monthly_obligations <= 0:
        return float("inf") if liquid_assets > 0 else 0.0

    return round(liquid_assets / monthly_obligations, 2)

def calculate_ltv(
        loan_amount: float,
        collateral_value: Optional[float]
) -> Optional[float]:
    """
    Computes Loan-to-Value ratio if collateral is provided.
    Formula: requested loan amount / stated collateral value
    """
    if collateral_value is None:
        return None
    if collateral_value <= 0:
        return float("inf")
    
    return round(loan_amount / collateral_value, 4)

def evaluate_ratios(profile: ApplicantProfile) -> Dict[str, Any]:
    """
    Computes all standard underwriting ratios based on the multimodal ApplicantProfile 
    and compares them against enterprise policy limits configured in settings.
    """
    dti = calculate_dti(
        monthly_debt=profile.total_monthly_debt,
        monthly_income=profile.stated_monthly_income
    )

    dscr = calculate_dscr(
        net_operating_income=profile.annual_operating_income,
        total_debt_service=profile.annual_debt_service
    )

    liquidity_runway = calculate_liquidity_runway(
        liquid_assets=profile.liquid_assets,
        monthly_obligations=profile.total_monthly_debt
    )
    
    ltv = calculate_ltv(
        loan_amount=profile.requested_loan_amount,
        collateral_value=profile.stated_collateral_value
    )

    breached_policies: List[str] = []

    # Apply Policy Thresholds
    if dti > settings.max_dti_threshold:
        breached_policies.append(
            f"DTI {dti:.2%} exceeds maximum policy threshold {settings.max_dti_threshold:.2%}"
        )

    if dscr is not None and dscr < settings.min_dscr_threshold:
        breached_policies.append(
            f"DSCR {dscr:.2f} is below minimum policy threshold {settings.min_dscr_threshold:.2f}x"
        )

    if liquidity_runway < settings.min_liquidity_months:
        breached_policies.append(
            f"Liquidity runway {liquidity_runway:.1f} months is below required reserve of {settings.min_liquidity_months:.1f} months"
        )

    preliminary_status = "PASS" if not breached_policies else "FAIL"

    return {
        "status": preliminary_status,
        "calculated_dti": dti,
        "calculated_dscr": dscr,
        "calculated_ltv": ltv,
        "liquidity_runway_months": liquidity_runway,
        "breached_policies": breached_policies
    }