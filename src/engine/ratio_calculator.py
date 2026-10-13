# Deterministic Financial Ratio Engine

"""
Architectural Intent - By decoupling quantitative calculation from language
models, the pipeline guarantees repeatable, non-hallucinatory ratio outputs.

The calcualations handle edge cases deterministically:

    * Division by zero or negative earnings in Debt-to-Income(DTI) calculations return infinity rather than an 
    unhandled crash.

    * Zero annual debt service in Debt Service Coverage Ratio (DSCR) calculations cleanly signals that the applicant
    carries no required debt service.

    * Zero burn rate or zero recurring debt in liquidity runway checks handles infinite coverage gracefully. 

"""

from typing import Any, Dict, List, Optional
from src.common.config import settings
from src.common.schemas import ApplicantFinancials

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

def evaluate_ratios(financials: ApplicantFinancials) -> Dict[str, Any]:
    """
    Computes all standard underwriting ratios and compares them
    against enterprise policy limits configured in settings.
    """

    dti = calculate_dti(
        monthly_debt=financials.total_monthly_debt,
        monthly_income=financials.stated_monthly_income
    )

    dscr = calculate_dscr(
        net_operating_income=financials.annual_operating_income,
        total_debt_service=financials.annual_debt_service
    )

    liquidity_runway = calculate_liquidity_runway(
        liquid_assets=financials.liquid_assets,
        monthly_obligations=financials.total_monthly_debt
    )

    breached_policies: List[str] = []

    if dti > settings.max_dti_threshold:
        breached_policies.append(
            f"DTI {dti:.2%} exceeds maximum policy threshold {settings.max_dti_threshold:.2%}"
        )

    if dscr is not None and dscr < settings.min_dscr_threshold:
        breached_policies.append(
            f"DSCR {dscr:.2f} is below minimum policy threshold {settings.min_dscr_threshold:.2%}"
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
        "liquidity_runway_months": liquidity_runway,
        "breached_policies": breached_policies
    }