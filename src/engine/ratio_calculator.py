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

from dataclasses import dataclass
from typing import Any

from src.common.config import settings
from src.common.schemas import ApplicantProfile, EntityType, ExtractedFinancialMetrics


def calculate_dti(monthly_debt: float, monthly_income: float) -> float:
    """
    Computes Debt-to-Income ratio.
    Formula: total monthly debt obligations / gross monthly income
    """
    if monthly_income <= 0:
        return float("inf")
    
    return round(monthly_debt / monthly_income, 4)

def calculate_dscr(
        net_operating_income: float | None, 
        total_debt_service: float | None
) -> float | None:
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
        collateral_value: float | None
) -> float | None:
    """
    Computes Loan-to-Value ratio if collateral is provided.
    Formula: requested loan amount / stated collateral value
    """
    if collateral_value is None:
        return None
    if collateral_value <= 0:
        return float("inf")
    
    return round(loan_amount / collateral_value, 4)

@dataclass(frozen=True)
class QualifyingFigures:
    monthly_income: float
    monthly_debt: float
    liquid_assets: float
    collateral_value: float | None


def qualifying_figures(
        profile: ApplicantProfile,
        metrics: ExtractedFinancialMetrics | None = None,
) -> QualifyingFigures:
    """
    Picks the figures used for policy ratios. Without verification data the stated claims are
    used; with it, the conservative side of stated vs verified is taken:
    income = verified, debt = max(stated, detected), reserves = min(stated, verified),
    collateral = min(stated, verified) when both exist.
    """
    if metrics is None:
        return QualifyingFigures(
            monthly_income=profile.stated_monthly_income,
            monthly_debt=profile.total_monthly_debt,
            liquid_assets=profile.liquid_assets,
            collateral_value=profile.stated_collateral_value,
        )
    collaterals = [
        v for v in (profile.stated_collateral_value, metrics.verified_collateral_value)
        if v is not None
    ]
    return QualifyingFigures(
        monthly_income=metrics.verified_monthly_income,
        monthly_debt=max(profile.total_monthly_debt, metrics.detected_monthly_liabilities),
        liquid_assets=min(profile.liquid_assets, metrics.verified_liquid_reserves),
        collateral_value=min(collaterals) if collaterals else None,
    )

def evaluate_ratios(
        profile: ApplicantProfile,
        metrics: ExtractedFinancialMetrics | None = None,
) -> dict[str, Any]:
    """
    Computes underwriting ratios based on Entity Type.
    Individuals are evaluated on DTI. Corporate entities are evaluated on DSCR.
    """
    figures = qualifying_figures(profile, metrics)
    breached_policies: list[str] = []
    missing_financials = False
    dti: float | None = None
    dscr: float | None = None

    # 1. Entity-Specific Credit Evaluation
    if profile.entity_type == EntityType.INDIVIDUAL:
        # INDIVIDUALS: Use Monthly DTI
        dti = calculate_dti(
            monthly_debt=figures.monthly_debt,
            monthly_income=figures.monthly_income
        )
        if dti > settings.max_dti_threshold:
            breached_policies.append(
                f"Individual DTI {dti:.2%} exceeds maximum policy threshold {settings.max_dti_threshold:.2%}"
            )
    else:
        # LLC, CORP, TRUST: Use Annual DSCR
        dscr = calculate_dscr(
            net_operating_income=profile.annual_operating_income,
            total_debt_service=profile.annual_debt_service
        )
        if dscr is None:
            missing_financials = True
            breached_policies.append("CRITICAL: Missing annual operating income or debt service for corporate entity.")
        elif dscr < settings.min_dscr_threshold:
            breached_policies.append(
                f"Corporate DSCR {dscr:.2f}x is below minimum policy threshold {settings.min_dscr_threshold:.2f}x"
            )

    # 2. Universal Metrics (Liquidity & Collateral apply to ALL entities)
    liquidity_runway = calculate_liquidity_runway(
        liquid_assets=figures.liquid_assets,
        monthly_obligations=figures.monthly_debt
    )
    
    ltv = calculate_ltv(
        loan_amount=profile.requested_loan_amount,
        collateral_value=figures.collateral_value
    )

    # Liquidity Guardrail
    if liquidity_runway < settings.min_liquidity_months:
        breached_policies.append(
            f"Liquidity runway {liquidity_runway:.1f} months is below required reserve of {settings.min_liquidity_months:.1f} months"
        )

    # LTV Policy Guardrail
    if ltv is not None and ltv > settings.max_ltv_threshold:
        breached_policies.append(
            f"LTV {ltv:.2%} exceeds maximum secured policy threshold of {settings.max_ltv_threshold:.2%}"
        )

    preliminary_status = "PASS" if not breached_policies else "FAIL"

    return {
        "status": preliminary_status,
        "calculated_dti": dti,
        "calculated_dscr": dscr,
        "calculated_ltv": ltv,
        "liquidity_runway_months": liquidity_runway,
        "breached_policies": breached_policies,
        "missing_financials": missing_financials,
    }