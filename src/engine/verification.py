"""
Verification aggregation: turns stated claims + verification-layer outputs (document parsing,
KYC vendor, fraud models) into the validated ``ExtractedFinancialMetrics`` contract.

Moved out of ``src/vision/parser.py`` so the JSON API does not depend on PDF libraries and so
income-variance maths lives in exactly one place.
"""

from src.common.config import settings
from src.common.schemas import ExtractedFinancialMetrics, MultimodalFraudMetrics


def income_variance_pct(stated_monthly_income: float, verified_monthly_income: float) -> float:
    """Signed % variance of verified vs stated income (0.0 when nothing was stated)."""
    if stated_monthly_income <= 0:
        return 0.0
    return round(
        ((verified_monthly_income - stated_monthly_income) / stated_monthly_income) * 100, 2
    )


def build_verified_metrics(
    *,
    stated_monthly_income: float,
    verified_income: float,
    detected_liabilities: float,
    verified_reserves: float,
    kyc_confidence_score: float,
    document_tampering_score: float,
    verified_collateral_value: float | None = None,
    pep_or_sanctions_hit: bool = False,
    ubo_verified: bool = True,
    synthetic_voice_probability: float | None = None,
    supplied_flags: list[str] | None = None,
) -> ExtractedFinancialMetrics:
    """Computes variance, compiles anomaly flags and returns the validated metrics payload."""
    variance_pct = income_variance_pct(stated_monthly_income, verified_income)
    anomaly_flags: list[str] = list(dict.fromkeys(supplied_flags or []))

    # Credit Anomalies
    if verified_income < stated_monthly_income * settings.income_mismatch_flag_ratio:
        anomaly_flags.append(
            f"Income mismatch: verified income is {abs(variance_pct):.1f}% lower than stated"
        )
    if detected_liabilities <= 0:
        anomaly_flags.append("Warning: zero recurring liabilities detected in document records")

    # Fraud & Compliance Anomalies
    if document_tampering_score > settings.tampering_flag_threshold:
        anomaly_flags.append(
            "High probability of document tampering/Photoshop metadata detected "
            f"({document_tampering_score:.1%})."
        )
    if pep_or_sanctions_hit:
        anomaly_flags.append("CRITICAL: Applicant or UBO matches active OFAC/Sanctions watchlist.")

    return ExtractedFinancialMetrics(
        verified_monthly_income=verified_income,
        detected_monthly_liabilities=detected_liabilities,
        verified_liquid_reserves=verified_reserves,
        verified_collateral_value=verified_collateral_value,
        pep_or_sanctions_hit=pep_or_sanctions_hit,
        ubo_verified=ubo_verified,
        fraud_metrics=MultimodalFraudMetrics(
            kyc_confidence_score=kyc_confidence_score,
            document_tampering_score=document_tampering_score,
            synthetic_voice_probability=synthetic_voice_probability,
        ),
        income_variance_pct=variance_pct,
        anomaly_flags=anomaly_flags,
    )
