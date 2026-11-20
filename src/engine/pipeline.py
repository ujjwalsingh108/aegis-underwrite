"""
Deterministic underwriting rules engine (the decision of record).

Coordinates verified metrics, deterministic ratio maths and compliance checks, then applies an
ordered waterfall of policy rules. Every decision carries the identifier of the rule that fired
plus human-readable reasons, so it is explainable for audit and adverse-action notices.

The same engine produces the ground-truth labels for LoRA training (dataset_generator.py).
"""

from dataclasses import dataclass, field

from src.common.config import settings
from src.common.schemas import (
    ApplicantProfile,
    AuditTrailEntry,
    EntityType,
    ExtractedFinancialMetrics,
    UnderwritingDecision,
    UnderwritingMemo,
)
from src.engine.ratio_calculator import evaluate_ratios


class Rule:
    """Stable identifiers for each waterfall rule (persisted with every decision)."""
    SANCTIONS_HIT = "R1_SANCTIONS_HIT"
    FRAUD_PROBABILITY = "R2_FRAUD_PROBABILITY"
    INCOMPLETE_FINANCIALS = "R3_INCOMPLETE_CORPORATE_FINANCIALS"
    UBO_UNVERIFIED = "R4_UBO_UNVERIFIED"
    CREDIT_DECLINE = "R5_MULTIPLE_BREACHES_OR_INCOME_SHORTFALL"
    CONDITIONAL = "R6_SINGLE_BREACH_OR_RISK_FACTORS"
    CLEAN = "R7_ALL_POLICIES_PASSED"


@dataclass
class PipelineResult:
    memo: UnderwritingMemo
    decision_rule: str
    decision_reasons: list[str] = field(default_factory=list)


def overall_fraud_probability(metrics: ExtractedFinancialMetrics) -> float:
    """Highest risk vector across tampering, KYC mismatch and synthetic voice."""
    return max(
        metrics.fraud_metrics.document_tampering_score,
        1.0 - metrics.fraud_metrics.kyc_confidence_score,
        metrics.fraud_metrics.synthetic_voice_probability or 0.0,
    )


class UnderwritingPipeline:
    """Coordinates parsing, deterministic compliance calculation, and decision rendering."""

    def evaluate(
            self,
            profile: ApplicantProfile,
            metrics: ExtractedFinancialMetrics,
    ) -> PipelineResult:
        """Runs strict policy, fraud, and AML checks and returns an explainable decision."""
        audit_trail = [
            AuditTrailEntry(
                step_name="Multimodal_Ingestion",
                agent_role="FastAPI_Gateway",
                status="SUCCESS",
                details=f"Received {profile.entity_type.value} application for {profile.applicant_id}",
            )
        ]

        # 1. Deterministic Mathematical Ratio Checks (on conservative verified figures)
        ratio_results = evaluate_ratios(profile, metrics)
        dti_val = ratio_results.get("calculated_dti")
        dti_str = f"{dti_val:.2%}" if dti_val is not None else "N/A"
        dscr_val = ratio_results.get("calculated_dscr")
        dscr_str = f"{dscr_val:.2f}x" if dscr_val is not None else "N/A"

        audit_trail.append(
            AuditTrailEntry(
                step_name="RatioEvaluation",
                agent_role="Deterministic_Math_Engine",
                status=ratio_results.get("status", "SUCCESS"),
                details=(
                    f"Computed DTI: {dti_str}, DSCR: {dscr_str}, "
                    f"Runway: {ratio_results.get('liquidity_runway_months', 0)} months"
                ),
            )
        )

        # 2. Extract Fraud & Compliance Metrics
        all_breaches = list(ratio_results.get("breached_policies", []))
        risk_factors = list(metrics.anomaly_flags)
        fraud_prob = overall_fraud_probability(metrics)
        severe_shortfall = (
            metrics.verified_monthly_income
            < profile.stated_monthly_income * settings.severe_income_shortfall_ratio
        )

        # 3. Waterfall Decision Logic (first matching rule wins)
        decision = UnderwritingDecision.APPROVE
        approved_amount = profile.requested_loan_amount
        covenants: list[str] = []
        reasons: list[str] = []

        # A. Fatal Fraud & AML Checks (Immediate Rejection)
        if metrics.pep_or_sanctions_hit:
            rule = Rule.SANCTIONS_HIT
            decision = UnderwritingDecision.REJECT_FRAUD
            all_breaches.append("OFAC/Sanctions list hit detected.")
            reasons.append("Applicant or beneficial owner matched a sanctions/PEP watchlist.")
            approved_amount = 0.0

        elif fraud_prob > settings.max_fraud_probability:
            rule = Rule.FRAUD_PROBABILITY
            decision = UnderwritingDecision.REJECT_FRAUD
            all_breaches.append(f"Fraud probability ({fraud_prob:.2%}) exceeds allowable threshold.")
            reasons.append(
                f"Fraud probability {fraud_prob:.2%} exceeds the "
                f"{settings.max_fraud_probability:.2%} limit."
            )
            approved_amount = 0.0

        # B. Cannot assess credit without corporate financials
        elif ratio_results.get("missing_financials"):
            rule = Rule.INCOMPLETE_FINANCIALS
            decision = UnderwritingDecision.MANUAL_REVIEW
            reasons.append("Annual operating income and debt service are required for DSCR.")

        # C. Critical Compliance & Entity Checks
        elif profile.entity_type in [EntityType.LLC, EntityType.CORP] and not metrics.ubo_verified:
            rule = Rule.UBO_UNVERIFIED
            decision = UnderwritingDecision.MANUAL_REVIEW
            risk_factors.append("Ultimate Beneficial Owner (UBO) structure unverified.")
            reasons.append("Ultimate Beneficial Owner structure could not be verified.")

        # D. Credit & Liquidity Checks
        elif len(all_breaches) > 1 or severe_shortfall:
            rule = Rule.CREDIT_DECLINE
            decision = UnderwritingDecision.REJECT_CREDIT
            risk_factors.append("Severe income variance or multiple policy breaches.")
            reasons.extend(all_breaches)
            if severe_shortfall:
                reasons.append(
                    f"Verified income is {abs(metrics.income_variance_pct):.1f}% below stated income."
                )
            approved_amount = 0.0

        elif len(all_breaches) == 1 or len(risk_factors) > 0:
            rule = Rule.CONDITIONAL
            decision = UnderwritingDecision.CONDITIONAL_APPROVAL
            covenants.append("Require 12 months of certified bank statements prior to closing.")
            if profile.entity_type != EntityType.INDIVIDUAL:
                covenants.append("Unconditional Personal Guarantee required from all UBOs.")
            reasons.extend(all_breaches or risk_factors)

        else:
            rule = Rule.CLEAN
            reasons.append("All credit, liquidity, collateral and fraud policies passed.")

        audit_trail.append(
            AuditTrailEntry(
                step_name="DecisionRuleCheck",
                agent_role="Deterministic_Math_Engine",
                status=decision.value,
                details=f"Rule {rule} fired. Fraud Prob: {fraud_prob:.2%}",
            )
        )

        memo = UnderwritingMemo(
            application_id=profile.applicant_id,
            decision=decision,
            calculated_dti=dti_val,
            calculated_dscr=dscr_val,
            calculated_ltv=ratio_results.get("calculated_ltv"),
            liquidity_runway_months=ratio_results["liquidity_runway_months"],
            overall_fraud_probability=round(fraud_prob, 4),
            breached_policies=all_breaches,
            risk_factors=risk_factors,
            approved_loan_amount=approved_amount,
            required_covenants=covenants,
            audit_trail=audit_trail,
        )
        return PipelineResult(memo=memo, decision_rule=rule, decision_reasons=reasons)

    def run_preliminary_audit(
            self,
            profile: ApplicantProfile,
            metrics: ExtractedFinancialMetrics,
    ) -> UnderwritingMemo:
        """Backwards-compatible entrypoint used by dataset generation and tests."""
        return self.evaluate(profile, metrics).memo
