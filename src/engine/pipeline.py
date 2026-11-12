"""
Implement End-to-End Underwriting Pipeline

Coordinates multimodal parsed records, deterministic ratio math, compliance checks, 
and constructs the baseline structured memo prior to or parallel with LLM evaluation.
"""

from typing import Dict, Any
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
    UnderwritingDecision,
    UnderwritingMemo,
    AuditTrailEntry,
    EntityType
)
from src.engine.ratio_calculator import evaluate_ratios
from src.common.config import settings

class UnderwritingPipeline:
    """Coordinates parsing, deterministic compliance calculation, and decision rendering."""

    def run_preliminary_audit(
            self,
            profile: ApplicantProfile,
            metrics: ExtractedFinancialMetrics
    ) -> UnderwritingMemo:
        """Runs strict policy, fraud, and AML checks to construct an audit-ready memo."""
        
        audit_trail = [
            AuditTrailEntry(
                step_name="Multimodal_Ingestion",
                agent_role="FastAPI_Gateway",
                status="SUCCESS",
                details=f"Received {profile.entity_type.value} application for {profile.applicant_id}"
            )
        ]

        # 1. Deterministic Mathematical Ratio Checks
        ratio_results = evaluate_ratios(profile)
        
        # Calculate LTV if collateral is present
        ltv = None
        if profile.stated_collateral_value and profile.stated_collateral_value > 0:
            ltv = profile.requested_loan_amount / profile.stated_collateral_value

        # Safely format DTI for the audit trail (LLCs/Corps will be N/A)
        dti_val = ratio_results.get('calculated_dti')
        dti_str = f"{dti_val:.2%}" if dti_val is not None else "N/A"
            
        audit_trail.append(
            AuditTrailEntry(
                step_name="RatioEvaluation",
                agent_role="Deterministic_Math_Engine",
                status=ratio_results.get("status", "SUCCESS"),
                details=f"Computed DTI: {dti_str}, Runway: {ratio_results.get('liquidity_runway_months', 0)} months",
            )
        )

        # 2. Extract Fraud & Compliance Metrics
        all_breaches = list(ratio_results.get("breached_policies", []))
        risk_factors = list(metrics.anomaly_flags)
        
        # Aggregate fraud probability (taking the highest risk vector)
        overall_fraud_prob = max(
            metrics.fraud_metrics.document_tampering_score,
            1.0 - metrics.fraud_metrics.kyc_confidence_score,
            metrics.fraud_metrics.synthetic_voice_probability or 0.0
        )

        # 3. Waterfall Decision Logic
        decision = UnderwritingDecision.APPROVE
        approved_amount = profile.requested_loan_amount
        covenants = []

        # A. Fatal Fraud & AML Checks (Immediate Rejection)
        if metrics.pep_or_sanctions_hit:
            decision = UnderwritingDecision.REJECT_FRAUD
            all_breaches.append("OFAC/Sanctions list hit detected.")
            approved_amount = 0.0
            
        elif overall_fraud_prob > settings.max_fraud_probability:
            decision = UnderwritingDecision.REJECT_FRAUD
            all_breaches.append(f"Fraud probability ({overall_fraud_prob:.2%}) exceeds allowable threshold.")
            approved_amount = 0.0

        # B. Critical Compliance & Entity Checks
        elif profile.entity_type in [EntityType.LLC, EntityType.CORP] and not metrics.ubo_verified:
            decision = UnderwritingDecision.MANUAL_REVIEW
            risk_factors.append("Ultimate Beneficial Owner (UBO) structure unverified.")
            
        # C. Credit & Liquidity Checks
        elif len(all_breaches) > 1 or (metrics.verified_monthly_income < profile.stated_monthly_income * 0.75):
            decision = UnderwritingDecision.REJECT_CREDIT
            risk_factors.append("Severe income variance or multiple policy breaches.")
            approved_amount = 0.0
            
        elif len(all_breaches) == 1 or len(risk_factors) > 0:
            decision = UnderwritingDecision.CONDITIONAL_APPROVAL
            covenants.append("Require 12 months of certified bank statements prior to closing.")
            if profile.entity_type != EntityType.INDIVIDUAL:
                covenants.append("Unconditional Personal Guarantee required from all UBOs.")

        audit_trail.append(
            AuditTrailEntry(
                step_name="DecisionRuleCheck",
                agent_role="Deterministic_Math_Engine",
                status=decision.value,
                details=f"Assigned preliminary decision. Fraud Prob: {overall_fraud_prob:.2%}"
            )
        )

        return UnderwritingMemo(
            application_id=profile.applicant_id,
            decision=decision,
            calculated_dti=ratio_results["calculated_dti"],
            calculated_dscr=ratio_results.get("calculated_dscr"),
            calculated_ltv=ltv,
            liquidity_runway_months=ratio_results["liquidity_runway_months"],
            overall_fraud_probability=round(overall_fraud_prob, 4),
            breached_policies=all_breaches,
            risk_factors=risk_factors,
            approved_loan_amount=approved_amount,
            required_covenants=covenants,
            audit_trail=audit_trail
        )