"""
Implement End-to-End Underwriting Pipeline

Bringing all the pieces together:
    * Accept applicant details,
    * Parsse extracted financial records,
    * Compute math ratios deterministically using ratio_calculator.py,
    * Run policy rules,
    * Generate an underwriting memo using the serving engine
"""

from typing import Dict, Any
from src.common.schemas import (
    ApplicantFinancials,
    ExtractedFinancialMetrics,
    UnderwritingDecision,
    UnderwritingMemo,
    AuditTrialEntry
)
from src.engine.ratio_calculator import evaluate_ratios

class UnderwritingPipeline:
    """Coordinates parsing, deterministic calculation, and decision rendering."""

    def run_preliminary_audit(
            self,
            financials: ApplicantFinancials,
            metrics: ExtractedFinancialMetrics
    ) -> UnderwritingMemo:
        """Runs policy checks and constructs an audit-ready memo."""
        audit_trail = [
            AuditTrialEntry(
                step_name="Ingestion",
                status="SUCCESS",
                details=f"Received application for borrower {financials.applicant_id}"
            )
        ]

        # Deterministic mathematical ratio checks
        ratio_results = evaluate_ratios(financials)
        audit_trail.append(
            AuditTrialEntry(
                step_name="RatioEvaluation",
                status=ratio_results["status"],
                details=f"Computed DTI: {ratio_results['calculated_dti']:.2%}, Runway: {ratio_results['liquidity_runway_months']} months",
            )
        )

        # Evaluate risk policies and anomalies
        all_breaches = list(ratio_results["breached_policies"])
        risk_factors = list(metrics.anomaly_flags)

        decision = UnderwritingDecision.APPROVE
        if len(all_breaches) > 1 or metrics.income_variance_pct < -25.0:
            decision = UnderwritingDecision.REJECT
        elif len(all_breaches) == 1 or len(risk_factors) > 0:
            decision = UnderwritingDecision.CONDITIONAL

        audit_trail.append(
            AuditTrialEntry(
                step_name="DecisionRuleCheck",
                status=decision.value,
                details=f"Assigned preliminary decision based on {len(all_breaches)} policy breaches",
            )
        )

        return UnderwritingMemo(
            application_id=financials.applicant_id,
            decision=decision,
            calculated_dti=ratio_results["calculated_dti"],
            calculated_dscr=ratio_results["calculated_dscr"],
            liquidity_runway_months=ratio_results["liquidity_runway_months"],
            breached_policies=all_breaches,
            risk_factors=risk_factors,
            covenants=[
                "Maintain quarterly bank statements",
                "Personal guarantee required if corporate entity"
            ]
            if decision == UnderwritingDecision.CONDITIONAL
            else [],
            audit_trail=audit_trail
        )
    