"""
Deterministic guardrails between the rules engine and the LLM advisory.

Policy:
* The deterministic rules engine is the decision of record.
* The LLM may only make an outcome MORE conservative, and only by routing it to MANUAL_REVIEW.
  It can never approve, relax, or auto-reject (an LLM-only adverse action would have no
  explainable, rule-based reason, which is unacceptable for adverse-action notices).
* Because the LLM cannot relax a decision, prompt injection through documents or anomaly
  flags cannot produce an approval.
"""

from src.common.schemas import (
    DECISION_SEVERITY,
    AuditTrailEntry,
    LLMAdvisory,
    LLMAdvisoryStatus,
    UnderwritingDecision,
    UnderwritingMemo,
)

ESCALATION_RULE = "G1_LLM_ADVISORY_ESCALATION"

_ESCALATABLE = {UnderwritingDecision.APPROVE, UnderwritingDecision.CONDITIONAL_APPROVAL}


def reconcile_with_advisory(
    memo: UnderwritingMemo,
    llm_memo: UnderwritingMemo | None,
    advisory_status: LLMAdvisoryStatus,
    failure_detail: str | None = None,
) -> tuple[UnderwritingMemo, LLMAdvisory]:
    """Returns the final memo and a description of what the advisory contributed."""
    audit = list(memo.audit_trail)

    if llm_memo is None:
        audit.append(
            AuditTrailEntry(
                step_name="LLMAdvisoryReview",
                agent_role="vLLM_LoRA_Advisor",
                status=advisory_status.value,
                details=failure_detail or "LLM advisory not requested.",
            )
        )
        return (
            memo.model_copy(update={"audit_trail": audit}),
            LLMAdvisory(status=advisory_status, detail=failure_detail),
        )

    more_conservative = DECISION_SEVERITY[llm_memo.decision] > DECISION_SEVERITY[memo.decision]
    escalate = more_conservative and memo.decision in _ESCALATABLE
    new_factors = [f for f in llm_memo.risk_factors if f and f not in memo.risk_factors]

    update: dict = {}
    if escalate:
        update["decision"] = UnderwritingDecision.MANUAL_REVIEW
        update["risk_factors"] = [
            *memo.risk_factors,
            f"LLM advisory recommended {llm_memo.decision.value}; routed to human review.",
        ]
    audit.append(
        AuditTrailEntry(
            step_name="LLMAdvisoryReview",
            agent_role="vLLM_LoRA_Advisor",
            status="ESCALATED" if escalate else "CONCURRED_OR_IGNORED",
            details=(
                f"Deterministic={memo.decision.value}, LLM={llm_memo.decision.value}. "
                + ("Escalated to MANUAL_REVIEW." if escalate else "Deterministic decision stands.")
            ),
        )
    )
    update["audit_trail"] = audit

    return (
        memo.model_copy(update=update),
        LLMAdvisory(
            status=LLMAdvisoryStatus.COMPLETED,
            suggested_decision=llm_memo.decision,
            risk_factors=new_factors[:20],
            escalated=escalate,
        ),
    )
