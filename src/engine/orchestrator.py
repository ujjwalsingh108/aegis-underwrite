"""
Underwriting service: the one canonical path from an application to a persisted decision.

    request -> verification metrics -> deterministic rules engine (decision of record)
            -> optional LLM advisory (vLLM/LoRA on Modal GPU) -> guardrail reconciliation
            -> decision ledger + structured audit log

Both API entrypoints (JSON and PDF upload) call this service, so decision logic exists once.
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from src.common.config import settings
from src.common.logger import get_logger, request_id_ctx
from src.common.prompts import build_inference_prompt
from src.common.schemas import (
    ApplicantProfile,
    DecisionRecord,
    ExtractedFinancialMetrics,
    LLMAdvisoryStatus,
    UnderwriteRequest,
    UnderwritingMemo,
)
from src.engine.decision_store import DecisionStore
from src.engine.guardrails import ESCALATION_RULE, reconcile_with_advisory
from src.engine.pipeline import UnderwritingPipeline
from src.engine.verification import build_verified_metrics

logger = get_logger("orchestrator")
audit_logger = get_logger("audit")


def _default_llm() -> Any:
    from src.serving.vllm_engine import UnderwritingLLMEngine

    return UnderwritingLLMEngine()


def _parse_pdf(path: str) -> tuple[str, int]:
    from src.vision.parser import FinancialDocumentParser

    parser = FinancialDocumentParser()
    return parser.extract_text(path), len(parser.extract_tables(path))


class PipelineOrchestrator:
    def __init__(
        self,
        store: DecisionStore,
        llm: Any | None = None,
        pipeline: UnderwritingPipeline | None = None,
        llm_enabled: bool | None = None,
    ):
        self.store = store
        self.pipeline = pipeline or UnderwritingPipeline()
        self.llm_enabled = settings.llm_review_enabled if llm_enabled is None else llm_enabled
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = _default_llm()
        return self._llm

    # ----------------------------------------------------------------------------------
    @staticmethod
    def build_metrics(request: UnderwriteRequest) -> ExtractedFinancialMetrics:
        return build_verified_metrics(
            stated_monthly_income=request.stated_monthly_income,
            verified_income=request.verified_monthly_income,
            detected_liabilities=(
                request.detected_monthly_liabilities
                if request.detected_monthly_liabilities is not None
                else request.total_monthly_debt
            ),
            verified_reserves=(
                request.verified_liquid_reserves
                if request.verified_liquid_reserves is not None
                else request.liquid_reserves
            ),
            verified_collateral_value=request.verified_collateral_value,
            kyc_confidence_score=request.kyc_confidence_score,
            document_tampering_score=request.document_tampering_score,
            synthetic_voice_probability=request.synthetic_voice_probability,
            pep_or_sanctions_hit=request.pep_or_sanctions_hit,
            ubo_verified=request.ubo_verified,
            supplied_flags=request.anomaly_flags,
        )

    async def _llm_advisory(
        self,
        profile: ApplicantProfile,
        metrics: ExtractedFinancialMetrics,
        document_excerpt: str | None,
    ) -> tuple[UnderwritingMemo | None, LLMAdvisoryStatus, str | None]:
        if not self.llm_enabled:
            return None, LLMAdvisoryStatus.DISABLED, "LLM advisory disabled by configuration."
        prompt = build_inference_prompt(profile, metrics, document_excerpt)
        try:
            raw = await asyncio.wait_for(
                self.llm.generate_memo.remote.aio(prompt), timeout=settings.llm_timeout_seconds
            )
            llm_memo = raw if isinstance(raw, UnderwritingMemo) else UnderwritingMemo.model_validate(raw)
            return llm_memo, LLMAdvisoryStatus.COMPLETED, None
        except Exception as exc:  # advisory must never take the decision path down
            logger.warning(
                "LLM advisory failed; deterministic decision stands",
                extra={"applicant_id": profile.applicant_id, "error_type": type(exc).__name__},
            )
            return None, LLMAdvisoryStatus.FAILED, f"LLM advisory unavailable ({type(exc).__name__})."

    async def underwrite(
        self,
        request: UnderwriteRequest,
        *,
        decided_by: str,
        decided_by_role: str,
        document_excerpt: str | None = None,
        source_document: str | None = None,
    ) -> DecisionRecord:
        profile = request.to_profile()
        metrics = self.build_metrics(request)

        result = self.pipeline.evaluate(profile, metrics)
        llm_memo, status, detail = await self._llm_advisory(profile, metrics, document_excerpt)
        final_memo, advisory = reconcile_with_advisory(result.memo, llm_memo, status, detail)

        decision_rule = ESCALATION_RULE if advisory.escalated else result.decision_rule
        reasons = list(result.decision_reasons)
        if advisory.escalated:
            reasons.append(
                f"LLM advisory recommended {advisory.suggested_decision.value}; "  # type: ignore[union-attr]
                "a human underwriter must review."
            )

        record = DecisionRecord(
            decision_id=str(uuid.uuid4()),
            request_id=request_id_ctx.get(),
            decided_at=datetime.now(timezone.utc),
            decided_by=decided_by,
            decided_by_role=decided_by_role,
            decision_rule=decision_rule,
            decision_reasons=reasons,
            memo=final_memo,
            llm_advisory=advisory,
            source_document=source_document,
        )
        await self.store.save(record)

        audit_logger.info(
            "underwriting_decision",
            extra={
                "event": "underwriting_decision",
                "decision_id": record.decision_id,
                "applicant_id": final_memo.application_id,
                "decision": final_memo.decision.value,
                "decision_rule": decision_rule,
                "breached_policies": final_memo.breached_policies,
                "fraud_probability": final_memo.overall_fraud_probability,
                "dti": final_memo.calculated_dti,
                "dscr": final_memo.calculated_dscr,
                "ltv": final_memo.calculated_ltv,
                "approved_loan_amount": final_memo.approved_loan_amount,
                "llm_advisory": advisory.status.value,
                "llm_escalated": advisory.escalated,
                "decided_by": decided_by,
                "decided_at": record.decided_at.isoformat(),
            },
        )
        return record

    async def process_application(
        self,
        request: UnderwriteRequest,
        file_bytes: bytes,
        filename: str,
        *,
        decided_by: str,
        decided_by_role: str,
    ) -> DecisionRecord:
        """Underwrites an application with a supporting PDF (text excerpt goes to the advisor).

        Financial figures still come from the request: automated extraction of figures from the
        document (Qwen2-VL vision agent) is not implemented yet.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
        try:
            raw_text, table_count = await asyncio.to_thread(_parse_pdf, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        logger.info(
            "document_parsed",
            extra={"doc_name": filename, "chars": len(raw_text), "tables": table_count},
        )
        return await self.underwrite(
            request,
            decided_by=decided_by,
            decided_by_role=decided_by_role,
            document_excerpt=raw_text or None,
            source_document=filename,
        )
