# Tests the underwriting service: sequential execution, persistence and temp-file cleanup.

import glob
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.schemas import DecisionRecord, UnderwriteRequest
from src.engine.decision_store import DecisionStore
from src.engine.orchestrator import PipelineOrchestrator


@pytest.fixture
def store():
    s = DecisionStore(":memory:")
    s.open()
    yield s
    s.close()


async def test_orchestrator_document_flow_and_cleanup(store, underwrite_payload, blank_pdf_bytes):
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))
    orchestrator = PipelineOrchestrator(store=store, llm_enabled=False)

    record = await orchestrator.process_application(
        UnderwriteRequest(**underwrite_payload),
        blank_pdf_bytes,
        "statement.pdf",
        decided_by="tester",
        decided_by_role="UNDERWRITER",
    )

    assert isinstance(record, DecisionRecord)
    assert record.memo.application_id == "APP-111"  # from the request, not a placeholder
    assert record.source_document == "statement.pdf"
    assert await store.get(record.decision_id) is not None
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))
    assert after <= before  # temp PDF removed


async def test_orchestrator_sends_canonical_prompt(store, underwrite_payload):
    llm = MagicMock()
    llm.generate_memo.remote.aio = AsyncMock(side_effect=ValueError("bad json"))
    orchestrator = PipelineOrchestrator(store=store, llm=llm, llm_enabled=True)

    record = await orchestrator.underwrite(
        UnderwriteRequest(**underwrite_payload), decided_by="t", decided_by_role="ADMIN"
    )

    prompt = llm.generate_memo.remote.aio.call_args.args[0]
    assert prompt.startswith("<|im_start|>system\nYou are an underwriting risk analyst.")
    assert "Applicant ID: APP-111\nEntity Type: INDIVIDUAL" in prompt
    assert record.llm_advisory.status.value == "FAILED"
    assert record.memo.decision.value == "APPROVE"
