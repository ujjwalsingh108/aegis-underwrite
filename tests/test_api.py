"""
API integration tests: auth/RBAC, validation, the end-to-end decision flow, the decision ledger,
document upload and error sanitisation.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.middleware import underwriting_limiter
from src.common.config import settings
from src.common.schemas import UnderwritingDecision, UnderwritingMemo

ADMIN = {"X-API-Key": settings.master_api_key.get_secret_value()}
UNDERWRITER = {"X-API-Key": settings.underwriter_api_key.get_secret_value()}
AUDITOR = {"X-API-Key": settings.auditor_api_key.get_secret_value()}


@pytest.fixture
def client():
    underwriting_limiter.reset()
    with TestClient(app) as c:  # runs lifespan (opens the in-memory decision store)
        yield c


def _mock_llm(decision: str = "APPROVE") -> MagicMock:
    memo = UnderwritingMemo(
        application_id="APP-111", decision=decision, liquidity_runway_months=12,
        risk_factors=["LLM noted thin credit file"],
    )
    llm = MagicMock()
    llm.generate_memo.remote.aio = AsyncMock(return_value=memo)
    return llm


def test_health_and_ready_are_public(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert "X-Request-ID" in ready.headers
    assert ready.headers["X-Content-Type-Options"] == "nosniff"


def test_request_id_is_propagated(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_underwrite_requires_valid_key(client, underwrite_payload):
    assert client.post("/underwrite", json=underwrite_payload).status_code in (401, 403)
    bad = client.post("/underwrite", json=underwrite_payload, headers={"X-API-Key": "nope"})
    assert bad.status_code == 401


def test_auditor_cannot_underwrite(client, underwrite_payload):
    response = client.post("/underwrite", json=underwrite_payload, headers=AUDITOR)
    assert response.status_code == 403


def test_underwrite_end_to_end_deterministic(client, underwrite_payload):
    """Happy path without the GPU: real rules engine, persisted, retrievable, explainable."""
    response = client.post("/underwrite", json=underwrite_payload, headers=UNDERWRITER)
    assert response.status_code == 200, response.text
    record = response.json()

    assert record["memo"]["decision"] == "APPROVE"
    assert record["decision_rule"] == "R7_ALL_POLICIES_PASSED"
    assert record["decision_reasons"]
    assert record["memo"]["calculated_dti"] == round(3000 / 9500, 4)  # verified income
    assert record["memo"]["approved_loan_amount"] == 150000.0
    assert record["llm_advisory"]["status"] == "DISABLED"
    assert record["decided_by"] == "underwriter-svc"

    fetched = client.get(f"/decisions/{record['decision_id']}", headers=AUDITOR)
    assert fetched.status_code == 200
    assert fetched.json()["memo"]["decision"] == "APPROVE"

    page = client.get("/decisions", params={"application_id": "APP-111"}, headers=AUDITOR).json()
    assert page["total"] >= 1 and page["items"][0]["decision_id"] == record["decision_id"]


def test_underwrite_rejects_on_breaches(client, underwrite_payload):
    payload = {**underwrite_payload, "total_monthly_debt": 6000.0, "liquid_reserves": 1000.0}
    record = client.post("/underwrite", json=payload, headers=UNDERWRITER).json()
    assert record["memo"]["decision"] == "REJECT_CREDIT"
    assert record["memo"]["approved_loan_amount"] == 0.0
    assert len(record["memo"]["breached_policies"]) == 2


def test_zero_debt_infinite_runway_round_trips(client, underwrite_payload):
    payload = {**underwrite_payload, "total_monthly_debt": 0.0}
    record = client.post("/underwrite", json=payload, headers=UNDERWRITER).json()
    assert record["memo"]["liquidity_runway_months"] == "Infinity"
    fetched = client.get(f"/decisions/{record['decision_id']}", headers=AUDITOR)
    assert fetched.status_code == 200


def test_llm_can_escalate_but_never_relax(client, underwrite_payload):
    orchestrator = app.state.orchestrator
    orchestrator.llm_enabled, orchestrator._llm = True, _mock_llm("REJECT_CREDIT")
    try:
        record = client.post("/underwrite", json=underwrite_payload, headers=UNDERWRITER).json()
        assert record["memo"]["decision"] == UnderwritingDecision.MANUAL_REVIEW.value
        assert record["decision_rule"] == "G1_LLM_ADVISORY_ESCALATION"
        assert record["llm_advisory"]["escalated"] is True

        # LLM says APPROVE for an applicant the rules engine rejects -> rejection stands.
        orchestrator._llm = _mock_llm("APPROVE")
        bad = {**underwrite_payload, "total_monthly_debt": 6000.0, "liquid_reserves": 1000.0}
        record = client.post("/underwrite", json=bad, headers=UNDERWRITER).json()
        assert record["memo"]["decision"] == "REJECT_CREDIT"
        assert record["llm_advisory"]["escalated"] is False
    finally:
        orchestrator.llm_enabled, orchestrator._llm = False, None


def test_llm_failure_degrades_gracefully(client, underwrite_payload):
    orchestrator = app.state.orchestrator
    failing = MagicMock()
    failing.generate_memo.remote.aio = AsyncMock(side_effect=RuntimeError("GPU unavailable"))
    orchestrator.llm_enabled, orchestrator._llm = True, failing
    try:
        response = client.post("/underwrite", json=underwrite_payload, headers=UNDERWRITER)
        assert response.status_code == 200
        assert response.json()["llm_advisory"]["status"] == "FAILED"
        assert "GPU unavailable" not in response.text
    finally:
        orchestrator.llm_enabled, orchestrator._llm = False, None


def test_underwrite_endpoint_missing_fields(client):
    response = client.post("/underwrite", json={"applicant_id": "APP-111"}, headers=ADMIN)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "override",
    [
        {"stated_monthly_income": -1},
        {"requested_loan_amount": 0},
        {"kyc_confidence_score": 1.5},
        {"applicant_id": "bad id with spaces"},
        {"anomaly_flags": ["x" * 201]},
    ],
)
def test_underwrite_rejects_invalid_values(client, underwrite_payload, override):
    response = client.post("/underwrite", json={**underwrite_payload, **override}, headers=ADMIN)
    assert response.status_code == 422


def test_document_endpoint(client, underwrite_payload, blank_pdf_bytes):
    response = client.post(
        "/underwrite/document",
        data={"application": json.dumps(underwrite_payload)},
        files={"file": ("statement.pdf", blank_pdf_bytes, "application/pdf")},
        headers=UNDERWRITER,
    )
    assert response.status_code == 200, response.text
    assert response.json()["source_document"] == "statement.pdf"


def test_document_endpoint_rejects_non_pdf(client, underwrite_payload):
    response = client.post(
        "/underwrite/document",
        data={"application": json.dumps(underwrite_payload)},
        files={"file": ("x.pdf", b"not a pdf", "application/pdf")},
        headers=UNDERWRITER,
    )
    assert response.status_code == 415


def test_unknown_decision_is_404(client):
    assert client.get("/decisions/does-not-exist", headers=AUDITOR).status_code == 404


def test_unhandled_errors_are_sanitized(client, underwrite_payload, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app.state.orchestrator, "underwrite", boom)
    response = client.post("/underwrite", json=underwrite_payload, headers=UNDERWRITER)
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["request_id"]


def test_rate_limit(client, underwrite_payload, monkeypatch):
    monkeypatch.setattr(underwriting_limiter, "limit", 2)
    codes = [
        client.post("/underwrite", json=underwrite_payload, headers=UNDERWRITER).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
