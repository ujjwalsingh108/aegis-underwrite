import io
import os
import sys
from unittest.mock import MagicMock

# --- Deterministic test configuration (env vars take precedence over a local .env) ---
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "MASTER_API_KEY": "test-master-key-0000000000",
        "UNDERWRITER_API_KEY": "test-underwriter-key-000000",
        "AUDITOR_API_KEY": "test-auditor-key-0000000000",
        "DECISION_STORE_PATH": ":memory:",
        "LLM_REVIEW_ENABLED": "false",
        "RATE_LIMIT_PER_MINUTE": "1000",
    }
)

# --- Windows Compatibility Mock ---
# vLLM requires uvloop, which is Linux-only and fails on Windows.
sys.modules["uvloop"] = MagicMock()
vllm_mock = MagicMock()
sys.modules["vllm"] = vllm_mock
sys.modules["vllm.lora"] = vllm_mock
sys.modules["vllm.lora.request"] = vllm_mock
sys.modules["vllm.sampling_params"] = vllm_mock
# ----------------------------------

import pytest  # noqa: E402

from src.common.schemas import ApplicantProfile, EntityType  # noqa: E402


@pytest.fixture
def valid_individual_profile():
    return ApplicantProfile(
        applicant_id="APP-1234",
        entity_type=EntityType.INDIVIDUAL,
        requested_loan_amount=200000.0,
        stated_monthly_income=10000.0,
        total_monthly_debt=3500.0,
        liquid_assets=50000.0,
        annual_operating_income=0.0,
        annual_debt_service=0.0
    )


@pytest.fixture
def valid_llc_profile():
    return ApplicantProfile(
        applicant_id="APP-9999",
        entity_type=EntityType.LLC,
        requested_loan_amount=1000000.0,
        stated_monthly_income=0.0,
        total_monthly_debt=20000.0,
        liquid_assets=250000.0,
        annual_operating_income=800000.0,
        annual_debt_service=240000.0,
        stated_collateral_value=1500000.0
    )


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def underwrite_payload() -> dict:
    return {
        "applicant_id": "APP-111",
        "requested_loan_amount": 150000.0,
        "stated_monthly_income": 10000.0,
        "verified_monthly_income": 9500.0,
        "total_monthly_debt": 3000.0,
        "liquid_reserves": 50000.0,
        "kyc_confidence_score": 0.97,
        "document_tampering_score": 0.02,
        "anomaly_flags": [],
    }
