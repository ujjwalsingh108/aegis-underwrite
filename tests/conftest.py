import sys
import pytest
from unittest.mock import MagicMock
from src.common.schemas import ApplicantProfile, EntityType


# --- Windows Compatibility Mock ---
# vLLM requires uvloop, which is Linux-only and fails on Windows.
# This intercepts the imports globally so pytest can run locally.
sys.modules['uvloop'] = MagicMock()
vllm_mock = MagicMock()
sys.modules['vllm'] = vllm_mock
sys.modules['vllm.lora'] = vllm_mock
sys.modules['vllm.lora.request'] = vllm_mock
sys.modules['vllm.sampling_params'] = vllm_mock
# ----------------------------------

import pytest
from src.common.schemas import ApplicantProfile

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