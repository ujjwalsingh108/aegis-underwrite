"""
src/tests/test_dataset.py

Unit tests for the local Groq-based Evol-Instruct dataset generation pipeline.
Uses mocking to bypass the Groq API network calls and sleep timers during CI/CD test runs.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.training.dataset_generator import generate_local_dataset
from src.common.schemas import UnderwritingDecision

# Mock output matching the Groq LLM's generated schema
MOCK_GROQ_OUTPUT = {
    "profile": {
        "applicant_id": "APP-GROQ-001",
        "entity_type": "INDIVIDUAL", # Switched to INDIVIDUAL so calculated_dti is populated
        "requested_loan_amount": 250000.0,
        "stated_monthly_income": 50000.0,
        "total_monthly_debt": 15000.0,
        "liquid_assets": 100000.0,
        "annual_operating_income": 0.0,
        "annual_debt_service": 0.0
    },
    "metrics": {
        "verified_monthly_income": 50000.0,
        "detected_monthly_liabilities": 15000.0,
        "verified_liquid_reserves": 100000.0,
        "pep_or_sanctions_hit": False,
        "ubo_verified": True,
        "fraud_metrics": {
            "kyc_confidence_score": 0.98,
            "document_tampering_score": 0.01,
            "synthetic_voice_probability": 0.02
        },
        "income_variance_pct": 0.0,
        "anomaly_flags": []
    }
}

@patch("src.training.dataset_generator.Groq")
@patch("src.training.dataset_generator.time.sleep")
def test_generate_and_split_dataset_logic(mock_sleep, MockGroq, tmp_path: Path):
    """
    Tests that the Groq raw JSON outputs are correctly parsed by the deterministic
    pipeline, formatted for LoRA training, and split into Train/Eval/Test files.
    """
    # 1. Setup Mocks
    mock_client_instance = MockGroq.return_value
    
    # Mock the nested Groq response object: response.choices[0].message.content
    mock_message = MagicMock()
    mock_message.content = json.dumps(MOCK_GROQ_OUTPUT)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client_instance.chat.completions.create.return_value = mock_response

    # Temporarily change directory to pytest's tmp_path to avoid polluting root directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # 2. Execute the generator function
        # Generate 10 samples to test the 80/10/10 split logic rapidly
        generate_local_dataset(total_samples=10)

        # 3. Assert File Creation & Data Splitting
        # The script creates a folder named "local_dataset"
        train_file = Path("local_dataset/train.jsonl")
        eval_file = Path("local_dataset/eval.jsonl")
        test_file = Path("local_dataset/test.jsonl")

        assert train_file.exists()
        assert eval_file.exists()
        assert test_file.exists()

        # 4. Assert Content and Schema Integrity
        with open(train_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 8  # 80% of 10 samples
            
            # Verify the structure of the first training sample
            data = json.loads(lines[0])
            assert "instruction" in data
            assert "input_context" in data
            assert "expected_memo_json" in data
            
            # Verify the pipeline correctly parsed the teacher's input into the context prompt
            assert "Applicant ID: APP-GROQ-001" in data["input_context"]
            assert "Entity Type: INDIVIDUAL" in data["input_context"]
            
            # Verify the deterministic mathematical pipeline correctly generated the ground-truth memo
            memo_data = json.loads(data["expected_memo_json"])
            assert "decision" in memo_data
            assert "calculated_dti" in memo_data
            
            # 15,000 / 50,000 = 0.30 DTI
            assert memo_data["calculated_dti"] == 0.30
            assert memo_data["decision"] in [d.value for d in UnderwritingDecision]
            
    finally:
        # Guarantee we restore the original working directory even if the test fails
        os.chdir(original_cwd)