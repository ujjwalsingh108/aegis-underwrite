"""
src/tests/test_dataset_generator.py

Unit tests for the Evol-Instruct dataset generation pipeline.
Uses mocking to bypass the GPU-bound vLLM Teacher Engine and Modal volume 
requirements during CI/CD test runs.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.training.dataset_generator import generate_and_split_dataset
from src.common.schemas import UnderwritingDecision

# Mock output matching the Teacher LLM's highly complex generated schema
MOCK_TEACHER_OUTPUT = {
    "profile": {
        "applicant_id": "APP-TEACHER-001",
        "entity_type": "LLC",
        "requested_loan_amount": 250000.0,
        "stated_monthly_income": 50000.0,
        "total_monthly_debt": 15000.0,
        "liquid_assets": 100000.0,
        "stated_collateral_value": 300000.0
    },
    "metrics": {
        "verified_monthly_income": 50000.0,
        "detected_monthly_liabilities": 15000.0,
        "verified_liquid_reserves": 100000.0,
        "verified_collateral_value": 300000.0,
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

@patch("src.training.dataset_generator.TeacherLLMEngine")
@patch("src.training.dataset_generator.Path")
def test_generate_and_split_dataset_logic(mock_path_class, mock_teacher_class, tmp_path: Path):
    """
    Tests that the teacher's raw JSON outputs are correctly parsed by the deterministic
    pipeline, formatted for LoRA training, and split into Train/Eval/Test files.
    """
    # 1. Setup Mocks
    # Redirect the Modal /vol/dataset path to our local pytest tmp_path
    mock_path_class.return_value = tmp_path
    
    # Mock the TeacherLLMEngine and its Modal .map() method to return 10 identical mock outputs
    mock_engine_instance = MagicMock()
    mock_engine_instance.generate_diverse_applicant.map.return_value = [MOCK_TEACHER_OUTPUT for _ in range(10)]
    mock_teacher_class.return_value = mock_engine_instance

    # 2. Execute the generator function
    # Because generate_and_split_dataset is decorated with @app.function, 
    # we must call .local() to execute the underlying Python logic without triggering a Modal cloud job.
    generate_and_split_dataset.local(total_samples=10)

    # 3. Assert File Creation & Data Splitting (80/10/10 of 10 samples = 8, 1, 1)
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    test_file = tmp_path / "test.jsonl"

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
        assert "Applicant ID: APP-TEACHER-001" in data["input_context"]
        assert "Entity Type: LLC" in data["input_context"]
        assert "Fraud Confidence" in data["input_context"]
        
        # Verify the deterministic mathematical pipeline correctly generated the ground-truth memo
        memo_data = json.loads(data["expected_memo_json"])
        assert "decision" in memo_data
        assert "calculated_dti" in memo_data
        
        # 15,000 / 50,000 = 0.30 DTI
        assert memo_data["calculated_dti"] == 0.30
        assert memo_data["decision"] in [d.value for d in UnderwritingDecision]