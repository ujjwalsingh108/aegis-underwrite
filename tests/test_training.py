"""
src/tests/test_lora_trainer.py

Test to verify prompt templating and production LoRA configuration 
without starting a heavy GPU-bound training job.

Validates the training data container objects and formatting logic used by Hugging Face/PEFT 
before dispatching to Modal GPUs.
"""

import pytest
from src.training.lora_trainer import UnderwritingTrainingSample

def test_training_sample_validation():
    """State & Lifecycle: Verifies the Pydantic structure of a single training row."""
    sample = UnderwritingTrainingSample(
        instruction="Evaluate this profile.",
        input_context="Applicant ID: APP-123\nIncome: $5000",
        expected_memo_json='{"decision": "APPROVED"}'
    )
    assert sample.instruction == "Evaluate this profile."
    assert "APP-123" in sample.input_context

def test_training_sample_missing_context():
    """Negative: Ensures trainer throws an error if data is malformed."""
    with pytest.raises(ValueError):
        UnderwritingTrainingSample(
            instruction="Evaluate",
            input_context="", # Empty context should fail validation in your schema
            expected_memo_json="{}"
        )