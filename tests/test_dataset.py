"""Dataset Generator Unit Tests"""

import json
from pathlib import Path
from src.training.dataset_generator import SyntheticUnderwritingDatasetGenerator

def test_generate_sample():
    generator = SyntheticUnderwritingDatasetGenerator()
    sample = generator.generate_sample(scenario_type="healthy")
    assert "Applicant ID:" in sample.input_context
    assert "Stated Monthly Income:" in sample.input_context

    memo_data = json.loads(sample.expected_memo_json)
    assert "decision" in memo_data
    assert "calculated_dti" in memo_data
    assert memo_data["decision"] in ["APPROVE", "CONDITIONAL", "REJECT"]

def test_generate_and_save_dataset(tmp_path: Path):
    generator = SyntheticUnderwritingDatasetGenerator()
    dataset_file = tmp_path / "train_samples.jsonl"
    samples = generator.generate_and_save_dataset(str(dataset_file), count=6)

    assert len(samples) == 6
    assert dataset_file.exists()

    with open(dataset_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 6
        for line in lines:
            data = json.loads(line)
            assert "instruction" in data
            assert "input_context" in data
            assert "expected_memo_json" in data