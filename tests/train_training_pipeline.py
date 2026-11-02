"""
src/tests/test_tokenization.py

A small mock/smoke test to ensure dataset formatting and token mapping run 
cleanly locally without invoking the GPU-bound training loop or the Teacher LLM.
"""

from datasets import Dataset
from src.training.lora_trainer import UnderwritingLoRATrainer

def test_tokenization_pipeline():
    """
    Verifies that the production tokenize_function correctly maps the 
    JSONL columns into padded input_ids and attention_masks for Causal LM training.
    """
    trainer_helper = UnderwritingLoRATrainer()

    # 1. Mock the dataset structure that would normally be generated 
    # by the Teacher LLM and loaded via the jsonl file.
    mock_samples = {
        "instruction": [
            "Generate risk assessment.",
            "Generate risk assessment."
        ],
        "input_context": [
            "Applicant ID: APP-001\nIncome: 100000",
            "Applicant ID: APP-002\nIncome: 50000"
        ],
        "expected_memo_json": [
            '{"decision": "APPROVE", "calculated_dti": 0.25}',
            '{"decision": "REJECT_CREDIT", "calculated_dti": 0.65}'
        ]
    }
    
    ds = Dataset.from_dict(mock_samples)

    # 2. Execute the exact production tokenization logic
    tokenized = ds.map(
        trainer_helper.tokenize_function,
        batched=True,
        remove_columns=["instruction", "input_context", "expected_memo_json"]
    )

    # 3. Assert correct tokenization tensors were created
    assert "input_ids" in tokenized.column_names
    assert "attention_mask" in tokenized.column_names
    assert len(tokenized["input_ids"]) == 2
    
    # 4. Verify strict maximum padding was applied for batched GPU matrix math
    assert len(tokenized["input_ids"][0]) == 2048