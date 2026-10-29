"""
A small mock/smoke test to ensure dataset formatting and token mapping run 
cleanly locally without invoking remote computer.
"""

from datasets import Dataset
from src.training.dataset_generator import SyntheticUnderwritingDatasetGenerator
from src.training.lora_trainer import UnderwritingLoRATrainer

def test_tokenization_pipeline():
    trainer_helper = UnderwritingLoRATrainer()
    tokenizer = trainer_helper.tokenizer

    data_gen = SyntheticUnderwritingDatasetGenerator()
    samples = [data_gen.generate_sample("healthy") for _ in range(2)]

    formatted = [trainer_helper.format_prompt(s) for s in samples]
    ds = Dataset.from_dict({"text": formatted})

    tokenized = ds.map(
        lambda x: tokenizer(x["text"], max_length=512, truncation=True),
        batched=True
    )

    assert "input_ids" in tokenized.column_names
    assert len(tokenized["input_ids"]) == 2