"""
Test to verify prompt templating and LoRA configuration without starting a heavy
training job.
"""

from src.training.lora_trainer import (
    UnderwritingLoRATrainer,
    UnderwritingTrainingSample
)

def test_lora_config_parameters():
    trainer = UnderwritingLoRATrainer()
    lora_conf = trainer.get_lora_config()
    assert lora_conf.r == 8
    assert lora_conf.lora_alpha == 16
    assert "q_proj" in lora_conf.target_modules # type: ignore
    assert "v_proj" in lora_conf.target_modules # type: ignore

def test_prompt_formatting():
    trainer = UnderwritingLoRATrainer()
    sample = UnderwritingTrainingSample(
        instruction="Generate risk assessment.",
        input_context="Income: 100000, Debt: 40000",
        expected_memo_json='{"decision": "APPROVE"}'
    )
    prompt = trainer.format_prompt(sample)
    assert "<|im_start|>system\nGenerate risk assessment." in prompt
    assert "<|im_start|>assistant\n{\"decision\": \"APPROVE\"}" in prompt