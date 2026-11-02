"""
src/tests/test_lora_trainer.py

Test to verify prompt templating and production LoRA configuration 
without starting a heavy GPU-bound training job.
"""

from src.training.lora_trainer import UnderwritingLoRATrainer

def test_lora_config_parameters():
    trainer = UnderwritingLoRATrainer()
    lora_conf = trainer.get_lora_config()
    
    # Assert production-grade rank and alpha scaling for complex JSON reasoning
    assert lora_conf.r == 16
    assert lora_conf.lora_alpha == 32
    
    # Assert comprehensive projection targeting
    assert lora_conf.target_modules == "all-linear"

def test_prompt_formatting():
    trainer = UnderwritingLoRATrainer()
    
    # The modern pipeline uses Hugging Face Datasets, which maps over dictionaries
    sample = {
        "instruction": "Generate risk assessment.",
        "input_context": "Income: 100000, Debt: 40000",
        "expected_memo_json": '{"decision": "APPROVE"}'
    }
    
    prompt = trainer.format_prompt(sample)
    
    # Verify strict Qwen ChatML token formatting
    assert "<|im_start|>system\nGenerate risk assessment.<|im_end|>" in prompt
    assert "<|im_start|>user\nIncome: 100000, Debt: 40000<|im_end|>" in prompt
    assert "<|im_start|>assistant\n{\"decision\": \"APPROVE\"}<|im_end|>" in prompt