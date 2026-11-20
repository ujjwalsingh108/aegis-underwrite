"""
Offline teacher-data generation: Groq produces edge-case applicants, the deterministic
rules engine produces the ground-truth memo, and rows are written in the canonical prompt
format shared with serving (src/common/prompts.py).
"""

import json
import random
import time
from pathlib import Path

from groq import Groq, RateLimitError

from src.common.config import settings
from src.common.logger import get_logger
from src.common.prompts import SYSTEM_INSTRUCTION, build_input_context
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
    UnderwritingTrainingSample,
)
from src.engine.pipeline import UnderwritingPipeline

logger = get_logger("dataset_generator")


def generate_local_dataset(total_samples: int = 3000):
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    client = Groq(api_key=api_key)
    pipeline = UnderwritingPipeline()
    dataset = []

    # 1. Force Groq to respect your exact schemas
    combined_schema = {
        "type": "object",
        "properties": {
            "profile": ApplicantProfile.model_json_schema(),
            "metrics": ExtractedFinancialMetrics.model_json_schema()
        },
        "required": ["profile", "metrics"]
    }
    
    system_prompt = (
        "You are a financial red-teamer generating data for an underwriting system. "
        "Output strictly valid JSON matching this exact schema: "
        f"{json.dumps(combined_schema)}"
    )

    logger.info("generation_started", extra={"total_samples": total_samples})
    
    samples_generated = 0
    seed = 0

    # 2. Rate Limit & Error Handling Loop
    while samples_generated < total_samples:
        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a highly complex, edge-case applicant profile. Seed: {seed}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7 + (seed % 3) * 0.1
            )
            
            data = json.loads(response.choices[0].message.content) # type: ignore
            
            # Deterministic Math Evaluation
            profile = ApplicantProfile.model_validate(data["profile"])
            metrics = ExtractedFinancialMetrics.model_validate(data["metrics"])
            memo = pipeline.run_preliminary_audit(profile, metrics)
            
            # Canonical prompt format shared with the serving path
            row = UnderwritingTrainingSample(
                instruction=SYSTEM_INSTRUCTION,
                input_context=build_input_context(profile, metrics),
                expected_memo_json=memo.model_dump_json(),
            )
            dataset.append(row.model_dump())
            
            samples_generated += 1
            seed += 1
            logger.info("sample_generated", extra={"done": samples_generated, "total": total_samples})
            
            # Pause to respect Groq free tier Tokens Per Minute limits
            time.sleep(3) 

        except RateLimitError:
            logger.warning("groq_rate_limited", extra={"sleep_seconds": 60})
            time.sleep(60)
        except Exception as e:
            logger.warning("sample_rejected", extra={"seed": seed, "error": str(e)[:300]})
            seed += 1 # Skip this seed and try the next one to avoid infinite loops

    # 3. Shuffle and split the dataset dynamically (80% / 10% / 10%)
    logger.info("generation_complete_splitting")
    random.seed(42)
    random.shuffle(dataset)

    train_idx = int(len(dataset) * 0.8)
    eval_idx = int(len(dataset) * 0.9)

    splits = {
        "train.jsonl": dataset[:train_idx],
        "eval.jsonl": dataset[train_idx:eval_idx],
        "test.jsonl": dataset[eval_idx:]
    }

    # 4. Save locally
    data_dir = Path("local_dataset")
    data_dir.mkdir(exist_ok=True)
    
    for filename, samples in splits.items():
        file_path = data_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        logger.info("split_saved", extra={"records": len(samples), "path": str(file_path)})

if __name__ == "__main__":
    generate_local_dataset(total_samples=3000)