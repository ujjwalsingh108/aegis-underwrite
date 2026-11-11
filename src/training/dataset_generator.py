import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import modal

from src.common.config import settings
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
)
from src.engine.pipeline import UnderwritingPipeline
from src.training.lora_trainer import UnderwritingTrainingSample
from src.serving.modal_app import app, app_volume

# We removed the GPU image. We only need a lightweight CPU container for API calls.
api_image = modal.Image.debian_slim(python_version="3.11").pip_install("openai", "pydantic")

@app.cls(
    image=api_image,
    secrets=[modal.Secret.from_name("teacher-api-key")], # E.g., OpenRouter or TogetherAI key
    timeout=1800,
)
class TeacherAPIClient:
    """Uses a Pay-Per-Token API instead of hosting 405B on Modal GPUs."""
    
    @modal.enter()
    def setup_client(self):
        from openai import AsyncOpenAI
        # Example using Together AI (or OpenRouter/Groq by changing the base_url)
        self.client = AsyncOpenAI(
            api_key=os.environ["TEACHER_API_KEY"],
            base_url="https://api.together.xyz/v1", 
        )

    @modal.method()
    async def generate_diverse_applicant(self, seed: int) -> Dict[str, Any]:
        """Prompts the external API teacher to generate a financial edge case."""
        
        # Define the exact JSON structure we want back
        combined_schema = {
            "type": "object",
            "properties": {
                "profile": ApplicantProfile.model_json_schema(),
                "metrics": ExtractedFinancialMetrics.model_json_schema()
            },
            "required": ["profile", "metrics"]
        }

        prompt = (
            "You are a financial red-teamer. Generate a highly complex, edge-case applicant "
            "profile for an underwriting system. Create seasonal workers, trust funds, "
            "shell LLCs, or applicants with high fraud probability and minor policy breaches.\n"
            f"Scenario seed: {seed}"
        )

        response = await self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful JSON data generator."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema", 
                "json_schema": {"name": "applicant_data", "schema": combined_schema}
            },
            temperature=0.7 + (seed % 3) * 0.1,
        )
        
        return json.loads(response.choices[0].message.content) # type: ignore


@app.function(volumes={"/vol": app_volume}, timeout=3600)
def generate_and_split_dataset(total_samples: int = 3000):
    """Generates data via API, runs the pipeline, and splits into Train/Eval/Test."""
    print(f"Starting Evol-Instruct API generation for {total_samples} samples...")
    
    teacher = TeacherAPIClient()
    pipeline = UnderwritingPipeline()
    
    # Executes the API calls concurrently
    raw_inputs = list(teacher.generate_diverse_applicant.map(range(total_samples)))
    
    dataset: List[UnderwritingTrainingSample] = []
    
    print("Processing Teacher outputs through Deterministic Pipeline...")
    for data in raw_inputs:
        try:
            profile = ApplicantProfile.model_validate(data["profile"])
            metrics = ExtractedFinancialMetrics.model_validate(data["metrics"])
            
            memo = pipeline.run_preliminary_audit(profile, metrics)
            
            instruction = (
                "You are an underwriting risk analyst. Review the verified applicant metrics, "
                "evaluate the financial ratios, and output the formal underwriting memo JSON."
            )
            
            input_context = (
                f"Applicant ID: {profile.applicant_id}\n"
                f"Entity Type: {profile.entity_type.value}\n"
                f"Stated Monthly Income: ${profile.stated_monthly_income:,.2f}\n"
                f"Verified Monthly Income: ${metrics.verified_monthly_income:,.2f}\n"
                f"Total Monthly Debt Obligations: ${profile.total_monthly_debt:,.2f}\n"
                f"Liquid Reserves: ${profile.liquid_assets:,.2f}\n"
                f"Anomaly Flags: {metrics.anomaly_flags}"
            )
            
            dataset.append(UnderwritingTrainingSample(
                instruction=instruction,
                input_context=input_context,
                expected_memo_json=memo.model_dump_json()
            ))
        except Exception as e:
            print(f"Skipping malformed sample: {e}")

    random.seed(42)
    random.shuffle(dataset)
    
    train_idx = int(len(dataset) * 0.8)
    eval_idx = int(len(dataset) * 0.9)
    
    splits = {
        "train.jsonl": dataset[:train_idx],
        "eval.jsonl": dataset[train_idx:eval_idx],
        "test.jsonl": dataset[eval_idx:]
    }
    
    data_dir = Path("/vol/dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, samples in splits.items():
        file_path = data_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps({
                    "instruction": sample.instruction,
                    "input_context": sample.input_context,
                    "expected_memo_json": sample.expected_memo_json
                }) + "\n")
        print(f"Saved {len(samples)} samples to {file_path}")

@app.local_entrypoint()
def main():
    generate_and_split_dataset.remote(total_samples=3000)