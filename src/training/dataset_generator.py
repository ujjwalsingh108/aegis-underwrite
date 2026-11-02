"""
src/training/dataset_generator.py

Modern Evol-Instruct Dataset Generation using a Teacher LLM.
This script uses a larger open-weights model to generate diverse financial 
edge-cases, processes them through our deterministic pipeline, and saves 
the resulting instruction pairs to the Modal network volume for LoRA training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import modal

from src.common.config import settings
from src.common.schemas import (
    ApplicantProfile,
    ExtractedFinancialMetrics,
    MultimodalFraudMetrics,
    EntityType
)
from src.engine.pipeline import UnderwritingPipeline
from src.training.lora_trainer import UnderwritingTrainingSample
from src.serving.modal_app import app, app_volume, gpu_image

# 1. Define the Teacher LLM Engine on Modal
@app.cls(
    image=gpu_image,
    gpu=settings.gpu_type,  # You may need A100s/H100s for massive teacher models
    volumes={"/vol": app_volume},
    secrets=[modal.Secret.from_name("underwriting_hf_token")],
    timeout=1800, # Longer timeout for dataset generation
)
class TeacherLLMEngine:
    """Runs the large Teacher Model (e.g., Llama-3.1-70B/405B) to generate diverse inputs."""
    
    @modal.enter()
    def load_teacher(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        print(f"Loading Teacher Model: {settings.teacher_model_name}")
        
        engine_args = AsyncEngineArgs(
            model=settings.teacher_model_name,
            tensor_parallel_size=1, # Increase if using multi-GPU for 70B+ models
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=False,
            trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate_diverse_applicant(self, seed: int) -> Dict[str, Any]:
        """Prompts the teacher to generate a highly complex financial edge case."""
        import uuid
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams # type: ignore
        
        # We force the Teacher to output BOTH input schemas at once
        combined_schema = {
            "type": "object",
            "properties": {
                "profile": ApplicantProfile.model_json_schema(),
                "metrics": ExtractedFinancialMetrics.model_json_schema()
            },
            "required": ["profile", "metrics"]
        }
        
        guided_params = GuidedDecodingParams(json=json.dumps(combined_schema))
        sampling_params = SamplingParams(
            temperature=0.7 + (seed % 3) * 0.1, # Varies creativity
            max_tokens=2048,
            guided_decoding=guided_params # type: ignore
        )

        prompt = (
            "<|im_start|>system\n"
            "You are a financial red-teamer. Generate a highly complex, edge-case applicant "
            "profile for an underwriting system. Create seasonal workers, trust funds, "
            "shell LLCs, or applicants with high fraud probability and minor policy breaches. "
            "Output strictly valid JSON matching the requested schema.<|im_end|>\n"
            f"<|im_start|>user\nScenario seed: {seed}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        request_id = str(uuid.uuid4())
        results = self.engine.generate(prompt, sampling_params, request_id)
        
        full_text = ""
        async for output in results:
            full_text = output.outputs[0].text
            
        # Clean and parse the Teacher's JSON output
        clean_json = full_text.removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean_json)


# 2. Orchestrate Generation, Pipeline Execution, and File Splitting
@app.function(volumes={"/vol": app_volume}, timeout=3600)
def generate_and_split_dataset(total_samples: int = 3000):
    """Generates data in parallel, runs the pipeline, and splits into Train/Eval/Test."""
    print(f"Starting Evol-Instruct generation for {total_samples} samples...")
    
    teacher = TeacherLLMEngine()
    pipeline = UnderwritingPipeline()
    
    # Generate diverse inputs in parallel using Modal's .map()
    # Note: map() handles the concurrency automatically based on available GPU workers
    raw_inputs = list(teacher.generate_diverse_applicant.map(range(total_samples)))
    
    dataset: List[UnderwritingTrainingSample] = []
    
    print("Processing Teacher outputs through Deterministic Pipeline...")
    for data in raw_inputs:
        try:
            profile = ApplicantProfile.model_validate(data["profile"])
            metrics = ExtractedFinancialMetrics.model_validate(data["metrics"])
            
            # Ground Truth: The deterministic pipeline calculates the PERFECT math and decision
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
                f"Collateral Value: ${profile.stated_collateral_value or 0:,.2f}\n"
                f"UBO Verified: {metrics.ubo_verified}\n"
                f"PEP/Sanctions Hit: {metrics.pep_or_sanctions_hit}\n"
                f"Fraud Confidence (Tampering): {metrics.fraud_metrics.document_tampering_score:.2f}\n"
                f"Anomaly Flags: {metrics.anomaly_flags}"
            )
            
            dataset.append(UnderwritingTrainingSample(
                instruction=instruction,
                input_context=input_context,
                expected_memo_json=memo.model_dump_json()
            ))
        except Exception as e:
            print(f"Skipping malformed sample: {e}")

    # 3. Shuffle and Split the Dataset
    print(f"Successfully generated {len(dataset)} valid samples. Shuffling and splitting...")
    random.seed(42)
    random.shuffle(dataset)
    
    # 80% Train, 10% Eval, 10% Test split
    train_idx = int(len(dataset) * 0.8)
    eval_idx = int(len(dataset) * 0.9)
    
    splits = {
        "train.jsonl": dataset[:train_idx],
        "eval.jsonl": dataset[train_idx:eval_idx],
        "test.jsonl": dataset[eval_idx:]
    }
    
    # 4. Save to Modal Persistent Volume
    data_dir = Path("/vol/dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, samples in splits.items():
        file_path = data_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "instruction": sample.instruction,
                    "input_context": sample.input_context,
                    "expected_memo_json": sample.expected_memo_json
                }
                f.write(json.dumps(record) + "\n")
        print(f"Saved {len(samples)} samples to {file_path}")

@app.local_entrypoint()
def main():
    """Trigger the cloud generation process from your local terminal."""
    generate_and_split_dataset.remote(total_samples=3000)