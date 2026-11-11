import json
import random
import os
import time
from pathlib import Path
from groq import Groq, RateLimitError

from src.common.schemas import ApplicantProfile, ExtractedFinancialMetrics
from src.engine.pipeline import UnderwritingPipeline

def generate_local_dataset(total_samples: int = 3000):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
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

    print(f"Starting generation of {total_samples} samples via Groq API...")
    
    samples_generated = 0
    seed = 0

    # 2. Rate Limit & Error Handling Loop
    while samples_generated < total_samples:
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
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
            
            # Format exactly how the LoRA adapter needs to see it
            input_context = (
                f"Applicant ID: {profile.applicant_id}\n"
                f"Entity Type: {profile.entity_type.value}\n"
                f"Stated Monthly Income: ${profile.stated_monthly_income:,.2f}\n"
                f"Verified Monthly Income: ${metrics.verified_monthly_income:,.2f}\n"
                f"Total Monthly Debt Obligations: ${profile.total_monthly_debt:,.2f}\n"
                f"Liquid Reserves: ${profile.liquid_assets:,.2f}\n"
                f"Anomaly Flags: {metrics.anomaly_flags}"
            )
            
            dataset.append({
                "instruction": "You are an underwriting risk analyst. Review the verified applicant metrics, evaluate the financial ratios, and output the formal underwriting memo JSON.",
                "input_context": input_context,
                "expected_memo_json": memo.model_dump_json()
            })
            
            samples_generated += 1
            seed += 1
            print(f"Success: {samples_generated}/{total_samples} generated.")
            
            # Pause to respect Groq free tier Tokens Per Minute limits
            time.sleep(3) 

        except RateLimitError:
            print("Groq Rate limit hit. Sleeping for 60 seconds before retrying...")
            time.sleep(60)
        except Exception as e:
            print(f"Validation or parsing error on seed {seed}: {e}")
            seed += 1 # Skip this seed and try the next one to avoid infinite loops

    # 3. Shuffle and split the dataset exactly as requested
    print("Generation complete. Shuffling and splitting dataset...")
    random.seed(42)
    random.shuffle(dataset)

    splits = {
        "train.jsonl": dataset[:2000],
        "eval.jsonl": dataset[2000:2500],
        "test.jsonl": dataset[2500:3000]
    }

    # 4. Save locally
    data_dir = Path("local_dataset")
    data_dir.mkdir(exist_ok=True)
    
    for filename, samples in splits.items():
        file_path = data_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} records to {file_path}")

if __name__ == "__main__":
    generate_local_dataset(total_samples=3000)