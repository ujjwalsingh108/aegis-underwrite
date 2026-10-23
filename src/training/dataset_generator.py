"""
To fine tune the LLM with LoRA, we need pairs of input applicant financials and their corresponding
target underwriting memos in JSON format.

This module uses FAKER to generate synthetic, policy-consistent underwriting training
samples(approvals, conditional approvals, and rejections) saved directly to a JSONL file.
"""

import json
from pathlib import Path
from typing import List
from faker import Faker

from src.common.schemas import (
    ApplicantFinancials,
    ExtractedFinancialMetrics
)
from src.engine.pipeline import UnderwritingPipeline
from src.training.lora_trainer import UnderwritingTrainingSample

fake = Faker()

class SyntheticUnderwritingDatasetGenerator:
    """Generates ground-truth pairs of applicant profiles and audited memos."""

    def __init__(self, seed: int = 42):
        Faker.seed(seed)
        self.pipeline = UnderwritingPipeline()

    def generate_sample(self, scenario_type: str = "random") -> UnderwritingTrainingSample:
        """Creates a single training sample with consistent deterministic outcomes."""
        applicant_id = f"APP-{fake.bothify(text='??-####').upper()}"

        if scenario_type == "healthy":
            # Clearly qualified: Low DTI, high reserves
            income = float(fake.random_int(min=120000, max=250000, step=5000))
            debt = float(fake.random_int(min=15000, max=30000, step=1000))
            reserves = float(fake.random_int(min=200000, max=600000, step=10000))
            verified_income = income * fake.random_element([0.98, 1.0, 1.02])
            anomaly_flags = []

        elif scenario_type == "breached":
            # High risk: High DTI or low reserves
            income = float(fake.random_int(min=40000, max=70000, step=2500))
            debt = float(fake.random_int(min=30000, max=45000, step=1000)) # > 50% DTI
            reserves = float(fake.random_int(min=5000, max=25000, step=2500))
            verified_income = income * 0.70 # Severe drop
            anomaly_flags = ["Income mismatch: verified income is 30.0% lower than stated"]

        else:
            # Borderline / Conditional
            income = float(fake.random_int(min=60000, max=100000, step=5000))
            debt = income * 0.38 # Borderline DTI
            reserves = debt * 2.0 # Tight runway (~2 months)
            verified_income = income
            anomaly_flags = []

        financials = ApplicantFinancials(
            applicant_id=applicant_id,
            stated_monthly_income=round(income, 2),
            total_monthly_debt=round(debt, 2),
            liquid_assets=round(reserves, 2)
        )

        variance_pct = round(((verified_income - income) / income) * 100, 2)

        metrics = ExtractedFinancialMetrics(
            verified_monthly_income=round(verified_income, 2),
            detected_monthly_liabilities=round(debt, 2),
            verified_liquid_reserves=round(reserves, 2),
            income_variance_pct=variance_pct,
            anomaly_flags=anomaly_flags
        )

        # Run pipeline to obtain true audited decision
        memo = self.pipeline.run_preliminary_audit(financials, metrics)

        instruction = (
            "You are an underwriting risk analyst. Review the verified applicant metrics, "
            "evaluate the financial ratios, and output the formal underwriting memo JSON."
        )

        input_context = (
            f"Applicant ID: {financials.applicant_id}\n"
            f"Stated Monthly Income: ${financials.stated_monthly_income:,.2f}\n"
            f"Verified Monthly Income: ${metrics.verified_monthly_income:,.2f}\n"
            f"Total Monthly Debt Obligations: ${financials.total_monthly_debt:,.2f}\n"
            f"Liquid Reserves: ${financials.liquid_assets:,.2f}\n"
            f"Income Variance: ${metrics.income_variance_pct}%\n"
            f"Anomaly Flags: {metrics.anomaly_flags}"
        )

        return UnderwritingTrainingSample(
            instruction=instruction,
            input_context=input_context,
            expected_memo_json=memo.model_dump_json()
        )

    def generate_and_save_dataset(
            self, output_path: str, count: int = 100
    ) -> List[UnderwritingTrainingSample]:
        """Generates a dataset split across different risk profiles and saves to JSONL."""
        samples: List[UnderwritingTrainingSample] = []
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        scenarios = ["healthy", "breached", "conditional"]

        with open(path, "w", encoding="utf-8") as f:
            for idx in range(count):
                scenario = scenarios[idx % len(scenarios)]
                sample = self.generate_sample(scenario_type=scenario)
                samples.append(sample)
                record = {
                    "instruction": sample.instruction,
                    "input_context": sample.input_context,
                    "expected_memo_json": sample.expected_memo_json
                }
                f.write(json.dumps(record) + "\n")
        return samples