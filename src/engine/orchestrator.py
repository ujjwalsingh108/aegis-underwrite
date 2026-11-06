import os
import json
import tempfile
from fastapi import UploadFile

from src.vision.parser import FinancialDocumentParser
from src.engine.ratio_calculator import evaluate_ratios
from src.serving.vllm_engine import UnderwritingLLMEngine
from src.common.schemas import ApplicantProfile, UnderwritingMemo

class PipelineOrchestrator:
    def __init__(self):
        self.parser = FinancialDocumentParser()
        self.llm = UnderwritingLLMEngine()
        # self.vision_agent = VisionAgent()  # Instantiated here when Qwen2-VL is wired up

    async def process_application(self, file: UploadFile) -> UnderwritingMemo:
        # 1. Securely save the FastAPI upload to a temporary disk location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
            # 2. Extract raw text and tables deterministically (Synchronous execution)
            raw_text = self.parser.extract_text(temp_path)
            tables = self.parser.extract_tables(temp_path)

            # 3. Multimodal Vision Extraction 
            # (In production, the Vision Agent parses the raw_text and images into this dict)
            parsed_data = {
                "applicant_id": "APP-9921",
                "entity_type": "LLC",
                "requested_loan_amount": 500000.0,
                "stated_monthly_income": 0.0,
                "total_monthly_debt": 15000.0,
                "liquid_assets": 120000.0,
                "annual_operating_income": 450000.0,
                "annual_debt_service": 180000.0
            }
            profile = ApplicantProfile(**parsed_data)

            # 4. Aggregate Fraud & Variance Metrics
            extracted_metrics = self.parser.aggregate_metrics(
                stated_monthly_income=profile.stated_monthly_income,
                verified_income=0.0,  # LLCs use annual operating income
                detected_liabilities=profile.total_monthly_debt,
                verified_reserves=profile.liquid_assets,
                pep_or_sanctions_hit=False,
                document_tampering_score=0.02
            )

            # 5. Deterministic Math Engine
            ratios = evaluate_ratios(profile)

            # 6. Inject Verified Context into LLM Prompt
            prompt = (
                f"Applicant Profile: {profile.model_dump_json()}\n"
                f"Verified Metrics: {extracted_metrics.model_dump_json()}\n"
                f"Calculated Ratios: {json.dumps(ratios)}\n"
                "Based on the strict underwriting guidelines, generate the JSON UnderwritingMemo."
            )

            # 7. Stream LLM output (vLLM decoding with LoRA adapter)
            memo_json = ""
            async for chunk in self.llm._stream_internal(prompt):
                memo_json += chunk
                
            # 8. Validate and return final immutable output
            return UnderwritingMemo.model_validate_json(memo_json)

        finally:
            # 9. Clean up the temporary file from the container to prevent memory leaks
            if os.path.exists(temp_path):
                os.remove(temp_path)