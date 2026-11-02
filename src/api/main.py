"""
FastAPI web gateway for the automated underwriting pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.common.schemas import UnderwritingMemo
from src.serving.vllm_engine import UnderwritingLLMEngine

app = FastAPI(
    title="Aegis Underwriting API",
    description="Automated LLM-driven underwriting pipeline.",
    version="1.0.0"
)

class UnderwriteRequest(BaseModel):
    """Incoming JSON payload from a frontend or loan origination system."""
    applicant_id: str
    stated_monthly_income: float
    verified_monthly_income: float
    total_monthly_debt: float
    liquid_reserves: float
    anomaly_flags: List[str] = []

@app.post("/underwrite", response_model=UnderwritingMemo)
async def process_underwriting(request: UnderwriteRequest):
    """Generates an underwriting memo for the provided applicant financials."""

    # 1. Calculate variance mathematically
    variance = 0.0
    if request.stated_monthly_income > 0:
        variance = ((request.verified_monthly_income - request.stated_monthly_income) / request.stated_monthly_income) * 100

    # 2. Format the prompt exactly as it was during SFT training
    prompt = (
        "<|im_start|>system\n"
        "You are an underwriting risk analyst. Review the verified applicant metrics, "
        "evaluate the financial ratios, and output the formal underwriting memo JSON "
        "with keys: application_id, decision, calculated_dti, calculated_dscr, "
        "liquidity_runway_months, breached_policies, risk_factors, covenants, audit_trail.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Applicant ID: {request.applicant_id}\n"
        f"Stated Monthly Income: ${request.stated_monthly_income:,.2f}\n"
        f"Verified Monthly Income: ${request.verified_monthly_income:,.2f}\n"
        f"Total Monthly Debt Obligations: ${request.total_monthly_debt:,.2f}\n"
        f"Liquid Reserves: ${request.liquid_reserves:,.2f}\n"
        f"Income Variance: {variance:.1f}%\n"
        f"Anomaly Flags: {request.anomaly_flags}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 3. Instantiate the Modal GPU worker proxy
    engine = UnderwritingLLMEngine()

    # 4. Dispatch the request to the GPU container and return the Pydantic model
    try:
        memo = engine.generate_memo.remote(prompt)
        return memo
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"LLM Schema Generation Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

