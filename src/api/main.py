"""
FastAPI web gateway for the automated underwriting pipeline.
"""

from typing import List
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.auth import AuthenticatedUser, RoleChecker, UserRole
from src.common.config import settings
from src.common.schemas import UnderwritingMemo
from src.serving.vllm_engine import UnderwritingLLMEngine

app = FastAPI(
    title="Aegis Underwriting API",
    description="Automated LLM-driven underwriting pipeline.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RBAC Role Guards ---
allow_underwriting = RoleChecker([UserRole.ADMIN, UserRole.UNDERWRITER])
allow_audit_read = RoleChecker([UserRole.ADMIN, UserRole.UNDERWRITER, UserRole.AUDITOR])
allow_admin_only = RoleChecker([UserRole.ADMIN])


class UnderwriteRequest(BaseModel):
    """Incoming JSON payload from a frontend or loan origination system."""
    applicant_id: str
    stated_monthly_income: float
    verified_monthly_income: float
    total_monthly_debt: float
    liquid_reserves: float
    anomaly_flags: List[str] = []


@app.get("/health")
async def health_check():
    """Public liveness probe for Modal and uptime monitors."""
    return {"status": "ok", "environment": settings.environment}


@app.get("/auth/me", response_model=AuthenticatedUser)
async def read_current_user(
    current_user: AuthenticatedUser = Depends(allow_audit_read),
):
    """Returns the authenticated caller's identity and assigned RBAC role."""
    return current_user


@app.post("/underwrite", response_model=UnderwritingMemo)
async def process_underwriting(
    request: UnderwriteRequest,
    current_user: AuthenticatedUser = Depends(allow_underwriting),
):
    """Generates an underwriting memo. Restricted to ADMIN and UNDERWRITER roles."""

    # 1. Calculate variance mathematically
    variance = 0.0
    if request.stated_monthly_income > 0:
        variance = (
            (request.verified_monthly_income - request.stated_monthly_income)
            / request.stated_monthly_income
        ) * 100

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
        memo = await engine.generate_memo.remote.aio(prompt)  # Non-blocking
        return memo
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"LLM Schema Generation Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))