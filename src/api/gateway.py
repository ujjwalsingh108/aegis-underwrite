"""
FastAPI ingestion gateway
"""

from fastapi import FastAPI, HTTPException, status
from src.common.config import settings
from src.common.schemas import (
    ApplicantFinancials,
    ExtractedFinancialMetrics,
    UnderwritingMemo
)
from src.engine.pipeline import UnderwritingPipeline

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Automated Underwriting and Financial Risk Engine"
)

pipeline = UnderwritingPipeline()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "app": settings.app_name}

@app.post(
    "/underwrite",
    response_model=UnderwritingMemo,
    status_code=status.HTTP_200_OK,
)
def evaluate_application(
    financials: ApplicantFinancials,
    metrics: ExtractedFinancialMetrics
):
    try:
        memo = pipeline.run_preliminary_audit(
            financials=financials,
            metrics=metrics
        )
        return memo
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )