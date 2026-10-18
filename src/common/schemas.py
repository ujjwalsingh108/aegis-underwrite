# Pydantic models for Underwriting Memo, Extracted Tables

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class UnderwritingDecision(str, Enum): # safety of enums (restricted set of values) plus the convenience of strings (easy comparison, serialization)
    """Final loan or credit underwriting recommendation."""
    APPROVE = "APPROVE"
    CONDITIONAL = "CONDITIONAL"
    REJECT = "REJECT"

class ApplicantFinancials(BaseModel):
    """Raw financial parameters submitted by the applicant or business entity."""
    applicant_id: str = Field(..., description="Unique borrower or corporate identifier")
    stated_monthly_income: float = Field(..., gt=0, description="Gross monthly income claimed by applicant")
    total_monthly_debt: float = Field(..., ge=0, description="Existing recurring monthly debt obligations")
    liquid_assets: float = Field(..., ge=0, description="Total accessible cash and equvivalents")
    annual_operating_income: Optional[float] = Field(
        default=None, ge=0, description="Annual net operating income for corporate underwriting"
    )
    annual_debt_service: Optional[float] = Field(
        default=None, ge=0, description="Total annual debt service principal and interest"
    )

class ExtractedFinancialMetrics(BaseModel):
    """Verified financial metrics extracted by document vision and OCR workers."""
    verified_monthly_income: float = Field(..., ge=0, description="Total income verified from paystubs or bank logs")
    detected_monthly_liabilities: float = Field(..., ge=0, description="Montly recurring payments detected")
    verified_liquid_reserves: float = Field(..., ge=0, description="Verified cash reserve balances")
    income_variance_pct: float = Field(
        ..., description="Percentage variance between stated and verified monthly income"
    )
    anomaly_flags: List[str] = Field(
        default_factory=list, description="List of suspicious traits, tampering indicators, or mismatches"
    )

class AuditTrialEntry(BaseModel):
    """Record of individual verification and processing steps."""
    step_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str
    details: Optional[str] = None

class UnderwritingMemo(BaseModel):
    """Final structured credit and risk memo emitted by the engine."""
    application_id: str
    decision: UnderwritingDecision
    calculated_dti: float = Field(..., description="Computed Debt-to-Income ratio")
    calculated_dscr: Optional[float] = Field(
        None,
        description="Computed Debt-Service Coverage Ratio for corporate entities"
    )
    liquidity_runway_months: float = Field(..., description="Estimated runway supported by liquid cash reserves")
    breached_policies: List[str] = Field(
        default_factory=list,
        description="Rules or limits exceeded during underwriting evaluation"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Key operational, credit, or market risks identified"
    )
    covenants: List[str] = Field(
        default_factory=list, 
        description="Stipulations or collateral conditions required for clearance"
    )
    audit_trail: List[AuditTrialEntry] = Field(
        default_factory=list,
        description="Sequential trace of execution steps"
    )
