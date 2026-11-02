"""
src/vision/parser.py

In financial underwriting, language models should never be fed raw, unparsed
PDF text because financial tables, loan balances, bank statements, and income slips
get distorted. 

This module acts as the deterministic parsing gateway. It extracts tabular data 
using pdfplumber, and aggregates outputs from our Vision/Audio AI agents to construct 
the strictly validated ExtractedFinancialMetrics payload required by the pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber
from pypdf import PdfReader

from src.common.schemas import (
    ExtractedFinancialMetrics,
    MultimodalFraudMetrics
)

class FinancialDocumentParser:
    """Extracts raw texts, detects tables, and aggregates multimodal metrics."""

    def extract_text(self, file_path: str) -> str:
        """Extracts standard sequential text across all pages."""
        reader = PdfReader(file_path)
        extracted_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)

    def extract_tables(self, file_path: str) -> List[List[List[Optional[str]]]]:
        """Extracts structured grid tables across all pages using bounding boxes."""
        all_tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        return all_tables

    def aggregate_metrics(
            self,
            stated_monthly_income: float,
            verified_income: float,
            detected_liabilities: float,
            verified_reserves: float,
            verified_collateral_value: Optional[float] = None,
            pep_or_sanctions_hit: bool = False,
            ubo_verified: bool = True,
            kyc_confidence_score: float = 0.95,
            document_tampering_score: float = 0.01,
            synthetic_voice_probability: Optional[float] = None
    ) -> ExtractedFinancialMetrics:
        """
        Calculates variance against applicant claims, compiles anomaly flags, 
        and maps inputs to the production ExtractedFinancialMetrics schema.
        """
        variance_pct = 0.0
        if stated_monthly_income > 0:
            variance_pct = round(
                ((verified_income - stated_monthly_income) / stated_monthly_income) * 100,
                2
            )
            
        anomaly_flags: List[str] = []
        
        # Credit Anomalies
        if verified_income < stated_monthly_income * 0.85:
            anomaly_flags.append(
                f"Income mismatch: verified income is {abs(variance_pct):.1f}% lower than stated"
            )
        if detected_liabilities <= 0:
            anomaly_flags.append("Warning: zero recurring liabilities detected in document records")
            
        # Fraud & Compliance Anomalies
        if document_tampering_score > 0.15:
            anomaly_flags.append(f"High probability of document tampering/Photoshop metadata detected ({document_tampering_score:.1%}).")
        if pep_or_sanctions_hit:
            anomaly_flags.append("CRITICAL: Applicant or UBO matches active OFAC/Sanctions watchlist.")

        # Construct nested fraud metrics
        fraud_metrics = MultimodalFraudMetrics(
            kyc_confidence_score=kyc_confidence_score,
            document_tampering_score=document_tampering_score,
            synthetic_voice_probability=synthetic_voice_probability
        )

        return ExtractedFinancialMetrics(
            verified_monthly_income=verified_income,
            detected_monthly_liabilities=detected_liabilities,
            verified_liquid_reserves=verified_reserves,
            verified_collateral_value=verified_collateral_value,
            pep_or_sanctions_hit=pep_or_sanctions_hit,
            ubo_verified=ubo_verified,
            fraud_metrics=fraud_metrics,
            income_variance_pct=variance_pct,
            anomaly_flags=anomaly_flags
        )