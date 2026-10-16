"""
In financial underwritings, language models should never be fed raw, unparsed
PDF text because financial tables, loan balances, bank statements, and income slips
get distorted. We need a deterministic parser that extracts tabular financial
numbers cleanly into the ExtractedFinancialMetrics schema we defined earlier.

The module extracts raw texts and tabular data from bank statements or financial
PDFs using pdfplumber and pypdf, then parses out the figures needed for verification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber
from pypdf import PdfReader
from src.common.schemas import ExtractedFinancialMetrics

class FinancialDocumentParser:
    """Extracts raw texts, detects tables, and isolates financial metrics from documents."""

    def extract_text(self, file_path: str) -> str:
        """Extracts standard sequential text across all pages."""
        reader  = PdfReader(file_path)
        extracted_text: List[str] = []
        for page_idx, page in enumerate(reader.pages):
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

    def parse_metrics(
            self,
            stated_monthly_income: float,
            verified_income: float,
            detected_liabilities: float,
            verified_reserves: float,
    ) -> ExtractedFinancialMetrics:
        """
        Calculates variance against applicant claims and detects anomaly flags.
        """
        variance_pct = 0.0
        if stated_monthly_income > 0:
            variance_pct = round(
                ((verified_income - stated_monthly_income) / stated_monthly_income) * 100,
                2
            )
        anomaly_flags: List[str] = []
        if verified_income < stated_monthly_income * 0.85:
            anomaly_flags.append(
                f"Income mismatch: verified income is {abs(variance_pct):.1f}% lower than stated"
            )
        if detected_liabilities <= 0:
            anomaly_flags.append("Warning: zero recurring liabilities detected in document records")

        return ExtractedFinancialMetrics(
            verified_monthly_income=verified_income,
            detected_monthly_liabilities=detected_liabilities,
            verified_liquid_reserves=verified_reserves,
            income_variance_pct=variance_pct,
            anomaly_flags=anomaly_flags
        )