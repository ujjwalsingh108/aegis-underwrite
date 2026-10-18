"""
Document Parser Verification with Synthetic Fixtures

Before wiring into an API gateway, we need automated tests verifying that, can read
PDF bytes and extract tables without breaking
"""

from pathlib import Path
import pytest
from pypdf import PdfWriter
from src.vision.parser import FinancialDocumentParser

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Generates a minimal valid PDF file dynamically for unit testing"""
    pdf_file = tmp_path / "sample_statement.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(pdf_file, "wb") as f:
        writer.write(f)
    return str(pdf_file)

def test_parser_extracts_text_from_valid_pdf(sample_pdf_path: str):
    parser = FinancialDocumentParser()
    extracted_text = parser.extract_text(sample_pdf_path)
    assert isinstance(extracted_text, str)

def test_parser_metric_calculation_and_variance():
    parser = FinancialDocumentParser()
    metrics = parser.parse_metrics(
        stated_monthly_income=100000.0,
        verified_income=80000.0,
        detected_liabilities=25000.0,
        verified_reserves=150000.0
    )
    assert metrics.verified_monthly_income == 80000.0
    assert metrics.income_variance_pct == -20.0
    assert len(metrics.anomaly_flags) == 1
    assert "Income mismatch" in metrics.anomaly_flags[0]