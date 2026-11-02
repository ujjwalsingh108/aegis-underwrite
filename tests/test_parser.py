"""
src/tests/test_parser.py

Document Parser Verification with Synthetic Fixtures

Before wiring into an API gateway, we need automated tests verifying that the 
parser can read PDF bytes, extract tables, and correctly aggregate the new 
multimodal compliance and fraud metrics.
"""

from pathlib import Path
import pytest
from pypdf import PdfWriter
from src.vision.parser import FinancialDocumentParser

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Generates a minimal valid PDF file dynamically for unit testing."""
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

def test_parser_aggregate_metrics_and_variance():
    """Tests that the deterministic math correctly calculates variance and credit anomalies."""
    parser = FinancialDocumentParser()
    
    # Swapped from parse_metrics to aggregate_metrics to match the new architecture
    metrics = parser.aggregate_metrics(
        stated_monthly_income=100000.0,
        verified_income=80000.0,  # 20% drop (triggers < 85% rule)
        detected_liabilities=25000.0,
        verified_reserves=150000.0,
        verified_collateral_value=None,
        pep_or_sanctions_hit=False,
        ubo_verified=True,
        kyc_confidence_score=0.98,
        document_tampering_score=0.01,
        synthetic_voice_probability=0.05
    )
    
    assert metrics.verified_monthly_income == 80000.0
    assert metrics.income_variance_pct == -20.0
    # Verifies the income mismatch flag was dynamically generated
    assert any("Income mismatch" in flag for flag in metrics.anomaly_flags)
    # Verifies the nested MultimodalFraudMetrics class instantiated correctly
    assert metrics.fraud_metrics.kyc_confidence_score == 0.98

def test_parser_aggregate_metrics_fraud_flags():
    """Tests that multimodal fraud and AML alerts are correctly mapped into anomaly flags."""
    parser = FinancialDocumentParser()
    metrics = parser.aggregate_metrics(
        stated_monthly_income=100000.0,
        verified_income=100000.0,  # Perfect income match
        detected_liabilities=25000.0,
        verified_reserves=150000.0,
        pep_or_sanctions_hit=True,       # FATAL: AML Hit
        ubo_verified=False,
        kyc_confidence_score=0.50,
        document_tampering_score=0.85    # FATAL: High probability of Photoshop
    )
    
    assert metrics.income_variance_pct == 0.0
    assert metrics.pep_or_sanctions_hit is True
    
    # Verify the parser injected the exact fraud anomaly alert strings
    assert any("document tampering" in flag.lower() for flag in metrics.anomaly_flags)
    assert any("ofac/sanctions" in flag.lower() for flag in metrics.anomaly_flags)