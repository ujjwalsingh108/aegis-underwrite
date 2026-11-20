"""
src/vision/parser.py

In financial underwriting, language models should never be fed raw, unparsed
PDF text because financial tables, loan balances, bank statements, and income slips
get distorted.

This module is the deterministic PDF extraction gateway (text + bounding-box tables).
Metric aggregation lives in ``src/engine/verification.py``.
"""

import pdfplumber
from pypdf import PdfReader


class FinancialDocumentParser:
    """Extracts raw text and structured tables from financial PDFs."""

    def extract_text(self, file_path: str) -> str:
        """Extracts standard sequential text across all pages."""
        reader = PdfReader(file_path)
        extracted_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text)

    def extract_tables(self, file_path: str) -> list[list[list[str | None]]]:
        """Extracts structured grid tables across all pages using bounding boxes."""
        all_tables: list[list[list[str | None]]] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        return all_tables
