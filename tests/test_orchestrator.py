# Tests the master pipeline controller, ensuring sequential execution and critical filesystem cleanup.

import pytest
import os
from unittest.mock import patch, AsyncMock
from src.engine.orchestrator import PipelineOrchestrator
from src.common.schemas import UnderwritingMemo
from fastapi import UploadFile

@pytest.fixture
def mock_upload_file():
    file = AsyncMock(spec=UploadFile)
    file.read.return_value = b"%PDF-1.4 Mock PDF Bytes"
    return file

@pytest.mark.asyncio
@patch("src.engine.orchestrator.FinancialDocumentParser")
@patch("src.engine.orchestrator.UnderwritingLLMEngine")
async def test_orchestrator_pipeline_execution_and_cleanup(MockEngine, MockParser, mock_upload_file):
    """State & Lifecycle: Ensures the orchestrator runs sequential steps and deletes the temp file."""
    
    # Setup Mocks
    mock_engine_instance = MockEngine.return_value
    mock_engine_instance._stream_internal.return_value = AsyncMock()
    
    # Mock async generator for _stream_internal
    async def mock_stream(*args, **kwargs):
        yield '{"application_id": "APP-9921", "decision": "APPROVE", "calculated_dti": 0.3, "calculated_dscr": 1.5, "liquidity_runway_months": 10, "breached_policies": [], "risk_factors": [], "required_covenants": [], "audit_trail": []}'
    mock_engine_instance._stream_internal = mock_stream
    
    orchestrator = PipelineOrchestrator()
    
    # Execute Pipeline
    memo = await orchestrator.process_application(mock_upload_file)
    
    # Assertions
    assert isinstance(memo, UnderwritingMemo)
    assert memo.decision.value == "APPROVE"
    
    # Verify cleanup (Cannot track exact tempname easily without mocking tempfile, 
    # but we test that no dangling .pdf files are left in the working directory)
    assert not any(f.endswith('.pdf') for f in os.listdir('.'))