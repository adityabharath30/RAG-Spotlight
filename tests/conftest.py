"""
Pytest configuration and shared fixtures.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    John Smith works at Acme Corporation as a Software Engineer.
    His salary is $150,000 per year. He started on January 15, 2024.
    The company is located in San Francisco, California.
    His employee ID is EMP-12345.
    """


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "text": "John Smith works at Acme Corporation as a Software Engineer. His salary is $150,000 per year.",
            "filename": "employee.txt",
            "filepath": "/docs/employee.txt",
        },
        {
            "text": "He started on January 15, 2024. The company is located in San Francisco.",
            "filename": "employee.txt",
            "filepath": "/docs/employee.txt",
        },
        {
            "text": "The quarterly report shows revenue of $5 million for Q3 2024.",
            "filename": "report.pdf",
            "filepath": "/docs/report.pdf",
        },
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "answer": "$150,000 per year",
        "confidence": 0.95,
    }


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client for testing."""
    with patch("app.llm.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            f'{{"answer": "{mock_openai_response["answer"]}", '
            f'"confidence": {mock_openai_response["confidence"]}}}'
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_embeddings():
    """Mock embedding generator."""
    with patch("app.embeddings.get_cached_model") as mock:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype("float32")
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "sample.pdf"
    # Create a minimal PDF
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer << /Size 5 /Root 1 0 R >>
startxref
307
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_txt(temp_dir):
    """Create a sample text file for testing."""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text("This is a sample text file for testing purposes.")
    return txt_path


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Set up test environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-12345"}):
        yield
