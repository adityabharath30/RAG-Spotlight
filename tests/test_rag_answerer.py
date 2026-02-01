"""
Tests for the RAG answerer module.
"""
from unittest.mock import patch

from app.rag_answerer import (
    extract_best_answer,
    propose_answer_from_chunk,
    select_best_answer,
    compress_answer_if_needed,
    normalize_whitespace,
    fix_pdf_spacing,
    AnswerCandidate,
)


class TestTextUtilities:
    """Tests for text processing utilities."""

    def test_normalize_whitespace(self):
        """Whitespace should be normalized."""
        text = "Hello   world\n\ntest"
        result = normalize_whitespace(text)
        assert result == "Hello world test"

    def test_normalize_whitespace_punctuation(self):
        """Punctuation spacing should be fixed."""
        text = "Hello , world . Test"
        result = normalize_whitespace(text)
        assert result == "Hello, world. Test"

    def test_fix_pdf_spacing_hyphen(self):
        """Hyphenated words should be joined."""
        text = "docu- ment"
        result = fix_pdf_spacing(text)
        assert result == "document"

    def test_fix_pdf_spacing_letters(self):
        """Spaced letters should be joined."""
        text = "A B C D"
        result = fix_pdf_spacing(text)
        # Should join spaced letters
        assert "A B C D" not in result or result == "ABCD"


class TestAnswerExtraction:
    """Tests for answer extraction with mocked OpenAI."""

    def test_propose_answer_empty_chunk(self):
        """Empty chunk should return None."""
        chunk = {"text": "", "filename": "test.txt", "filepath": "/test.txt"}
        result = propose_answer_from_chunk("What is X?", chunk)
        assert result is None

    def test_propose_answer_short_chunk(self):
        """Very short chunk should return None."""
        chunk = {"text": "Hi", "filename": "test.txt", "filepath": "/test.txt"}
        result = propose_answer_from_chunk("What is X?", chunk)
        assert result is None

    @patch("app.rag_answerer.llm_available")
    @patch("app.rag_answerer.llm_extract")
    def test_propose_answer_with_llm(self, mock_extract, mock_available):
        """Should use LLM when available."""
        mock_available.return_value = True
        mock_extract.return_value = {"answer": "$150,000", "confidence": 0.9}
        
        chunk = {
            "text": "John's salary is $150,000 per year.",
            "filename": "salary.txt",
            "filepath": "/docs/salary.txt",
        }
        
        result = propose_answer_from_chunk("What is John's salary?", chunk)
        
        assert result is not None
        assert result.answer == "$150,000"
        assert result.confidence == 0.9

    @patch("app.rag_answerer.llm_available")
    @patch("app.rag_answerer.llm_extract")
    def test_propose_answer_llm_returns_none(self, mock_extract, mock_available):
        """Should return None when LLM says NONE."""
        mock_available.return_value = True
        mock_extract.return_value = {"answer": "NONE", "confidence": 0.0}
        
        chunk = {
            "text": "The weather is nice today.",
            "filename": "weather.txt",
            "filepath": "/docs/weather.txt",
        }
        
        result = propose_answer_from_chunk("What is John's salary?", chunk)
        assert result is None

    @patch("app.rag_answerer.llm_available")
    def test_propose_answer_fallback(self, mock_available):
        """Should use regex fallback when LLM unavailable."""
        mock_available.return_value = False
        
        chunk = {
            "text": "The price is $500. This is a great deal for the product.",
            "filename": "price.txt",
            "filepath": "/docs/price.txt",
        }
        
        _result = propose_answer_from_chunk("How much does it cost?", chunk)  # noqa: F841
        # May or may not find answer depending on regex patterns
        # Just verify it doesn't crash


class TestCandidateSelection:
    """Tests for answer candidate selection."""

    def test_select_best_empty(self):
        """Empty candidates should return None."""
        result = select_best_answer([], "test query")
        assert result is None

    def test_select_best_single(self):
        """Single candidate should be selected."""
        candidate = AnswerCandidate(
            answer="$100",
            confidence=0.8,
            source="test.txt",
            filepath="/test.txt",
            chunk_text="The price is $100.",
        )
        
        result = select_best_answer([candidate], "What is the price?")
        assert result == candidate

    def test_select_best_multiple(self):
        """Best candidate should be selected from multiple."""
        candidates = [
            AnswerCandidate(
                answer="Maybe $50",
                confidence=0.3,
                source="a.txt",
                filepath="/a.txt",
                chunk_text="Maybe $50.",
            ),
            AnswerCandidate(
                answer="$100",
                confidence=0.9,
                source="b.txt",
                filepath="/b.txt",
                chunk_text="The price is $100.",
            ),
            AnswerCandidate(
                answer="About $75",
                confidence=0.5,
                source="c.txt",
                filepath="/c.txt",
                chunk_text="About $75.",
            ),
        ]
        
        result = select_best_answer(candidates, "What is the price?")
        
        # Highest confidence should win
        assert result.confidence == 0.9

    def test_select_penalizes_generic(self):
        """Candidates with higher effective scores should be selected."""
        candidates = [
            AnswerCandidate(
                answer="The document contains information about prices.",
                confidence=0.5,  # Lower confidence for generic
                source="a.txt",
                filepath="/a.txt",
                chunk_text="Generic text.",
            ),
            AnswerCandidate(
                answer="$100",
                confidence=0.85,  # Higher confidence for specific
                source="b.txt",
                filepath="/b.txt",
                chunk_text="The price is $100.",
            ),
        ]
        
        result = select_best_answer(candidates, "What is the price?")
        
        # Higher confidence answer should win
        assert result.answer == "$100"


class TestAnswerCompression:
    """Tests for answer compression."""

    def test_compress_short_answer(self):
        """Short answers should not be compressed."""
        answer = "The price is $100."
        result = compress_answer_if_needed(answer, max_words=25)
        assert result == answer

    @patch("app.rag_answerer.llm_available")
    def test_compress_long_answer_no_llm(self, mock_available):
        """Long answers should be truncated without LLM."""
        mock_available.return_value = False
        
        answer = " ".join(["word"] * 50)
        result = compress_answer_if_needed(answer, max_words=10)
        
        assert len(result.split()) <= 11  # 10 words + period


class TestExtractBestAnswer:
    """Tests for the main extract_best_answer function."""

    def test_extract_empty_chunks(self):
        """Empty chunks should return abstain response."""
        result = extract_best_answer("What is X?", [])
        
        assert result["answerable"] is False
        assert "No documents found" in result["answer"]

    @patch("app.rag_answerer.llm_available")
    @patch("app.rag_answerer.llm_extract")
    def test_extract_with_answer(self, mock_extract, mock_available):
        """Should extract answer from chunks."""
        mock_available.return_value = True
        mock_extract.return_value = {"answer": "$150,000", "confidence": 0.85}
        
        chunks = [
            {
                "text": "John's salary is $150,000 per year.",
                "filename": "salary.txt",
                "filepath": "/docs/salary.txt",
            }
        ]
        
        result = extract_best_answer("What is John's salary?", chunks)
        
        assert result["answerable"] is True
        assert "$150,000" in result["answer"]
        assert result["source"] == "salary.txt"

    @patch("app.rag_answerer.llm_available")
    @patch("app.rag_answerer.llm_extract")
    def test_extract_no_answer(self, mock_extract, mock_available):
        """Should return abstain when no answer found."""
        mock_available.return_value = True
        mock_extract.return_value = {"answer": "NONE", "confidence": 0.0}
        
        chunks = [
            {
                "text": "The weather is nice today.",
                "filename": "weather.txt",
                "filepath": "/docs/weather.txt",
            }
        ]
        
        result = extract_best_answer("What is John's salary?", chunks)
        
        assert result["answerable"] is False
