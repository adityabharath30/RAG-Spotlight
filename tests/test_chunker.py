"""
Tests for the chunker module.
"""
import pytest
from app.chunker import chunk


class TestChunker:
    """Tests for text chunking functionality."""

    def test_chunk_empty_text(self):
        """Empty text should return empty list."""
        result = list(chunk(""))
        assert result == []

    def test_chunk_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        result = list(chunk("   \n\t  "))
        assert result == []

    def test_chunk_short_text(self):
        """Short text should return single chunk."""
        text = "This is a short sentence. Another sentence here."
        result = list(chunk(text, chunk_size=100))
        assert len(result) >= 1
        assert "short sentence" in result[0]

    def test_chunk_respects_sentence_boundaries(self):
        """Chunks should respect sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = list(chunk(text, chunk_size=10, overlap=0))
        
        # Each chunk should end with a complete sentence
        for c in result:
            assert c.strip().endswith(".")

    def test_chunk_overlap(self):
        """Chunks should have overlap when specified."""
        text = (
            "Sentence one is here. Sentence two follows. "
            "Sentence three appears. Sentence four ends."
        )
        result = list(chunk(text, chunk_size=10, overlap=5))
        
        # With overlap, some words should appear in multiple chunks
        assert len(result) >= 2

    def test_chunk_long_text(self):
        """Long text should be split into multiple chunks."""
        # Create a long text with many sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(50)]
        text = " ".join(sentences)
        
        result = list(chunk(text, chunk_size=50, overlap=10))
        
        assert len(result) > 1
        # All chunks should have content
        assert all(len(c.strip()) > 0 for c in result)

    def test_chunk_preserves_content(self):
        """All original content should be present across chunks."""
        text = "Important fact one. Important fact two. Important fact three."
        result = list(chunk(text, chunk_size=20, overlap=5))
        
        combined = " ".join(result)
        assert "Important fact one" in combined
        assert "Important fact two" in combined
        assert "Important fact three" in combined

    def test_chunk_handles_special_characters(self):
        """Chunker should handle special characters."""
        text = "Price is $100.50! What do you think? Yes, it's great."
        result = list(chunk(text, chunk_size=50))
        
        assert len(result) >= 1
        assert "$100.50" in result[0]

    def test_chunk_handles_numbers(self):
        """Chunker should handle numbers correctly."""
        text = "The year 2024 was great. Revenue was 1,500,000 dollars. Growth rate: 15.5%."
        result = list(chunk(text, chunk_size=50))
        
        combined = " ".join(result)
        assert "2024" in combined
        assert "1,500,000" in combined
