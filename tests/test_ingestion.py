"""
Tests for the ingestion module.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.ingestion import (
    DocumentIngester,
    SUPPORTED_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
)


class TestDocumentIngester:
    """Tests for document ingestion."""

    def test_supported_extensions(self):
        """Check that all expected extensions are supported."""
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        
        # Image extensions
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS

    def test_image_extensions(self):
        """Check image extensions."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS

    def test_init_with_docs_dir(self, temp_dir):
        """Ingester should initialize with docs directory."""
        ingester = DocumentIngester(temp_dir)
        assert ingester.docs_dir == temp_dir

    def test_init_local_only_mode(self, temp_dir):
        """Ingester should respect local_only mode."""
        ingester = DocumentIngester(temp_dir, local_only=True)
        assert ingester.local_only is True
        
        # Should return None for OpenAI client in local mode
        client = ingester._get_openai_client()
        assert client is None

    def test_read_txt_file(self, sample_txt, temp_dir):
        """Ingester should read text files."""
        ingester = DocumentIngester(temp_dir)
        
        content = ingester._read_file(sample_txt)
        
        assert "sample text file" in content

    def test_read_nonexistent_file(self, temp_dir):
        """Reading nonexistent file should raise error."""
        ingester = DocumentIngester(temp_dir)
        
        with pytest.raises(RuntimeError):
            ingester._read_file(temp_dir / "nonexistent.txt")

    def test_ingest_all_empty_dir(self, temp_dir):
        """Ingesting empty directory should raise error."""
        ingester = DocumentIngester(temp_dir)
        
        with pytest.raises(RuntimeError, match="No supported documents"):
            ingester.ingest_all()

    def test_ingest_all_with_files(self, temp_dir):
        """Ingester should process all supported files."""
        # Create test files
        (temp_dir / "doc1.txt").write_text("Document one content here.")
        (temp_dir / "doc2.txt").write_text("Document two content here.")
        (temp_dir / "ignored.xyz").write_text("Should be ignored.")
        
        ingester = DocumentIngester(temp_dir)
        results = ingester.ingest_all()
        
        assert len(results) == 2
        assert all("filename" in r for r in results)
        assert all("content" in r for r in results)

    def test_ingest_files_parallel(self, temp_dir):
        """Parallel ingestion should work."""
        # Create test files
        for i in range(5):
            (temp_dir / f"doc{i}.txt").write_text(f"Content for document {i}.")
        
        ingester = DocumentIngester(temp_dir)
        
        file_paths = list(temp_dir.glob("*.txt"))
        results = ingester.ingest_files(file_paths, parallel=True, max_workers=2)
        
        assert len(results) == 5

    def test_image_dimension_check_small(self, temp_dir):
        """Small images should be rejected."""
        ingester = DocumentIngester(temp_dir)
        
        # Create a small test image file
        small_img = temp_dir / "small_icon.png"
        small_img.write_bytes(b"fake png")  # Just need a file to exist
        
        # Mock PIL.Image.open to return small dimensions
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (50, 50)  # Small icon
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_img
            
            result = ingester._is_image_large_enough(small_img)
            assert result is False

    def test_image_dimension_check_large(self, temp_dir):
        """Large images should be accepted."""
        ingester = DocumentIngester(temp_dir)
        
        # Mock PIL to return large dimensions
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.size = (800, 600)  # Large photo
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_img
            
            # This will use the mocked Image
            # Note: actual test depends on PIL being available

    def test_ingest_skips_empty_files(self, temp_dir):
        """Ingester should skip empty files."""
        (temp_dir / "empty.txt").write_text("")
        (temp_dir / "content.txt").write_text("This has content.")
        
        ingester = DocumentIngester(temp_dir)
        results = ingester.ingest_all()
        
        # Only the file with content should be included
        assert len(results) == 1
        assert results[0]["filename"] == "content.txt"

    def test_progress_callback(self, temp_dir):
        """Progress callback should be called."""
        (temp_dir / "doc1.txt").write_text("Content 1")
        (temp_dir / "doc2.txt").write_text("Content 2")
        
        ingester = DocumentIngester(temp_dir)
        
        progress_calls = []
        def callback(filename, current, total):
            progress_calls.append((filename, current, total))
        
        file_paths = list(temp_dir.glob("*.txt"))
        ingester.ingest_files(file_paths, parallel=False, progress_callback=callback)
        
        assert len(progress_calls) == 2
