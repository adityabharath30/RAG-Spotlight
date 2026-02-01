"""
Tests for the privacy module.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.privacy import PrivacyManager


class TestPrivacyManager:
    """Tests for privacy controls."""

    @pytest.fixture
    def privacy_manager(self, temp_dir):
        """Create a privacy manager with temp directory."""
        with patch("app.privacy.DATA_DIR", temp_dir):
            with patch("app.privacy.INDEX_PATH", temp_dir / "index"):
                manager = PrivacyManager(temp_dir)
                yield manager

    def test_list_indexed_files_empty(self, privacy_manager, temp_dir):
        """Empty manifest should return empty list."""
        files = privacy_manager.list_indexed_files()
        assert files == []

    def test_list_indexed_files_with_data(self, privacy_manager, temp_dir):
        """Should list files from manifest."""
        # Create a manifest
        manifest_data = {
            "files": {
                "/path/to/doc1.pdf": {
                    "indexed_at": "2024-01-15T10:00:00",
                    "chunk_count": 5,
                    "size": 1024,
                },
                "/path/to/doc2.txt": {
                    "indexed_at": "2024-01-16T11:00:00",
                    "chunk_count": 3,
                    "size": 512,
                },
            },
            "last_full_scan": "2024-01-16T12:00:00",
        }
        
        manifest_path = temp_dir / "scan_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        
        files = privacy_manager.list_indexed_files()
        
        assert len(files) == 2
        assert all("filepath" in f for f in files)
        assert all("filename" in f for f in files)

    def test_get_storage_stats_empty(self, privacy_manager):
        """Empty storage should return zero stats."""
        stats = privacy_manager.get_storage_stats()
        
        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0

    def test_get_storage_stats_with_data(self, privacy_manager, temp_dir):
        """Should calculate storage stats."""
        # Create manifest
        manifest_data = {
            "files": {
                "/doc1.pdf": {"chunk_count": 10, "size": 1024},
                "/doc2.pdf": {"chunk_count": 5, "size": 2048},
            }
        }
        (temp_dir / "scan_manifest.json").write_text(json.dumps(manifest_data))
        
        stats = privacy_manager.get_storage_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_chunks"] == 15

    def test_export_manifest(self, privacy_manager, temp_dir):
        """Should export manifest to file."""
        # Create manifest
        manifest_data = {
            "files": {"/doc.pdf": {"indexed_at": "2024-01-01", "chunk_count": 5}},
            "last_full_scan": "2024-01-01",
        }
        (temp_dir / "scan_manifest.json").write_text(json.dumps(manifest_data))
        
        output_path = temp_dir / "export.json"
        result = privacy_manager.export_manifest(output_path)
        
        assert result is True
        assert output_path.exists()
        
        exported = json.loads(output_path.read_text())
        assert "files" in exported
        assert "export_date" in exported

    def test_export_manifest_no_data(self, privacy_manager, temp_dir):
        """Export should fail gracefully with no data."""
        output_path = temp_dir / "export.json"
        result = privacy_manager.export_manifest(output_path)
        
        assert result is False

    def test_delete_file_from_index(self, privacy_manager, temp_dir):
        """Should remove file from manifest."""
        manifest_data = {
            "files": {
                "/doc1.pdf": {"chunk_count": 5},
                "/doc2.pdf": {"chunk_count": 3},
            }
        }
        manifest_path = temp_dir / "scan_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        
        result = privacy_manager.delete_file_from_index("/doc1.pdf")
        
        assert result is True
        
        # Verify file was removed
        updated = json.loads(manifest_path.read_text())
        assert "/doc1.pdf" not in updated["files"]
        assert "/doc2.pdf" in updated["files"]

    def test_delete_nonexistent_file(self, privacy_manager, temp_dir):
        """Deleting nonexistent file should return False."""
        manifest_data = {"files": {"/doc.pdf": {"chunk_count": 5}}}
        (temp_dir / "scan_manifest.json").write_text(json.dumps(manifest_data))
        
        result = privacy_manager.delete_file_from_index("/nonexistent.pdf")
        assert result is False

    def test_delete_index(self, privacy_manager, temp_dir):
        """Should delete index files."""
        # Create fake index files
        (temp_dir / "index.faiss").write_text("fake faiss")
        (temp_dir / "index.pkl").write_text("fake pickle")
        
        with patch("app.privacy.INDEX_PATH", temp_dir / "index"):
            result = privacy_manager.delete_index()
        
        assert result is True

    def test_delete_all_data_requires_confirm(self, privacy_manager):
        """delete_all_data should require confirmation."""
        result = privacy_manager.delete_all_data(confirm=False)
        assert result is False

    def test_delete_all_data_with_confirm(self, privacy_manager, temp_dir):
        """delete_all_data should delete everything."""
        # Create various data files
        (temp_dir / "scan_manifest.json").write_text("{}")
        (temp_dir / "audit.log").write_text("log data")
        
        result = privacy_manager.delete_all_data(confirm=True)
        
        assert result is True
        assert not (temp_dir / "scan_manifest.json").exists()
        assert not (temp_dir / "audit.log").exists()

    def test_generate_privacy_report(self, privacy_manager, temp_dir):
        """Should generate privacy report."""
        manifest_data = {
            "files": {"/doc.pdf": {"chunk_count": 5, "size": 1024}},
        }
        (temp_dir / "scan_manifest.json").write_text(json.dumps(manifest_data))
        
        report = privacy_manager.generate_privacy_report()
        
        assert "summary" in report
        assert "data_locations" in report
        assert "how_to_delete" in report

    def test_get_indexed_file_count(self, privacy_manager, temp_dir):
        """Should return correct file count."""
        manifest_data = {
            "files": {
                "/doc1.pdf": {},
                "/doc2.pdf": {},
                "/doc3.pdf": {},
            }
        }
        (temp_dir / "scan_manifest.json").write_text(json.dumps(manifest_data))
        
        count = privacy_manager.get_indexed_file_count()
        assert count == 3
