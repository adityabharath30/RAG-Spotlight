"""
Tests for the scanner configuration module.
"""
from pathlib import Path
from unittest.mock import patch

from app.scanner_config import (
    ScannerConfig,
    load_config,
    _expand_path,
)


class TestScannerConfig:
    """Tests for ScannerConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ScannerConfig()
        
        assert config.recursive is True
        assert config.follow_symlinks is False
        assert config.max_depth == 10
        assert config.process_images is False  # Disabled by default
        assert config.min_image_width == 200
        assert config.min_image_height == 200

    def test_default_excluded_dirs(self):
        """Should have default excluded directories."""
        config = ScannerConfig()
        
        excluded = config.excluded_directories
        assert any(".ssh" in d for d in excluded)
        assert any(".git" in d for d in excluded)
        assert any("node_modules" in d for d in excluded)

    def test_default_excluded_patterns(self):
        """Should have default excluded patterns."""
        config = ScannerConfig()
        
        patterns = config.excluded_file_patterns
        assert any(".pem" in p for p in patterns)
        assert any(".env" in p for p in patterns)
        assert any("credentials" in p for p in patterns)

    def test_is_directory_excluded_ssh(self):
        """SSH directory should be excluded."""
        config = ScannerConfig()
        
        assert config.is_directory_excluded(Path("/home/user/.ssh"))
        assert config.is_directory_excluded(Path("/Users/me/.ssh/keys"))

    def test_is_directory_excluded_git(self):
        """Git directory should be excluded."""
        config = ScannerConfig()
        
        assert config.is_directory_excluded(Path("/project/.git"))
        assert config.is_directory_excluded(Path("/project/.git/objects"))

    def test_is_directory_excluded_allowed(self):
        """Regular directories should not be excluded."""
        config = ScannerConfig()
        
        assert not config.is_directory_excluded(Path("/home/user/Documents"))
        assert not config.is_directory_excluded(Path("/Users/me/Desktop"))

    def test_is_file_excluded_env(self):
        """Env files should be excluded."""
        config = ScannerConfig()
        
        assert config.is_file_excluded(Path("/project/.env"))
        assert config.is_file_excluded(Path("/project/.env.local"))
        assert config.is_file_excluded(Path("/project/production.env"))

    def test_is_file_excluded_credentials(self):
        """Credential files should be excluded."""
        config = ScannerConfig()
        
        assert config.is_file_excluded(Path("/project/credentials.json"))
        assert config.is_file_excluded(Path("/project/aws_credentials"))

    def test_is_file_excluded_keys(self):
        """Key files should be excluded."""
        config = ScannerConfig()
        
        assert config.is_file_excluded(Path("/home/.ssh/id_rsa"))
        assert config.is_file_excluded(Path("/certs/server.pem"))
        assert config.is_file_excluded(Path("/certs/private.key"))

    def test_is_file_excluded_icons(self):
        """Icon files should be excluded."""
        config = ScannerConfig()
        
        assert config.is_file_excluded(Path("/app/icon.png"))
        assert config.is_file_excluded(Path("/app/toolbarButton-save.png"))
        assert config.is_file_excluded(Path("/app/logo.png"))

    def test_is_file_excluded_allowed(self):
        """Regular files should not be excluded."""
        config = ScannerConfig()
        
        assert not config.is_file_excluded(Path("/docs/report.pdf"))
        assert not config.is_file_excluded(Path("/docs/notes.txt"))
        assert not config.is_file_excluded(Path("/photos/vacation.jpg"))

    def test_is_file_size_valid(self, temp_dir):
        """File size validation should work."""
        config = ScannerConfig(
            min_file_size_bytes=100,
            max_file_size_mb=1.0,
        )
        
        # Create test files
        small_file = temp_dir / "small.txt"
        small_file.write_text("x" * 50)  # 50 bytes - too small
        
        valid_file = temp_dir / "valid.txt"
        valid_file.write_text("x" * 500)  # 500 bytes - valid
        
        assert not config.is_file_size_valid(small_file)
        assert config.is_file_size_valid(valid_file)

    def test_should_process_image_disabled(self):
        """Images should not be processed when disabled."""
        config = ScannerConfig(process_images=False)
        
        assert not config.should_process_image(Path("/photos/test.jpg"))

    def test_should_process_image_local_only(self):
        """Images should not be processed in local-only mode."""
        config = ScannerConfig(process_images=True, local_only_mode=True)
        
        assert not config.should_process_image(Path("/photos/test.jpg"))

    def test_get_scan_directories_filters_nonexistent(self, temp_dir):
        """Should filter out nonexistent directories."""
        existing_dir = temp_dir / "exists"
        existing_dir.mkdir()
        
        config = ScannerConfig(
            scan_directories=[
                existing_dir,
                Path("/nonexistent/directory"),
            ]
        )
        
        valid_dirs = config.get_scan_directories()
        
        assert len(valid_dirs) == 1
        assert valid_dirs[0] == existing_dir


class TestExpandPath:
    """Tests for path expansion."""

    def test_expand_home(self):
        """Should expand ~ to home directory."""
        result = _expand_path("~/Documents")
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_expand_env_var(self):
        """Should expand environment variables."""
        with patch.dict("os.environ", {"TEST_DIR": "/test/path"}):
            result = _expand_path("$TEST_DIR/subdir")
            assert "/test/path" in str(result)

    def test_expand_absolute(self):
        """Absolute paths should remain absolute."""
        result = _expand_path("/absolute/path")
        assert str(result) == "/absolute/path"


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_missing_config(self, temp_dir):
        """Missing config should use defaults."""
        config = load_config(temp_dir / "nonexistent.yaml")
        
        # Should have default values
        assert config.recursive is True
        assert config.process_images is False

    def test_load_valid_config(self, temp_dir):
        """Should load config from YAML file."""
        config_content = """
scan_directories:
  - ~/Documents
  - ~/Desktop

process_images: true
min_image_width: 300
parallel_workers: 8
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)
        
        config = load_config(config_path)
        
        assert config.process_images is True
        assert config.min_image_width == 300
        assert config.parallel_workers == 8
        assert len(config.scan_directories) == 2

    def test_load_invalid_yaml(self, temp_dir):
        """Invalid YAML should use defaults."""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("{ invalid yaml [")
        
        config = load_config(config_path)
        
        # Should fall back to defaults
        assert config.recursive is True
