"""
User-friendly error handling for the RAG system.

Provides:
- Custom exception classes with helpful messages
- Error formatting for UI display
- Recovery suggestions
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RAGError(Exception):
    """Base exception for RAG system errors."""
    
    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        suggestion: str | None = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.suggestion = suggestion
        self.severity = severity
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "error": self.message,
            "user_message": self.user_message,
            "suggestion": self.suggestion,
            "severity": self.severity.value,
            "details": self.details,
        }
    
    def format_for_ui(self) -> str:
        """Format error for display in UI."""
        parts = [f"❌ {self.user_message}"]
        if self.suggestion:
            parts.append(f"💡 {self.suggestion}")
        return "\n".join(parts)


class ConfigurationError(RAGError):
    """Error in configuration."""
    pass


class IndexError(RAGError):
    """Error with the search index."""
    pass


class IngestionError(RAGError):
    """Error during document ingestion."""
    pass


class SearchError(RAGError):
    """Error during search."""
    pass


class APIError(RAGError):
    """Error with external API (OpenAI, etc.)."""
    pass


# ============================================================================
# Common Error Factories
# ============================================================================

def index_not_found() -> IndexError:
    """Create error for missing index."""
    return IndexError(
        message="FAISS index not found",
        user_message="Search index not found",
        suggestion="Run 'python scripts/watcher.py --scan-now' to build the index",
        severity=ErrorSeverity.ERROR,
    )


def no_documents_found(directory: str) -> IngestionError:
    """Create error for empty documents directory."""
    return IngestionError(
        message=f"No documents found in {directory}",
        user_message="No documents to index",
        suggestion=f"Add PDF, DOCX, or TXT files to {directory}",
        severity=ErrorSeverity.WARNING,
        details={"directory": directory},
    )


def api_key_missing(key_name: str = "OPENAI_API_KEY") -> ConfigurationError:
    """Create error for missing API key."""
    return ConfigurationError(
        message=f"{key_name} not found",
        user_message="OpenAI API key not configured",
        suggestion="Add your API key to .env file or use: python -c \"from app.security import ...; km.set_api_key('OPENAI_API_KEY', 'sk-...')\"",
        severity=ErrorSeverity.ERROR,
        details={"key_name": key_name},
    )


def api_rate_limit() -> APIError:
    """Create error for API rate limiting."""
    return APIError(
        message="OpenAI API rate limit exceeded",
        user_message="Too many requests to OpenAI",
        suggestion="Wait a moment and try again, or reduce batch size in scanner_config.yaml",
        severity=ErrorSeverity.WARNING,
    )


def api_quota_exceeded() -> APIError:
    """Create error for API quota exceeded."""
    return APIError(
        message="OpenAI API quota exceeded",
        user_message="OpenAI API quota exceeded",
        suggestion="Check your OpenAI billing at platform.openai.com/usage",
        severity=ErrorSeverity.ERROR,
    )


def file_read_error(filepath: str, reason: str = "") -> IngestionError:
    """Create error for file read failure."""
    return IngestionError(
        message=f"Failed to read {filepath}: {reason}",
        user_message=f"Could not read file: {filepath.split('/')[-1]}",
        suggestion="Check if the file is corrupted or password-protected",
        severity=ErrorSeverity.WARNING,
        details={"filepath": filepath, "reason": reason},
    )


def unsupported_file_type(filepath: str, extension: str) -> IngestionError:
    """Create error for unsupported file type."""
    return IngestionError(
        message=f"Unsupported file type: {extension}",
        user_message=f"Cannot process .{extension} files",
        suggestion="Supported types: PDF, DOCX, TXT, MD, CSV, XLSX, JPG, PNG",
        severity=ErrorSeverity.INFO,
        details={"filepath": filepath, "extension": extension},
    )


def scan_directory_not_found(directory: str) -> ConfigurationError:
    """Create error for missing scan directory."""
    return ConfigurationError(
        message=f"Scan directory not found: {directory}",
        user_message=f"Directory not found: {directory}",
        suggestion="Check scanner_config.yaml and verify the directory exists",
        severity=ErrorSeverity.WARNING,
        details={"directory": directory},
    )


def encryption_error(operation: str) -> RAGError:
    """Create error for encryption/decryption failure."""
    return RAGError(
        message=f"Encryption {operation} failed",
        user_message=f"Security error during {operation}",
        suggestion="Try deleting data/.salt and rebuilding the index",
        severity=ErrorSeverity.ERROR,
    )


# ============================================================================
# Error Formatting
# ============================================================================

def format_exception_for_user(exc: Exception) -> str:
    """
    Format any exception into a user-friendly message.
    
    Handles both RAGError and standard exceptions.
    """
    if isinstance(exc, RAGError):
        return exc.format_for_ui()
    
    # Handle common Python exceptions
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    
    # Map common exceptions to friendly messages
    friendly_messages = {
        "FileNotFoundError": ("File not found", "Check if the file exists"),
        "PermissionError": ("Permission denied", "Check file permissions"),
        "ConnectionError": ("Connection failed", "Check your internet connection"),
        "TimeoutError": ("Request timed out", "Try again in a moment"),
        "JSONDecodeError": ("Invalid data format", "The file may be corrupted"),
        "KeyError": ("Missing configuration", "Check your settings"),
        "ValueError": ("Invalid value", "Check your input"),
    }
    
    if exc_type in friendly_messages:
        user_msg, suggestion = friendly_messages[exc_type]
        return f"❌ {user_msg}\n💡 {suggestion}"
    
    # Default formatting
    return f"❌ {exc_type}: {exc_msg}"


def log_error_with_context(
    error: Exception,
    context: str,
    logger=None
) -> None:
    """
    Log an error with additional context.
    
    Args:
        error: The exception that occurred
        context: Description of what was happening
        logger: Optional logger instance
    """
    import logging
    log = logger or logging.getLogger("rag")
    
    if isinstance(error, RAGError):
        log.error(
            "%s: %s (severity=%s)",
            context,
            error.message,
            error.severity.value,
            extra=error.details,
        )
    else:
        log.error("%s: %s", context, str(error), exc_info=True)
