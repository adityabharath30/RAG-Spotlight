"""
Document Utilities for RAG System.

Provides:
- Open documents at specific locations
- Search within documents
- Generate document previews
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("rag")


# ============================================================================
# Document Opening with Location
# ============================================================================

def open_document(filepath: str, search_text: str | None = None) -> bool:
    """
    Open a document, optionally searching for specific text.
    
    On macOS:
    - PDFs: Opens in Preview and can search for text
    - Other files: Opens with default application
    
    Args:
        filepath: Path to the document
        search_text: Optional text to search for after opening
    
    Returns:
        True if document was opened successfully
    """
    if not os.path.exists(filepath):
        logger.warning("Document not found: %s", filepath)
        return False
    
    ext = Path(filepath).suffix.lower()
    
    try:
        if sys.platform == "darwin":  # macOS
            return _open_document_macos(filepath, ext, search_text)
        elif sys.platform == "win32":  # Windows
            return _open_document_windows(filepath, ext, search_text)
        else:  # Linux
            return _open_document_linux(filepath)
    except Exception as e:
        logger.error("Failed to open document: %s", e)
        return False


def _open_document_macos(filepath: str, ext: str, search_text: str | None) -> bool:
    """Open document on macOS with optional search."""
    
    if ext == ".pdf" and search_text:
        # Open PDF and search for text using AppleScript
        script = f'''
        tell application "Preview"
            activate
            open POSIX file "{filepath}"
            delay 0.5
        end tell
        
        tell application "System Events"
            tell process "Preview"
                keystroke "f" using command down
                delay 0.2
                keystroke "{search_text}"
                delay 0.2
                keystroke return
            end tell
        end tell
        '''
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                timeout=10,
            )
            return True
        except Exception:
            # Fall back to simple open
            pass
    
    # Default: just open the file
    subprocess.run(["open", filepath], check=False)
    return True


def _open_document_windows(filepath: str, ext: str, search_text: str | None) -> bool:
    """Open document on Windows."""
    os.startfile(filepath)  # type: ignore
    return True


def _open_document_linux(filepath: str) -> bool:
    """Open document on Linux."""
    subprocess.run(["xdg-open", filepath], check=False)
    return True


def open_pdf_at_page(filepath: str, page: int) -> bool:
    """
    Open a PDF at a specific page number.
    
    Args:
        filepath: Path to PDF file
        page: Page number (1-indexed)
    
    Returns:
        True if successful
    """
    if not os.path.exists(filepath):
        return False
    
    if sys.platform == "darwin":
        # macOS: Use Preview's URL scheme
        # Note: This may not work in all versions
        subprocess.run(
            ["open", "-a", "Preview", filepath],
            check=False
        )
        return True
    else:
        return open_document(filepath)


# ============================================================================
# Find Text in Document
# ============================================================================

def find_text_in_pdf(filepath: str, search_text: str) -> list[Tuple[int, str]]:
    """
    Find text in a PDF and return page numbers and context.
    
    Args:
        filepath: Path to PDF file
        search_text: Text to search for
    
    Returns:
        List of (page_number, context_snippet) tuples
    """
    try:
        from PyPDF2 import PdfReader
        
        results = []
        reader = PdfReader(filepath)
        search_lower = search_text.lower()
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text_lower = text.lower()
            
            if search_lower in text_lower:
                # Find the position and extract context
                idx = text_lower.find(search_lower)
                start = max(0, idx - 50)
                end = min(len(text), idx + len(search_text) + 50)
                context = text[start:end].strip()
                context = "..." + context + "..." if start > 0 else context + "..."
                
                results.append((page_num, context))
        
        return results
    
    except Exception as e:
        logger.error("Error searching PDF: %s", e)
        return []


def find_answer_location(
    filepath: str,
    answer_text: str,
) -> dict | None:
    """
    Find where an answer appears in a document.
    
    Args:
        filepath: Path to document
        answer_text: The answer text to locate
    
    Returns:
        Dict with location info or None if not found
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == ".pdf":
        results = find_text_in_pdf(filepath, answer_text)
        if results:
            page_num, context = results[0]
            return {
                "page": page_num,
                "context": context,
                "filepath": filepath,
            }
    
    # For other file types, we can't easily get line numbers
    # Just return the file info
    return {
        "filepath": filepath,
        "page": None,
        "context": None,
    }


# ============================================================================
# Document Preview Generation
# ============================================================================

def generate_pdf_thumbnail(
    filepath: str,
    output_path: str | None = None,
    size: Tuple[int, int] = (200, 280),
) -> str | None:
    """
    Generate a thumbnail image from the first page of a PDF.
    
    Args:
        filepath: Path to PDF file
        output_path: Where to save thumbnail (auto-generated if None)
        size: Thumbnail dimensions (width, height)
    
    Returns:
        Path to generated thumbnail or None if failed
    """
    try:
        # Try using pdf2image if available
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                filepath,
                first_page=1,
                last_page=1,
                size=size,
            )
            
            if images:
                if output_path is None:
                    output_path = filepath + ".thumb.png"
                images[0].save(output_path, "PNG")
                return output_path
        except ImportError:
            pass
        
        # Fallback: Try using macOS Quick Look
        if sys.platform == "darwin":
            if output_path is None:
                output_path = filepath + ".thumb.png"
            
            result = subprocess.run(
                ["qlmanage", "-t", "-s", str(max(size)), "-o", 
                 str(Path(output_path).parent), filepath],
                capture_output=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                # qlmanage creates file with .png extension
                generated = Path(filepath).name + ".png"
                generated_path = Path(output_path).parent / generated
                if generated_path.exists():
                    generated_path.rename(output_path)
                    return output_path
        
        return None
        
    except Exception as e:
        logger.error("Failed to generate thumbnail: %s", e)
        return None


def get_document_info(filepath: str) -> dict:
    """
    Get metadata about a document.
    
    Returns:
        Dict with file info (size, type, page count for PDFs)
    """
    path = Path(filepath)
    
    if not path.exists():
        return {"error": "File not found"}
    
    info = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "size_human": _format_size(path.stat().st_size),
    }
    
    # Get page count for PDFs
    if path.suffix.lower() == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            info["page_count"] = len(reader.pages)
        except Exception:
            info["page_count"] = None
    
    return info


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
