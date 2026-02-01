"""
Natural Language Query Filters.

Parses queries like:
- "PDFs from last week"
- "documents modified in January"
- "Excel files from 2024"

Extracts filters and the actual search query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class QueryFilters:
    """Parsed filters from a natural language query."""
    
    # The actual search query (with filter text removed)
    query: str
    
    # File type filters
    file_types: list[str] = field(default_factory=list)
    
    # Date filters
    date_from: datetime | None = None
    date_to: datetime | None = None
    
    # Directory filter
    directory: str | None = None
    
    # Original query for reference
    original_query: str = ""
    
    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.file_types or 
            self.date_from or 
            self.date_to or 
            self.directory
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "file_types": self.file_types,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "directory": self.directory,
            "has_filters": self.has_filters(),
        }


# ============================================================================
# File Type Patterns
# ============================================================================

FILE_TYPE_PATTERNS = {
    # PDFs
    r"\bpdf(?:s)?\b": [".pdf"],
    r"\bdocument(?:s)?\b": [".pdf", ".docx", ".doc"],
    
    # Word documents
    r"\bword\b": [".docx", ".doc"],
    r"\bdocx?\b": [".docx", ".doc"],
    
    # Spreadsheets
    r"\bexcel\b": [".xlsx", ".xls", ".csv"],
    r"\bspreadsheet(?:s)?\b": [".xlsx", ".xls", ".csv"],
    r"\bcsv\b": [".csv"],
    r"\bxlsx?\b": [".xlsx", ".xls"],
    
    # Text files
    r"\btext file(?:s)?\b": [".txt", ".md"],
    r"\btxt\b": [".txt"],
    r"\bmarkdown\b": [".md"],
    
    # Images
    r"\bimage(?:s)?\b": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
    r"\bphoto(?:s)?\b": [".jpg", ".jpeg", ".png"],
    r"\bjpe?g\b": [".jpg", ".jpeg"],
    r"\bpng\b": [".png"],
}

# ============================================================================
# Date Patterns
# ============================================================================

# Relative time patterns
RELATIVE_DATE_PATTERNS = {
    r"\btoday\b": lambda: (datetime.now().replace(hour=0, minute=0, second=0), None),
    r"\byesterday\b": lambda: (
        (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0),
        datetime.now().replace(hour=0, minute=0, second=0)
    ),
    r"\blast\s+week\b": lambda: (datetime.now() - timedelta(weeks=1), None),
    r"\bthis\s+week\b": lambda: (
        datetime.now() - timedelta(days=datetime.now().weekday()),
        None
    ),
    r"\blast\s+month\b": lambda: (datetime.now() - timedelta(days=30), None),
    r"\bthis\s+month\b": lambda: (
        datetime.now().replace(day=1, hour=0, minute=0, second=0),
        None
    ),
    r"\blast\s+year\b": lambda: (datetime.now() - timedelta(days=365), None),
    r"\bthis\s+year\b": lambda: (
        datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0),
        None
    ),
    r"\blast\s+(\d+)\s+days?\b": lambda m: (
        datetime.now() - timedelta(days=int(m.group(1))),
        None
    ),
    r"\blast\s+(\d+)\s+weeks?\b": lambda m: (
        datetime.now() - timedelta(weeks=int(m.group(1))),
        None
    ),
    r"\blast\s+(\d+)\s+months?\b": lambda m: (
        datetime.now() - timedelta(days=int(m.group(1)) * 30),
        None
    ),
}

# Month names
MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# ============================================================================
# Filter Phrases (to remove from query)
# ============================================================================

FILTER_PHRASES = [
    r"\bfrom\s+(?:the\s+)?(?:last|this|past)\s+\w+\b",
    r"\bmodified\s+(?:in|on|during|since)\s+\w+\b",
    r"\bcreated\s+(?:in|on|during|since)\s+\w+\b",
    r"\bin\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\bfrom\s+\d{4}\b",
    r"\bin\s+\d{4}\b",
    r"\bsince\s+\w+\b",
]


def parse_query(query: str) -> QueryFilters:
    """
    Parse a natural language query to extract filters.
    
    Examples:
        "PDFs from last week" -> QueryFilters(query="", file_types=[".pdf"], date_from=...)
        "salary in my documents" -> QueryFilters(query="salary", ...)
        "Excel files modified in January" -> QueryFilters(query="", file_types=[".xlsx"], ...)
    
    Args:
        query: The natural language query
    
    Returns:
        QueryFilters with extracted filters and cleaned query
    """
    original = query
    query_lower = query.lower()
    
    file_types: list[str] = []
    date_from: datetime | None = None
    date_to: datetime | None = None
    directory: str | None = None
    
    # Extract file types
    for pattern, extensions in FILE_TYPE_PATTERNS.items():
        if re.search(pattern, query_lower):
            file_types.extend(extensions)
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)
    
    # Remove duplicates
    file_types = list(set(file_types))
    
    # Extract relative dates
    for pattern, date_func in RELATIVE_DATE_PATTERNS.items():
        match = re.search(pattern, query_lower)
        if match:
            if callable(date_func):
                try:
                    # Check if the function needs the match object
                    import inspect
                    sig = inspect.signature(date_func)
                    if len(sig.parameters) > 0:
                        result = date_func(match)
                    else:
                        result = date_func()
                    
                    if isinstance(result, tuple):
                        date_from, date_to = result
                    else:
                        date_from = result
                except Exception:
                    pass
            
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)
            break
    
    # Extract month names with optional year
    month_pattern = r"\b(in|from|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b"
    match = re.search(month_pattern, query_lower)
    if match and not date_from:
        month_name = match.group(2)
        year = int(match.group(3)) if match.group(3) else datetime.now().year
        month = MONTHS.get(month_name, 1)
        
        date_from = datetime(year, month, 1)
        # End of month
        if month == 12:
            date_to = datetime(year + 1, 1, 1)
        else:
            date_to = datetime(year, month + 1, 1)
        
        query = re.sub(month_pattern, "", query, flags=re.IGNORECASE)
    
    # Extract year
    year_pattern = r"\b(?:from|in)\s+(\d{4})\b"
    match = re.search(year_pattern, query_lower)
    if match and not date_from:
        year = int(match.group(1))
        date_from = datetime(year, 1, 1)
        date_to = datetime(year + 1, 1, 1)
        query = re.sub(year_pattern, "", query, flags=re.IGNORECASE)
    
    # Extract directory patterns
    dir_pattern = r"\b(?:in|from)\s+(?:my\s+)?(documents?|desktop|downloads?|pictures?)\b"
    match = re.search(dir_pattern, query_lower)
    if match:
        folder = match.group(1).rstrip("s").capitalize()
        if folder == "Document":
            folder = "Documents"
        elif folder == "Download":
            folder = "Downloads"
        elif folder == "Picture":
            folder = "Pictures"
        directory = f"~/{folder}"
        query = re.sub(dir_pattern, "", query, flags=re.IGNORECASE)
    
    # Clean up the query
    query = re.sub(r"\s+", " ", query).strip()
    
    # Remove dangling filter words
    query = re.sub(r"^\s*(from|in|modified|created|during|since)\s*$", "", query, flags=re.IGNORECASE)
    query = query.strip()
    
    return QueryFilters(
        query=query,
        file_types=file_types,
        date_from=date_from,
        date_to=date_to,
        directory=directory,
        original_query=original,
    )


def apply_filters_to_results(
    results: list[dict],
    filters: QueryFilters,
) -> list[dict]:
    """
    Apply parsed filters to search results.
    
    Args:
        results: List of search results with 'filepath', 'filename', 'indexed_at'
        filters: Parsed query filters
    
    Returns:
        Filtered results
    """
    if not filters.has_filters():
        return results
    
    filtered = []
    
    for result in results:
        # Check file type
        if filters.file_types:
            ext = "." + result.get("filename", "").split(".")[-1].lower()
            if ext not in filters.file_types:
                continue
        
        # Check date range
        if filters.date_from or filters.date_to:
            indexed_at = result.get("indexed_at")
            if indexed_at:
                try:
                    if isinstance(indexed_at, str):
                        doc_date = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
                    else:
                        doc_date = indexed_at
                    
                    if filters.date_from and doc_date < filters.date_from:
                        continue
                    if filters.date_to and doc_date >= filters.date_to:
                        continue
                except Exception:
                    pass
        
        # Check directory
        if filters.directory:
            import os
            expanded = os.path.expanduser(filters.directory)
            filepath = result.get("filepath", "")
            if not filepath.startswith(expanded):
                continue
        
        filtered.append(result)
    
    return filtered


def format_filters_description(filters: QueryFilters) -> str:
    """
    Generate a human-readable description of active filters.
    
    Example: "PDFs from last week in Documents"
    """
    parts = []
    
    if filters.file_types:
        types = ", ".join(ext.upper().lstrip(".") for ext in filters.file_types[:3])
        if len(filters.file_types) > 3:
            types += f" +{len(filters.file_types) - 3} more"
        parts.append(types)
    
    if filters.date_from:
        if filters.date_to:
            parts.append(f"from {filters.date_from.strftime('%b %d')} to {filters.date_to.strftime('%b %d')}")
        else:
            parts.append(f"since {filters.date_from.strftime('%b %d, %Y')}")
    
    if filters.directory:
        parts.append(f"in {filters.directory}")
    
    return " ".join(parts) if parts else ""
