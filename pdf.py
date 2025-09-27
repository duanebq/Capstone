# pdf.py — exposes: extract_text(path) -> str
# Primary: pdfminer.six. Fallback (optional): PyPDF2 if installed.
# Always returns a string; never raises to the caller.

from typing import Optional

# --- primary extractor ---
try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
except Exception:  # very unlikely if pdfminer.six is installed
    _pdfminer_extract = None

# --- optional fallback (works if PyPDF2 is installed) ---
try:
    from PyPDF2 import PdfReader  # optional
except Exception:
    PdfReader = None


def _pdfminer_text(path: str) -> str:
    if _pdfminer_extract is None:
        return ""
    try:
        text = _pdfminer_extract(path) or ""
        return text.strip()
    except Exception:
        return ""


def _pypdf2_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_text(path: str) -> str:
    """
    Return plain text extracted from a PDF file at `path`.
    Uses pdfminer.six; falls back to PyPDF2 if available.
    Returns "" on failure (never raises).
    """
    text = _pdfminer_text(path)
    if not text:
        text = _pypdf2_text(path)
    return text or ""
