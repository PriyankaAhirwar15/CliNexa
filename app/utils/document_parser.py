"""
CliNexa Healthcare Intelligence Platform
Module: Document Parser
Description: Secure extraction of medical report text from PDF, DOCX, and TXT files
with file-size limits, sanitization, and graceful failure handling.
"""

import io
from typing import Dict, Any
import pypdf
import docx

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def extract_text_from_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Safely extract plain text from an uploaded document.
    """
    if not file_bytes:
        return {
            "success": False,
            "text": "",
            "error": "The uploaded file is empty."
        }

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return {
            "success": False,
            "text": "",
            "error": f"File size exceeds allowable limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        }

    ext = filename.lower().split(".")[-1] if "." in filename else ""

    try:
        if ext == "txt":
            # Attempt UTF-8 decoding with fallback
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="replace")
            return {"success": True, "text": text.strip(), "error": None}

        elif ext == "pdf":
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)
            extracted_pages = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())
            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                return {
                    "success": False,
                    "text": "",
                    "error": "Information could not be confidently extracted from the provided document. The PDF may be a scanned image without OCR text."
                }
            return {"success": True, "text": full_text, "error": None}

        elif ext in ["docx", "doc"]:
            stream = io.BytesIO(file_bytes)
            doc = docx.Document(stream)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if row_cells:
                        paragraphs.append(" | ".join(row_cells))
            full_text = "\n".join(paragraphs).strip()
            if not full_text:
                return {
                    "success": False,
                    "text": "",
                    "error": "Information could not be confidently extracted from the provided document."
                }
            return {"success": True, "text": full_text, "error": None}

        else:
            return {
                "success": False,
                "text": "",
                "error": f"Unsupported file format '.{ext}'. Supported formats are PDF, DOCX, and TXT."
            }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"Unable to parse document: {str(e)}"
        }
