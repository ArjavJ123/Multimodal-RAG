# ingest/ingestors/base.py — File-type dispatcher for document ingestion.

import os

from ingest.ingestors.pdf import PDFIngestor
from ingest.ingestors.docx import DocxIngestor
from ingest.ingestors.txt import TxtIngestor
from utils.dataclasses import IngestorResult


def ingest_file(path: str) -> IngestorResult:
    """
    Dispatch to the correct ingestor based on file extension.

    Supported:
        .pdf  → PDFIngestor  (text + tables via pdfplumber, images via pymupdf)
        .docx → DocxIngestor (text via docx2txt, tables via python-docx)
        .txt  → TxtIngestor  (text via TextLoader)
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return PDFIngestor().ingest(path)
    elif ext == ".docx":
        return DocxIngestor().ingest(path)
    elif ext == ".txt":
        return TxtIngestor().ingest(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
