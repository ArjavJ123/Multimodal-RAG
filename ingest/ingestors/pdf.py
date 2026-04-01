# ingest/ingestors/pdf.py — PDF ingestor.
#
# Text   : pdfplumber page.extract_text() — one Document per page
# Tables : pdfplumber page.extract_tables() — one table chunk per table,
#          formatted as a markdown table so headers stay with data rows
# Images : pymupdf (fitz) page.get_images() — one ImageCrop per embedded image

import os
from typing import List

import fitz
import pdfplumber
from langchain_core.documents import Document

from utils.dataclasses import ImageCrop, IngestorResult


def _to_markdown(table: List[List]) -> str:
    """
    Convert a pdfplumber table (list of rows, each a list of cell strings)
    into a markdown table string.

    None cells are replaced with empty strings. Each row is deduplicated
    for repeated merged cells that pdfplumber sometimes emits.
    """
    cleaned = [[str(cell).strip() if cell is not None else "" for cell in row]
               for row in table if any(cell for cell in row)]
    if len(cleaned) < 2:
        return ""

    # Determine column count from the widest row
    ncols = max(len(row) for row in cleaned)

    def pad(row: List[str]) -> List[str]:
        return row + [""] * (ncols - len(row))

    header = pad(cleaned[0])
    md_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in cleaned[1:]:
        md_rows.append("| " + " | ".join(pad(row)) + " |")

    return "\n".join(md_rows)


class PDFIngestor:
    def ingest(self, path: str) -> IngestorResult:
        source = os.path.basename(path)
        text_docs: List[Document] = []
        table_chunks: List[dict] = []
        image_crops: List[ImageCrop] = []

        # ── Text + Tables ────────────────────────────────────────────────────
        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages):

                # Text — extract full page text (includes table text as plain lines;
                # the dedicated table chunks below provide the structured view)
                text = page.extract_text()
                if text and text.strip():
                    text_docs.append(Document(
                        page_content=text,
                        metadata={"source": source, "page": page_idx + 1},
                    ))

                # Tables — keep each table whole so headers are never orphaned
                tables = page.extract_tables()
                for table in (tables or []):
                    if not table or len(table) < 2:
                        continue
                    md = _to_markdown(table)
                    if md:
                        table_chunks.append({
                            "text": md,
                            "source_file": source,
                            "page_number": page_idx + 1,
                        })

        print(f"[PDFIngestor] {source}: {len(text_docs)} page(s), "
              f"{len(table_chunks)} table(s)")

        # ── Images ───────────────────────────────────────────────────────────
        doc = fitz.open(path)
        for page_idx, page in enumerate(doc):
            for ref in page.get_images(full=True):
                xref = ref[0]
                image_data = doc.extract_image(xref)
                image_crops.append(ImageCrop(
                    page=page_idx + 1,
                    image_bytes=image_data["image"],
                    source_file=source,
                ))
        doc.close()

        print(f"[PDFIngestor] {source}: {len(image_crops)} image(s)")

        return IngestorResult(
            text_docs=text_docs,
            table_chunks=table_chunks,
            image_crops=image_crops,
        )
