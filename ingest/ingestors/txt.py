# ingest/ingestors/txt.py — Plain text ingestor.
#
# Text   : TextLoader — full file as a single Document
# Tables : none (plain text has no structured table metadata)
# Images : none

import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from utils.dataclasses import IngestorResult


class TxtIngestor:
    def ingest(self, path: str) -> IngestorResult:
        source = os.path.basename(path)
        text_docs: List[Document] = TextLoader(path).load()
        for doc in text_docs:
            doc.metadata["source"] = source

        print(f"[TxtIngestor] {source}: {len(text_docs)} doc(s)")

        return IngestorResult(
            text_docs=text_docs,
            table_chunks=[],
            image_crops=[],
        )
