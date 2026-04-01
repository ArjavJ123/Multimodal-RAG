# utils/citations.py — Parse LLM answer citations and match to retrieved chunks.
#
# Citation formats (defined in settings.SYSTEM_PROMPT):
#   Text  : (`filename.pdf`, p.12)
#   Chart : (`Chart Name`, `document_name.pdf`)

import re
from typing import List

from utils.dataclasses import DocumentChunk

# (`source_file`, p.N)  — text citation
_TEXT_RE = re.compile(r'\(`([^`]+?)`\s*,\s*p\.(\d+)\)')

# (`Chart Name`, `source_file`)  — chart citation (correct format)
_CHART_RE = re.compile(r'\(`([^`]+?)`\s*,\s*`([^`]+?)`\)')

# (`source_file`)  — fallback: model cited just the doc name for a chart
_FILE_ONLY_RE = re.compile(r'\(`([^`]+?\.(?:pdf|docx|txt))`\)')


def parse_used_chunks(answer: str, retrieved: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Return the subset of retrieved chunks that are explicitly cited in the answer.

    Matching rules:
      - Text citation  (`f.pdf`, p.N)         → chunk where source_file==f.pdf and page_number==N
      - Chart citation (`Chart Name`, `f.pdf`) → chunk where chart_name==Chart Name and source_file==f.pdf

    Each chunk is included at most once regardless of how many times it is cited.
    If no citations are found (e.g. model said it lacked information), returns an empty list.
    """
    used: List[DocumentChunk] = []
    seen: set = set()

    def _add(chunk: DocumentChunk) -> None:
        if chunk.chunk_id not in seen:
            used.append(chunk)
            seen.add(chunk.chunk_id)

    for m in _TEXT_RE.finditer(answer):
        src, page = m.group(1), int(m.group(2))
        for chunk in retrieved:
            if chunk.chunk_type == "text" and chunk.source_file == src and chunk.page_number == page:
                _add(chunk)

    for m in _CHART_RE.finditer(answer):
        chart_name, src = m.group(1), m.group(2)
        for chunk in retrieved:
            if (chunk.chunk_type == "chart_caption"
                    and chunk.chart_name == chart_name
                    and chunk.source_file == src):
                _add(chunk)

    # Fallback: model wrote (`doc.pdf`) instead of the full chart citation format.
    # Match chart chunks from that document whose name appears anywhere in the answer.
    for m in _FILE_ONLY_RE.finditer(answer):
        src = m.group(1)
        for chunk in retrieved:
            if (chunk.chunk_type == "chart_caption"
                    and chunk.source_file == src
                    and chunk.chart_name
                    and chunk.chart_name in answer):
                _add(chunk)

    return used
