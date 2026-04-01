# utils/dataclasses.py — Shared data models and graph state definitions.
#
# Chunk hierarchy (graph / in-flight):
#
#   ChunkBase          ← chunk_id, source_file, page_number, chunk_type, text
#   ├── DocumentChunk  ← adds chart_name, chart_description, image_path  (graph state)
#   └── BaseChunkEntry ← adds message_id, thread_id, timestamp           (JSON input)
#       ├── TextChunkEntry
#       └── ChartChunkEntry
#
# JSON storage format (*_chunks.json):
#
#   List[MessageChunkGroup]
#   └── MessageChunkGroup  ← message_id, docs
#       └── DocEntry       ← source_file, chunks
#           └── StoredChunk ← all chunk fields except message_id and source_file

from dataclasses import dataclass, field
from typing import Annotated, Any, List, Literal, Optional, TypedDict, Union

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------

class ImageCrop(BaseModel):
    page: int
    image_bytes: bytes
    source_file: str


class CaptionOutput(BaseModel):
    """Structured caption extracted from a chart or image in a business document."""

    chart_name: str = Field(
        description="A short, specific name for the chart, e.g. 'Quarterly Revenue Bar Chart 2023-2024'."
    )
    description: str = Field(
        description=(
            "Detailed description covering: (1) chart type, (2) axis labels and units, "
            "(3) key data values or ranges, and (4) the overall trend or insight shown."
        )
    )


class Caption(BaseModel):
    page: int
    chart_name: str
    description: str
    source_file: str


# ---------------------------------------------------------------------------
# Shared chunk base — fields common to ALL chunk representations
# ---------------------------------------------------------------------------

class ChunkBase(BaseModel):
    """Identity and content fields shared by every chunk type."""
    chunk_id: str = ""
    source_file: str
    page_number: int
    chunk_type: Literal["text", "table", "chart_caption"] = "text"
    text: str


# ---------------------------------------------------------------------------
# Graph state chunk — extends ChunkBase with chart-specific graph fields
# ---------------------------------------------------------------------------

class DocumentChunk(ChunkBase):
    """Chunk flowing through graph state. Adds optional chart visual metadata."""
    chart_name: Optional[str] = None
    chart_description: Optional[str] = None
    image_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Storage chunk base — extends ChunkBase with audit/storage fields
# ---------------------------------------------------------------------------

class BaseChunkEntry(ChunkBase):
    """Base for all chunk types written to JSON files. Adds storage metadata."""
    message_id: str
    thread_id: str
    timestamp: str


class TextChunkEntry(BaseChunkEntry):
    """A plain text or table chunk entry."""
    chunk_type: Literal["text", "table"] = "text"


class ChartChunkEntry(BaseChunkEntry):
    """A chart chunk entry — adds visual metadata on top of BaseChunkEntry."""
    chunk_type: Literal["chart_caption"] = "chart_caption"
    chart_name: str
    image_path: str      # absolute path to the saved PNG
    description: str     # structured caption from GPT vision


# Discriminated union — chunk_type selects the concrete model at deserialisation
ChunkEntry = Annotated[
    Union[TextChunkEntry, ChartChunkEntry],
    Field(discriminator="chunk_type"),
]


# ---------------------------------------------------------------------------
# JSON storage models — the nested structure written to *_chunks.json
# ---------------------------------------------------------------------------

class StoredChunk(BaseModel):
    """Chunk fields stored inside a DocEntry (source_file and message_id lifted out)."""
    chunk_id: str
    thread_id: str
    page_number: int
    chunk_type: Literal["text", "table", "chart_caption"] = "text"
    text: str
    timestamp: str
    chart_name: Optional[str] = None
    image_path: Optional[str] = None
    description: Optional[str] = None


class DocEntry(BaseModel):
    """Groups chunks that belong to the same source document within a message."""
    source_file: str
    chunks: List[StoredChunk]


class MessageChunkGroup(BaseModel):
    """Top-level entry in *_chunks.json — one per message_id."""
    message_id: str
    docs: List[DocEntry]


# ---------------------------------------------------------------------------
# Chat message entry (written to messages.json)
# ---------------------------------------------------------------------------

class MessageEntry(BaseModel):
    """One Q&A turn written to messages.json."""
    message_id: str
    thread_id: str
    question: str
    answer: str
    history_context: List[str]   # message_ids of previous turns passed to the LLM
    timestamp: str


# ---------------------------------------------------------------------------
# Ingestor result — returned by every file-type ingestor
# ---------------------------------------------------------------------------

@dataclass
class IngestorResult:
    """Unified output returned by every file-type ingestor."""
    text_docs: List[Document] = field(default_factory=list)
    # Each entry: {"text": str, "source_file": str, "page_number": int}
    table_chunks: List[dict] = field(default_factory=list)
    image_crops: List["ImageCrop"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph state TypedDicts
# ---------------------------------------------------------------------------

class IngestState(TypedDict):
    file_paths: List[str]
    thread_id: str          # passed from app.py — needed for image saving + chroma path
    message_id: str         # pre-computed by app.py before ingest starts
    chunk_start_idx: int    # offset so chunk IDs are unique across ingest batches
    text_docs: List[Any]    # one Document per page/file — fed into text splitter
    table_chunks: List[Any] # raw table dicts {text, source_file, page_number}
    image_crops: List[ImageCrop]
    captions: List[Caption]
    final_chunks: List[DocumentChunk]
    doc_lengths: Any        # Dict[source_file, total_chars] — computed in chunk_and_embed
    chroma_dir: Optional[str]   # override chroma path (eval mode); None → use settings.chroma_path
    images_dir: Optional[str]   # override images path (eval mode); None → use chat_store default


class QueryState(TypedDict):
    question: str
    thread_id: str           # needed to open the correct per-thread ChromaDB
    chat_history: Optional[List[Any]]
    retrieved_chunks: List[DocumentChunk]
    answer: str
    chroma_dir: Optional[str]   # override chroma path (eval mode); None → use settings.chroma_path
    total_chunks: Optional[int]  # pre-computed chunk count (eval mode); None → look up from chat_store
