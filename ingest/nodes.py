# ingest/nodes.py — Node functions for the ingest LangGraph.
# Each node receives the full IngestState and returns a partial dict to merge back.

import os
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from utils.dataclasses import IngestState, DocumentChunk
from utils.visuals import caption_image_crops
from utils.chat_store import save_image, images_dir_for_thread
from ingest.ingestors.base import ingest_file
import settings


# ---------------------------------------------------------------------------
# Node 1: load_documents
# ---------------------------------------------------------------------------

def load_documents(state: IngestState) -> Dict[str, Any]:
    """
    Dispatch each file to its type-specific ingestor and aggregate results.

    Each ingestor returns an IngestorResult with:
        text_docs    — one LangChain Document per page/block (for text splitting)
        table_chunks — whole tables as markdown dicts (not split)
        image_crops  — embedded image bytes (for vision captioning)

    Returns:
        {"text_docs": ..., "table_chunks": ..., "image_crops": ...}
    """
    all_text_docs: List[Document] = []
    all_table_chunks: List[dict] = []
    all_image_crops = []

    for path in state["file_paths"]:
        try:
            result = ingest_file(path)
            all_text_docs.extend(result.text_docs)
            all_table_chunks.extend(result.table_chunks)
            all_image_crops.extend(result.image_crops)
        except Exception as e:
            print(f"[load_documents] Failed to ingest {path}: {e}")

    print(f"[load_documents] Total: {len(all_text_docs)} text page(s), "
          f"{len(all_table_chunks)} table(s), {len(all_image_crops)} image(s)")

    return {
        "text_docs": all_text_docs,
        "table_chunks": all_table_chunks,
        "image_crops": all_image_crops,
    }


# ---------------------------------------------------------------------------
# Node 2: caption_visuals
# ---------------------------------------------------------------------------

def caption_visuals(state: IngestState) -> Dict[str, Any]:
    """
    Send each ImageCrop to GPT vision for structured captioning.

    Returns:
        {"captions": List[Caption]}
    """
    captions = caption_image_crops(state["image_crops"])
    return {"captions": captions}


# ---------------------------------------------------------------------------
# Node 3: chunk_and_embed
# ---------------------------------------------------------------------------

def chunk_and_embed(state: IngestState) -> Dict[str, Any]:
    """
    1. Split text_docs into overlapping text chunks.
    2. Keep table_chunks whole (no splitting — headers must stay with data rows).
    3. Create one chart_caption chunk per caption.
    4. Embed all chunks and upsert to this thread's ChromaDB collection.

    Returns:
        {"final_chunks": List[DocumentChunk], "doc_lengths": Dict[str, int]}
    """
    thread_id = state["thread_id"]
    chunk_start_idx = state["chunk_start_idx"]
    chroma_dir = state.get("chroma_dir") or settings.chroma_path(thread_id)
    images_dir = state.get("images_dir") or images_dir_for_thread(thread_id)

    # Caption lookup: (source_file, page) → (chart_name, description)
    caption_lookup: Dict[Tuple[str, int], Tuple[str, str]] = {
        (c.source_file, c.page): (c.chart_name, c.description)
        for c in state.get("captions", [])
    }

    # Image bytes lookup: (source_file, page) → bytes
    image_lookup: Dict[Tuple[str, int], bytes] = {
        (crop.source_file, crop.page): crop.image_bytes
        for crop in state.get("image_crops", [])
    }

    # Doc lengths: total characters per source file across all text pages
    doc_lengths: Dict[str, int] = {}
    for doc in state["text_docs"]:
        src = os.path.basename(doc.metadata.get("source", ""))
        doc_lengths[src] = doc_lengths.get(src, 0) + len(doc.page_content)

    # Split text pages into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    split_text_chunks = splitter.split_documents(state["text_docs"])

    all_lc_docs: List[Document] = []
    final_chunks: List[DocumentChunk] = []
    idx = chunk_start_idx

    # 1. Text chunks
    for chunk in split_text_chunks:
        chunk_id = f"chunk_{idx:03d}"
        idx += 1
        source_file = os.path.basename(chunk.metadata.get("source", ""))
        page = chunk.metadata.get("page", -1)
        chunk.metadata.update({"chunk_type": "text", "chunk_id": chunk_id})

        all_lc_docs.append(chunk)
        final_chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            text=chunk.page_content,
            source_file=source_file,
            page_number=page,
            chunk_type="text",
        ))

    # 2. Table chunks — stored whole, not split
    for tbl in state.get("table_chunks", []):
        chunk_id = f"chunk_{idx:03d}"
        idx += 1
        lc_doc = Document(
            page_content=tbl["text"],
            metadata={
                "source": tbl["source_file"],
                "page": tbl["page_number"],
                "chunk_type": "table",
                "chunk_id": chunk_id,
            },
        )
        all_lc_docs.append(lc_doc)
        final_chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            text=tbl["text"],
            source_file=tbl["source_file"],
            page_number=tbl["page_number"],
            chunk_type="table",
        ))

    # 3. Chart caption chunks — one per captioned image
    for (source_file, page), (chart_name, description) in caption_lookup.items():
        chunk_id = f"chunk_{idx:03d}"
        idx += 1

        image_path: Optional[str] = None
        img_bytes = image_lookup.get((source_file, page))
        if img_bytes:
            image_path = save_image(images_dir, chunk_id, img_bytes)

        chart_text = f"[Chart — {chart_name}]: {description}"
        lc_doc = Document(
            page_content=chart_text,
            metadata={
                "source": source_file,
                "page": page,
                "chunk_type": "chart_caption",
                "chunk_id": chunk_id,
                "chart_name": chart_name,
                "chart_description": description,
                "image_path": image_path or "",
            },
        )
        all_lc_docs.append(lc_doc)
        final_chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            text=chart_text,
            source_file=source_file,
            page_number=page,
            chunk_type="chart_caption",
            chart_name=chart_name,
            chart_description=description,
            image_path=image_path,
        ))

    print(f"[chunk_and_embed] {len(split_text_chunks)} text + "
          f"{len(state.get('table_chunks', []))} table + "
          f"{len(caption_lookup)} chart = {len(final_chunks)} total chunk(s)")

    # Embed and persist
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(all_lc_docs)

    print(f"[chunk_and_embed] Upserted {len(all_lc_docs)} chunk(s) to ChromaDB")

    return {"final_chunks": final_chunks, "doc_lengths": doc_lengths}


# ---------------------------------------------------------------------------
# Conditional edge helper
# ---------------------------------------------------------------------------

def has_images(state: IngestState) -> str:
    """Route to caption_visuals if images were found, else skip to chunk_and_embed."""
    if state.get("image_crops"):
        return "caption_visuals"
    return "chunk_and_embed"
