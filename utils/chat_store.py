# utils/chat_store.py — Persistent chat storage under ./chats/

import json
import os
import re
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from utils.dataclasses import ChunkEntry, DocumentChunk, MessageEntry

CHATS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chats"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _thread_dir(thread_id: str) -> str:
    return os.path.join(CHATS_DIR, thread_id)


def _images_dir(thread_id: str) -> str:
    return os.path.join(_thread_dir(thread_id), "images")


def _chunk_file(thread_id: str, filename: str) -> str:
    """Path for files inside message_chunk_metadata/."""
    return os.path.join(_thread_dir(thread_id), "message_chunk_metadata", filename)


def _eval_file(thread_id: str, filename: str) -> str:
    """Path for files inside eval_metrics/."""
    return os.path.join(_thread_dir(thread_id), "eval_metrics", filename)


def _file(thread_id: str, filename: str) -> str:
    """Path for top-level thread files (messages.json)."""
    return os.path.join(_thread_dir(thread_id), filename)


def _read(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------

def create_thread() -> str:
    """Create a new thread directory with all subfolders and empty JSON files."""
    os.makedirs(CHATS_DIR, exist_ok=True)
    existing = [
        d for d in os.listdir(CHATS_DIR)
        if os.path.isdir(os.path.join(CHATS_DIR, d)) and re.match(r"thread_\d+$", d)
    ]
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    thread_id = f"thread_{(max(nums) + 1 if nums else 1):03d}"

    os.makedirs(_images_dir(thread_id), exist_ok=True)
    os.makedirs(os.path.join(_thread_dir(thread_id), "message_chunk_metadata"), exist_ok=True)
    os.makedirs(os.path.join(_thread_dir(thread_id), "eval_metrics"), exist_ok=True)

    _write(_file(thread_id, "messages.json"), [])
    for fname in ("uploaded_chunks.json", "retrieved_chunks.json", "used_chunks.json"):
        _write(_chunk_file(thread_id, fname), [])
    for fname in ("chunk_metrics.json", "retrieval_metrics.json", "usage_metrics.json"):
        _write(_eval_file(thread_id, fname), [])

    return thread_id


# ---------------------------------------------------------------------------
# ID peeking
# ---------------------------------------------------------------------------

def peek_next_message_id(thread_id: str) -> str:
    n = len(_read(_file(thread_id, "messages.json")))
    return f"msg_{n + 1:03d}"


def peek_next_chunk_idx(thread_id: str) -> int:
    total = 0
    for msg_group in _read(_chunk_file(thread_id, "uploaded_chunks.json")):
        for doc in msg_group.get("docs", []):
            total += len(doc.get("chunks", []))
    return total + 1


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def images_dir_for_thread(thread_id: str) -> str:
    return _images_dir(thread_id)


def save_image(images_dir: str, chunk_id: str, image_bytes: bytes) -> str:
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, f"{chunk_id}.png")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


# ---------------------------------------------------------------------------
# Chunk metadata write
# ---------------------------------------------------------------------------

def append_chunk_entries(thread_id: str, filename: str, entries: list) -> None:
    """
    Write chunk entries to message_chunk_metadata/{filename} using grouped format:
      List[{ message_id, docs: [{ source_file, chunks: [...] }] }]
    source_file and message_id are lifted out of each chunk dict.
    """
    if not entries:
        return
    path = _chunk_file(thread_id, filename)
    data = _read(path)

    message_id = entries[0].message_id

    by_source: dict = {}
    for e in entries:
        d = e.model_dump()
        src = d.pop("source_file")
        d.pop("message_id")
        by_source.setdefault(src, []).append(d)

    msg_group = next((m for m in data if m["message_id"] == message_id), None)
    if msg_group is None:
        msg_group = {"message_id": message_id, "docs": []}
        data.append(msg_group)

    for src, chunks in by_source.items():
        doc = next((d for d in msg_group["docs"] if d["source_file"] == src), None)
        if doc is None:
            msg_group["docs"].append({"source_file": src, "chunks": chunks})
        else:
            doc["chunks"].extend(chunks)

    _write(path, data)


# ---------------------------------------------------------------------------
# Eval metrics write
# ---------------------------------------------------------------------------

def append_chunk_metrics(
    thread_id: str,
    message_id: str,
    final_chunks: List[DocumentChunk],
    doc_lengths: Dict[str, int],
) -> None:
    """
    Write chunk_metrics.json entry for an ingest batch.

    Per source file:
      num_chunks, num_text_chunks, num_table_chunks, num_chart_chunks,
      doc_length_chars, chunks: [{chunk_id, chunk_type, length_chars}]
    """
    by_source: dict = {}
    for c in final_chunks:
        by_source.setdefault(c.source_file, []).append(c)

    docs = []
    for src, chunks in by_source.items():
        text_n  = sum(1 for c in chunks if c.chunk_type == "text")
        table_n = sum(1 for c in chunks if c.chunk_type == "table")
        chart_n = sum(1 for c in chunks if c.chunk_type == "chart_caption")
        docs.append({
            "source_file": src,
            "num_chunks": len(chunks),
            "num_text_chunks": text_n,
            "num_table_chunks": table_n,
            "num_chart_chunks": chart_n,
            "doc_length_chars": doc_lengths.get(src, 0),
            "chunks": [
                {"chunk_id": c.chunk_id, "chunk_type": c.chunk_type, "length_chars": len(c.text)}
                for c in chunks
            ],
        })

    path = _eval_file(thread_id, "chunk_metrics.json")
    data = _read(path)
    data.append({"message_id": message_id, "docs": docs})
    _write(path, data)


def append_retrieval_metrics(
    thread_id: str,
    message_id: str,
    retrieved: List[DocumentChunk],
) -> None:
    """
    Write retrieval_metrics.json entry for a query turn.

    Global: total_retrieved, num_text_retrieved, num_table_retrieved, num_chart_retrieved.
    Per source file: num_chunks_fetched, num_text_fetched, num_table_fetched, num_chart_fetched,
                     chunks: [{chunk_id, rank, chunk_type}]
    Rank 1 = most similar.
    """
    by_source: dict = {}
    for rank, chunk in enumerate(retrieved, 1):
        by_source.setdefault(chunk.source_file, []).append(
            {"chunk_id": chunk.chunk_id, "rank": rank, "chunk_type": chunk.chunk_type}
        )

    docs = []
    for src, chunks in by_source.items():
        text_n  = sum(1 for c in chunks if c["chunk_type"] == "text")
        table_n = sum(1 for c in chunks if c["chunk_type"] == "table")
        chart_n = sum(1 for c in chunks if c["chunk_type"] == "chart_caption")
        docs.append({
            "source_file": src,
            "num_chunks_fetched": len(chunks),
            "num_text_fetched": text_n,
            "num_table_fetched": table_n,
            "num_chart_fetched": chart_n,
            "chunks": chunks,
        })

    path = _eval_file(thread_id, "retrieval_metrics.json")
    data = _read(path)
    data.append({
        "message_id": message_id,
        "total_retrieved": len(retrieved),
        "num_text_retrieved": sum(1 for c in retrieved if c.chunk_type == "text"),
        "num_table_retrieved": sum(1 for c in retrieved if c.chunk_type == "table"),
        "num_chart_retrieved": sum(1 for c in retrieved if c.chunk_type == "chart_caption"),
        "docs": docs,
    })
    _write(path, data)


def append_usage_metrics(
    thread_id: str,
    message_id: str,
    retrieved: List[DocumentChunk],
    used: List[DocumentChunk],
) -> None:
    """
    Write usage_metrics.json entry for a query turn.

    Global: total_used, num_text_used, num_table_used, num_chart_used.
    Per source file: num_chunks_used, num_text_used, num_table_used, num_chart_used,
                     chunks: [{chunk_id, rank_in_retrieval, chunk_type}]
    rank_in_retrieval = position in the retrieved list (1-based); -1 if not found.
    """
    rank_map = {c.chunk_id: rank for rank, c in enumerate(retrieved, 1)}

    by_source: dict = {}
    for chunk in used:
        by_source.setdefault(chunk.source_file, []).append({
            "chunk_id": chunk.chunk_id,
            "rank_in_retrieval": rank_map.get(chunk.chunk_id, -1),
            "chunk_type": chunk.chunk_type,
        })

    docs = []
    for src, chunks in by_source.items():
        text_n  = sum(1 for c in chunks if c["chunk_type"] == "text")
        table_n = sum(1 for c in chunks if c["chunk_type"] == "table")
        chart_n = sum(1 for c in chunks if c["chunk_type"] == "chart_caption")
        docs.append({
            "source_file": src,
            "num_chunks_used": len(chunks),
            "num_text_used": text_n,
            "num_table_used": table_n,
            "num_chart_used": chart_n,
            "chunks": chunks,
        })

    path = _eval_file(thread_id, "usage_metrics.json")
    data = _read(path)
    data.append({
        "message_id": message_id,
        "total_used": len(used),
        "num_text_used": sum(1 for c in used if c.chunk_type == "text"),
        "num_table_used": sum(1 for c in used if c.chunk_type == "table"),
        "num_chart_used": sum(1 for c in used if c.chunk_type == "chart_caption"),
        "docs": docs,
    })
    _write(path, data)


# ---------------------------------------------------------------------------
# History / display reads
# ---------------------------------------------------------------------------

def get_total_chunk_count(thread_id: str) -> int:
    """Return total chunks indexed for a thread by summing chunk_metrics.json."""
    total = 0
    for entry in _read(_eval_file(thread_id, "chunk_metrics.json")):
        for doc in entry.get("docs", []):
            total += doc.get("num_chunks", 0)
    return total


def get_lc_history(thread_id: str) -> List[BaseMessage]:
    history: List[BaseMessage] = []
    for entry in _read(_file(thread_id, "messages.json")):
        history.append(HumanMessage(content=entry["question"]))
        history.append(AIMessage(content=entry["answer"]))
    return history


def get_all_message_ids(thread_id: str) -> List[str]:
    return [e["message_id"] for e in _read(_file(thread_id, "messages.json"))]


def get_messages_for_display(thread_id: str) -> list:
    display = []
    for entry in _read(_file(thread_id, "messages.json")):
        display.append({"role": "user",      "content": entry["question"]})
        display.append({"role": "assistant", "content": entry["answer"]})
    return display


def get_uploaded_docs(thread_id: str) -> dict:
    docs: dict = {}
    for msg_group in _read(_chunk_file(thread_id, "uploaded_chunks.json")):
        mid = msg_group.get("message_id", "")
        for doc_entry in msg_group.get("docs", []):
            src = doc_entry["source_file"]
            for chunk in doc_entry.get("chunks", []):
                docs.setdefault(src, []).append(
                    {**chunk, "source_file": src, "message_id": mid}
                )
    return docs


def append_message_entry(thread_id: str, entry: MessageEntry) -> None:
    path = _file(thread_id, "messages.json")
    data = _read(path)
    data.append(entry.model_dump())
    _write(path, data)
