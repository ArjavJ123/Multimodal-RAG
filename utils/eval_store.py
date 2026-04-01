# utils/eval_store.py — Persistent eval storage under ./eval_runs/
# Mirrors chat_store.py but writes to eval_runs/run_NNN/query_NNN/ instead of chats/.

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List

import settings
from utils.dataclasses import DocumentChunk

EVAL_RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_runs"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_dir(run_id: str) -> str:
    return os.path.join(EVAL_RUNS_DIR, run_id)


def _query_dir(run_id: str, query_id: str) -> str:
    return os.path.join(_run_dir(run_id), query_id)


def _read(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Path helpers (used by eval.py to pass overrides into run_ingest / run_query)
# ---------------------------------------------------------------------------

def chroma_dir_for_run(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "chroma")


def images_dir_for_run(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "images")


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def create_run(
    queries_file: str,
    docs_ingested: List[str],
    answer_model: str,
    embedding_model: str,
) -> str:
    """Create a new run directory, write run_meta.json, and return the run_id."""
    os.makedirs(EVAL_RUNS_DIR, exist_ok=True)
    existing = [
        d for d in os.listdir(EVAL_RUNS_DIR)
        if os.path.isdir(os.path.join(EVAL_RUNS_DIR, d)) and re.match(r"run_\d+$", d)
    ]
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    run_id = f"run_{(max(nums) + 1 if nums else 1):03d}"

    os.makedirs(_run_dir(run_id), exist_ok=True)
    os.makedirs(images_dir_for_run(run_id), exist_ok=True)

    _write(os.path.join(_run_dir(run_id), "run_meta.json"), {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs_ingested": docs_ingested,
        "queries_file": queries_file,
        "answer_model": answer_model,
        "embedding_model": embedding_model,
    })
    _write(os.path.join(_run_dir(run_id), "chunk_metrics.json"), [])

    return run_id


def next_query_id(run_id: str) -> str:
    """Auto-increment query_NNN inside the run directory."""
    existing = [
        d for d in os.listdir(_run_dir(run_id))
        if os.path.isdir(_query_dir(run_id, d)) and re.match(r"query_\d+$", d)
    ]
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    query_id = f"query_{(max(nums) + 1 if nums else 1):03d}"
    os.makedirs(_query_dir(run_id, query_id), exist_ok=True)
    return query_id


# ---------------------------------------------------------------------------
# Chunk count helpers
# ---------------------------------------------------------------------------

def peek_next_chunk_idx(run_id: str) -> int:
    total = 0
    for entry in _read(os.path.join(_run_dir(run_id), "chunk_metrics.json")):
        for doc in entry.get("docs", []):
            total += doc.get("num_chunks", 0)
    return total + 1


def get_total_chunk_count(run_id: str) -> int:
    total = 0
    for entry in _read(os.path.join(_run_dir(run_id), "chunk_metrics.json")):
        for doc in entry.get("docs", []):
            total += doc.get("num_chunks", 0)
    return total


# ---------------------------------------------------------------------------
# Ingest metrics (written once per run, not per query)
# ---------------------------------------------------------------------------

def append_chunk_metrics(
    run_id: str,
    final_chunks: List[DocumentChunk],
    doc_lengths: Dict[str, int],
) -> None:
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

    path = os.path.join(_run_dir(run_id), "chunk_metrics.json")
    data = _read(path)
    data.append({"docs": docs})
    _write(path, data)


# ---------------------------------------------------------------------------
# Query results (one set of files per query_NNN)
# ---------------------------------------------------------------------------

def append_query_result(
    run_id: str,
    query_id: str,
    question: str,
    answer: str,
    timestamp: str,
) -> None:
    _write(os.path.join(_query_dir(run_id, query_id), "query_meta.json"), {
        "query_id": query_id,
        "run_id": run_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
    })


def append_chunk_entries(run_id: str, query_id: str, filename: str, entries: list) -> None:
    """Write retrieved or used chunk entries grouped by source file."""
    if not entries:
        _write(os.path.join(_query_dir(run_id, query_id), filename),
               {"query_id": query_id, "docs": []})
        return

    by_source: dict = {}
    for e in entries:
        d = e.model_dump()
        src = d.pop("source_file")
        d.pop("message_id", None)
        by_source.setdefault(src, []).append(d)

    docs = [{"source_file": src, "chunks": chunks} for src, chunks in by_source.items()]
    _write(os.path.join(_query_dir(run_id, query_id), filename),
           {"query_id": query_id, "docs": docs})


def append_retrieval_metrics(
    run_id: str,
    query_id: str,
    retrieved: List[DocumentChunk],
) -> None:
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

    _write(os.path.join(_query_dir(run_id, query_id), "retrieval_metrics.json"), {
        "query_id": query_id,
        "total_retrieved": len(retrieved),
        "num_text_retrieved": sum(1 for c in retrieved if c.chunk_type == "text"),
        "num_table_retrieved": sum(1 for c in retrieved if c.chunk_type == "table"),
        "num_chart_retrieved": sum(1 for c in retrieved if c.chunk_type == "chart_caption"),
        "docs": docs,
    })


def append_usage_metrics(
    run_id: str,
    query_id: str,
    retrieved: List[DocumentChunk],
    used: List[DocumentChunk],
) -> None:
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

    _write(os.path.join(_query_dir(run_id, query_id), "usage_metrics.json"), {
        "query_id": query_id,
        "total_used": len(used),
        "num_text_used": sum(1 for c in used if c.chunk_type == "text"),
        "num_table_used": sum(1 for c in used if c.chunk_type == "table"),
        "num_chart_used": sum(1 for c in used if c.chunk_type == "chart_caption"),
        "docs": docs,
    })
