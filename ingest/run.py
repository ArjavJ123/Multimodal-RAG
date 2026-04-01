# ingest/run.py — Callable ingest entry point used by app.py and eval.py.

import os
import glob
from typing import Dict, List, Optional, Tuple

from ingest.graph import build_ingest_graph
from utils.dataclasses import DocumentChunk


def run_ingest(
    file_paths: List[str],
    thread_id: str,
    message_id: str,
    chunk_start_idx: int,
    chroma_dir: Optional[str] = None,
    images_dir: Optional[str] = None,
) -> Tuple[List[DocumentChunk], Dict[str, int]]:
    """
    Run the ingest LangGraph pipeline and return the final chunks and doc lengths.

    Args:
        file_paths:       Absolute paths to the files to ingest.
        thread_id:        Thread or run identifier (used for chroma/image paths when
                          chroma_dir/images_dir are not supplied).
        message_id:       Message ID for chunk ID continuity.
        chunk_start_idx:  Starting index for chunk IDs (ensures uniqueness across batches).
        chroma_dir:       Override chroma persist directory (eval mode). None → default.
        images_dir:       Override images save directory (eval mode). None → default.

    Returns:
        (final_chunks, doc_lengths)
    """
    state = {
        "file_paths": file_paths,
        "thread_id": thread_id,
        "message_id": message_id,
        "chunk_start_idx": chunk_start_idx,
    }
    if chroma_dir is not None:
        state["chroma_dir"] = chroma_dir
    if images_dir is not None:
        state["images_dir"] = images_dir

    graph = build_ingest_graph()
    result = graph.invoke(state)
    return result.get("final_chunks", []), result.get("doc_lengths", {})
