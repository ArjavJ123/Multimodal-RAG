# eval.py — CLI entry point for bulk ingestion + evaluation.
#
# Usage:
#   python eval.py --docs ./docs/ --queries queries.json
#   python eval.py --docs ./docs/ --queries queries.json --stream
#
# queries.json format:
#   [
#     {"query_id": "query_001", "question": "What is global GDP growth in 2025?"},
#     {"query_id": "query_002", "question": "Which countries face the highest inflation?"}
#   ]
#
# Output: eval_runs/run_NNN/ (auto-incremented)

import argparse
import glob
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

import settings
import utils.eval_store as eval_store
from ingest.run import run_ingest
from query.run import run_query
from utils.citations import parse_used_chunks
from utils.dataclasses import ChartChunkEntry, TextChunkEntry

load_dotenv()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_chunk_entry(chunk, query_id: str, run_id: str, timestamp: str):
    base = dict(
        chunk_id=chunk.chunk_id,
        message_id=query_id,
        thread_id=run_id,
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        timestamp=timestamp,
    )
    if chunk.chunk_type == "chart_caption":
        return ChartChunkEntry(
            **base,
            chunk_type="chart_caption",
            chart_name=chunk.chart_name or "",
            image_path=chunk.image_path or "",
            description=chunk.chart_description or "",
            text=chunk.text,
        )
    return TextChunkEntry(**base, chunk_type=chunk.chunk_type, text=chunk.text)


def _discover_docs(docs_dir: str):
    paths = []
    for ext in ("*.pdf", "*.docx", "*.txt"):
        paths.extend(glob.glob(os.path.join(docs_dir, "**", ext), recursive=True))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest documents and run eval queries.")
    parser.add_argument("--docs",    required=True, help="Directory of documents to ingest")
    parser.add_argument("--queries", required=True, help="JSON file with list of {query_id, question}")
    parser.add_argument("--stream",  action="store_true", help="Stream LLM output to stdout")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = json.load(f)

    doc_files = _discover_docs(args.docs)
    if not doc_files:
        print(f"No supported documents found in: {args.docs}")
        return

    # --- Create run ---
    run_id = eval_store.create_run(
        queries_file=os.path.abspath(args.queries),
        docs_ingested=[os.path.basename(p) for p in doc_files],
        answer_model=settings.ANSWER_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
    )
    print(f"[eval] {run_id} — {len(doc_files)} doc(s), {len(queries)} query/queries")

    chroma_dir = eval_store.chroma_dir_for_run(run_id)
    images_dir = eval_store.images_dir_for_run(run_id)

    # --- Ingest ---
    print(f"[eval] Ingesting...")
    final_chunks, doc_lengths = run_ingest(
        file_paths=doc_files,
        thread_id=run_id,
        message_id="ingest_batch",
        chunk_start_idx=eval_store.peek_next_chunk_idx(run_id),
        chroma_dir=chroma_dir,
        images_dir=images_dir,
    )
    eval_store.append_chunk_metrics(run_id, final_chunks, doc_lengths)
    total_chunks = len(final_chunks)
    print(f"[eval] Ingested {total_chunks} chunk(s)")

    # --- Queries ---
    for q in queries:
        question = q["question"]
        query_id = eval_store.next_query_id(run_id)
        print(f"\n[eval] {query_id}: {question}")

        retrieved = []
        if args.stream:
            token_gen, retrieved = run_query(
                question=question,
                thread_id=run_id,
                stream=True,
                chroma_dir=chroma_dir,
                total_chunks=total_chunks,
            )
            answer = ""
            for token in token_gen:
                print(token, end="", flush=True)
                answer += token
            print()
        else:
            answer, retrieved = run_query(
                question=question,
                thread_id=run_id,
                stream=False,
                chroma_dir=chroma_dir,
                total_chunks=total_chunks,
            )
            print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        used = parse_used_chunks(answer, retrieved)
        now = _now()

        eval_store.append_query_result(run_id, query_id, question, answer, now)
        eval_store.append_chunk_entries(run_id, query_id, "retrieved_chunks.json",
            [_to_chunk_entry(c, query_id, run_id, now) for c in retrieved])
        eval_store.append_chunk_entries(run_id, query_id, "used_chunks.json",
            [_to_chunk_entry(c, query_id, run_id, now) for c in used])
        eval_store.append_retrieval_metrics(run_id, query_id, retrieved)
        eval_store.append_usage_metrics(run_id, query_id, retrieved, used)

    print(f"\n[eval] Done. Results in eval_runs/{run_id}/")


if __name__ == "__main__":
    main()
