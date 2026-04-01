# app.py — Streamlit UI for the AI Leadership Insight Agent.
# Run with: streamlit run app.py

import os
import shutil
import tempfile
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import settings
from ingest.run import run_ingest
from query.run import run_query
from utils.chat_store import (
    CHATS_DIR,
    append_chunk_entries,
    append_chunk_metrics,
    append_message_entry,
    append_retrieval_metrics,
    append_usage_metrics,
    create_thread,
    get_all_message_ids,
    get_lc_history,
    get_messages_for_display,
    get_uploaded_docs,
    peek_next_chunk_idx,
    peek_next_message_id,
)
from utils.citations import parse_used_chunks
from utils.dataclasses import ChartChunkEntry, MessageEntry, TextChunkEntry

load_dotenv()

_DOCS_PANEL_OPEN_W  = 300
_DOCS_PANEL_CLOSED_W = 36


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_chunk_entry(chunk, message_id: str, thread_id: str):
    base = dict(
        chunk_id=chunk.chunk_id,
        message_id=message_id,
        thread_id=thread_id,
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        timestamp=_now(),
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


# ---------------------------------------------------------------------------
# On every page reload — wipe ChromaDB and chats
# ---------------------------------------------------------------------------

if "db_initialised" not in st.session_state:
    # Clear chromadb's internal connection registry before deleting files.
    # Streamlit reruns in the same Python process, so stale chromadb connections
    # survive reloads. Deleting the sqlite3 file while a connection is open causes
    # SQLITE_READONLY_DBMOVED (code 1032) on the next write to the same path.
    try:
        import chromadb.api.client as _chroma_client
        _chroma_client.SharedSystemClient._identifer_to_system.clear()
    except Exception:
        pass

    if os.path.exists(CHATS_DIR):
        shutil.rmtree(CHATS_DIR)
    os.makedirs(CHATS_DIR, exist_ok=True)
    st.session_state.db_initialised = True

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Leadership Insight Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    first_thread = create_thread()
    st.session_state.thread_id = first_thread
    st.session_state.threads = [first_thread]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_per_thread" not in st.session_state:
    st.session_state.ingested_per_thread = {}

if "docs_panel_open" not in st.session_state:
    st.session_state.docs_panel_open = True

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# ---------------------------------------------------------------------------
# Dynamic CSS
# ---------------------------------------------------------------------------

panel_w = _DOCS_PANEL_OPEN_W if st.session_state.docs_panel_open else _DOCS_PANEL_CLOSED_W

st.markdown(f"""
<style>
/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    min-width: 220px !important;
    max-width: 260px !important;
}}

/* Base: all sidebar buttons — flat, left-aligned, no border */
[data-testid="stSidebar"] button {{
    text-align: left !important;
    justify-content: flex-start !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
    padding: 0.4rem 0.75rem !important;
    width: 100% !important;
    background: transparent !important;
    color: rgba(250,250,250,0.65) !important;
}}
[data-testid="stSidebar"] button:hover {{
    background: rgba(250,250,250,0.07) !important;
    color: rgba(250,250,250,0.95) !important;
}}

/* "+ New chat" — outlined, centred */
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button {{
    border: 1px solid rgba(250,250,250,0.25) !important;
    background: transparent !important;
    color: rgba(250,250,250,0.85) !important;
    justify-content: center !important;
    text-align: center !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button:hover {{
    border-color: rgba(250,250,250,0.45) !important;
    background: rgba(250,250,250,0.05) !important;
}}

/* ── Fixed right documents panel ─────────────────────────────────────────── */
[data-testid="stColumn"]:last-child {{
    position: fixed !important;
    right: 0 !important;
    top: 0 !important;
    bottom: 0 !important;
    width: {panel_w}px !important;
    min-width: {panel_w}px !important;
    max-width: {panel_w}px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background-color: #0e1117 !important;
    border-left: 1px solid rgba(250,250,250,0.08) !important;
    padding: 0.75rem 0.6rem 4rem !important;
    z-index: 999 !important;
}}

[data-testid="stColumn"]:first-child {{
    padding-right: {panel_w + 12}px !important;
}}

/* Toggle arrow */
[data-testid="stColumn"]:last-child div[data-testid="stButton"]:first-child > button {{
    padding: 0.2rem 0.5rem !important;
    min-height: unset !important;
    font-size: 1rem !important;
    border: 1px solid rgba(250,250,250,0.15) !important;
    color: rgba(250,250,250,0.6) !important;
    border-radius: 5px !important;
    width: auto !important;
    justify-content: center !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Thread Management (LHS)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 💬 Chats")

    if st.button("＋  New chat", use_container_width=True):
        new_tid = create_thread()
        st.session_state.threads.append(new_tid)
        st.session_state.thread_id = new_tid
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p style="font-size:0.78rem;color:rgba(250,250,250,0.45);'
        'margin:0.9rem 0 0.3rem;font-weight:600;letter-spacing:0.03em">'
        'Select session</p>',
        unsafe_allow_html=True,
    )

    for tid in reversed(st.session_state.threads):
        is_active = tid == st.session_state.thread_id
        if st.button(tid, key=f"thr_{tid}", use_container_width=True):
            if not is_active:
                st.session_state.thread_id = tid
                st.session_state.messages = get_messages_for_display(tid)
                st.rerun()

# JS: apply blue border to the active thread button by matching its text content.
# Runs after Streamlit renders the sidebar (300 ms delay).
_active = st.session_state.thread_id
components.html(f"""
<script>
setTimeout(function() {{
    var btns = window.parent.document.querySelectorAll('[data-testid="stSidebar"] button');
    btns.forEach(function(b) {{
        // reset any previous highlight
        b.style.outline = '';
        b.style.outlineOffset = '';
        b.style.fontWeight = '';
        b.style.color = '';
        if (b.innerText.trim() === '{_active}') {{
            b.style.outline = '2px solid #3b82f6';
            b.style.outlineOffset = '-2px';
            b.style.fontWeight = '700';
            b.style.color = '#ffffff';
        }}
    }});
}}, 300);
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

chat_col, docs_col = st.columns([4, 1], gap="small")

# ---------------------------------------------------------------------------
# Right column — collapsible Documents Panel
# ---------------------------------------------------------------------------

with docs_col:
    arrow = "›" if not st.session_state.docs_panel_open else "‹"
    if st.button(arrow, key="docs_toggle"):
        st.session_state.docs_panel_open = not st.session_state.docs_panel_open
        st.rerun()

    if st.session_state.docs_panel_open:
        st.markdown("### Documents")

        thread_id = st.session_state.thread_id
        ingested_names: set = st.session_state.ingested_per_thread.get(thread_id, set())

        if st.button("＋  Add documents", use_container_width=True, key="add_docs_btn"):
            st.session_state.show_uploader = not st.session_state.show_uploader

        if st.session_state.show_uploader:
            uploaded_files = st.file_uploader(
                "Upload",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key=f"uploader_{thread_id}",
                label_visibility="collapsed",
            )
            if uploaded_files:
                to_ingest = [f for f in uploaded_files if f.name not in ingested_names]
                if to_ingest:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        file_paths = []
                        for uf in to_ingest:
                            dest = os.path.join(tmp_dir, uf.name)
                            with open(dest, "wb") as fh:
                                fh.write(uf.getbuffer())
                            file_paths.append(dest)

                        with st.spinner(f"Ingesting {len(to_ingest)} file(s)…"):
                            try:
                                message_id = peek_next_message_id(thread_id)
                                chunk_start_idx = peek_next_chunk_idx(thread_id)
                                final_chunks, doc_lengths = run_ingest(
                                    file_paths=file_paths,
                                    thread_id=thread_id,
                                    message_id=message_id,
                                    chunk_start_idx=chunk_start_idx,
                                )
                                entries = [_to_chunk_entry(c, message_id, thread_id) for c in final_chunks]
                                append_chunk_entries(thread_id, "uploaded_chunks.json", entries)
                                append_chunk_metrics(thread_id, message_id, final_chunks, doc_lengths)

                                if thread_id not in st.session_state.ingested_per_thread:
                                    st.session_state.ingested_per_thread[thread_id] = set()
                                st.session_state.ingested_per_thread[thread_id].update(
                                    f.name for f in to_ingest
                                )
                                st.session_state.show_uploader = False
                                st.success(f"{len(final_chunks)} chunk(s) indexed.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Ingestion failed: {e}")

        # Docs accordion
        docs = get_uploaded_docs(thread_id)
        if docs:
            st.divider()
            for doc_name, chunks in docs.items():
                with st.expander(f"📄 {doc_name}  ({len(chunks)})"):
                    for chunk in chunks:
                        ctype = chunk.get("chunk_type", "text")
                        cid   = chunk.get("chunk_id", "")
                        pg    = chunk.get("page_number", "?")
                        icon  = "📊" if ctype == "chart_caption" else ("📋" if ctype == "table" else "📝")

                        # Header: icon + id + page + type badge
                        st.markdown(
                            f"{icon} **{cid}** — p.{pg} "
                            f"<span style='font-size:0.7rem;color:rgba(250,250,250,0.4);"
                            f"background:rgba(250,250,250,0.08);padding:1px 5px;"
                            f"border-radius:4px'>{ctype}</span>",
                            unsafe_allow_html=True,
                        )

                        # Inline image toggle for chart chunks
                        if ctype == "chart_caption":
                            image_path = chunk.get("image_path", "")
                            chart_name = chunk.get("chart_name") or "Chart"
                            img_key = f"show_img_{cid}"
                            expanded = st.session_state.get(img_key, False)
                            btn_label = f"{'▼' if expanded else '▶'} 🖼 {chart_name}"
                            if st.button(btn_label, key=f"imgbtn_{cid}", use_container_width=True):
                                st.session_state[img_key] = not expanded
                                st.rerun()
                            if expanded and image_path and os.path.exists(image_path):
                                st.image(image_path, use_container_width=True)

                        # Text preview
                        text = chunk.get("text", "")
                        st.caption(text[:180] + ("…" if len(text) > 180 else ""))
        else:
            st.caption("No documents ingested yet.")

# ---------------------------------------------------------------------------
# Left column — Chat message history
# ---------------------------------------------------------------------------

with chat_col:
    st.title("📊 Leadership Insight Agent")
    st.caption("Ask questions about your company documents. Answers are grounded and cited.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input — outside columns so it pins to the bottom of the viewport
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about your documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                thread_id = st.session_state.thread_id
                message_id = peek_next_message_id(thread_id)

                token_gen, retrieved = run_query(
                    question=question,
                    thread_id=thread_id,
                    chat_history=get_lc_history(thread_id),
                    stream=True,
                )
                answer = st.write_stream(token_gen)
                answer = answer or "I don't have enough information in the provided documents to answer this question."

                if retrieved:
                    with st.expander(f"Sources ({len(retrieved)} chunk(s) retrieved)"):
                        for i, chunk in enumerate(retrieved, 1):
                            src = os.path.basename(chunk.source_file)
                            label = f"**{i}. {src}** — page {chunk.page_number} `{chunk.chunk_type}`"
                            st.markdown(label)
                            st.caption(chunk.text[:300] + "…")

            except Exception as e:
                answer = f"Error generating answer: {e}"
                retrieved = []
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer.replace("$", r"\$")})

    # Persist everything before rerun
    used = parse_used_chunks(answer, retrieved)

    append_message_entry(thread_id, MessageEntry(
        message_id=message_id,
        thread_id=thread_id,
        question=question,
        answer=answer,
        history_context=get_all_message_ids(thread_id),
        timestamp=_now(),
    ))
    append_chunk_entries(thread_id, "retrieved_chunks.json",
                         [_to_chunk_entry(c, message_id, thread_id) for c in retrieved])
    append_chunk_entries(thread_id, "used_chunks.json",
                         [_to_chunk_entry(c, message_id, thread_id) for c in used])
    append_retrieval_metrics(thread_id, message_id, retrieved)
    append_usage_metrics(thread_id, message_id, retrieved, used)

    # Rerun so all messages render from the loop in the correct order
    st.rerun()
