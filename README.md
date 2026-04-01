# Leadership Insight Agent — Multimodal RAG

A Streamlit application and CLI evaluation tool that lets company leadership ask natural language questions about internal documents. Answers are grounded in retrieved content and cited back to the source page.

Supports PDF, DOCX, and TXT files. Extracts plain text, structured tables, and embedded charts — each stored as a distinct chunk type with appropriate metadata.

---

## How It Works

Two LangGraph pipelines — ingest and query — share a single set of callable entry points (`ingest/run.py`, `query/run.py`). Both the Streamlit UI and the CLI evaluation script call these same functions; they differ only in where results are persisted.

```
┌─────────────────────────────────────────────────────────┐
│  app.py (Streamlit UI)        eval.py (CLI)             │
│       │                            │                    │
│  run_ingest()              run_ingest()                 │
│  run_query(stream=True)    run_query(stream=True/False) │
│       │                            │                    │
│  chat_store.py             eval_store.py                │
│  chats/thread_NNN/         eval_runs/run_NNN/           │
└─────────────────────────────────────────────────────────┘
         │                        │
    ingest/run.py            query/run.py
         │                        │
   ingest/graph.py          query/graph.py
         │                        │
   ingest/nodes.py          query/nodes.py
         │                        │
   ChromaDB (per thread/run) ─────┘
```

### Ingest Pipeline

```
File Upload (PDF / DOCX / TXT)
        │
        ▼
  File-Type Ingestor          ← dispatched by ingest/ingestors/base.py
  ├── PDFIngestor              pdfplumber (text + tables) + pymupdf (images)
  ├── DocxIngestor             Docx2txtLoader (text) + python-docx (tables)
  └── TxtIngestor              TextLoader (text only)
        │
        ├── text_docs  ──────► RecursiveCharacterTextSplitter ──► text chunks
        ├── table_chunks ────► kept whole (no splitting)       ──► table chunks
        └── image_crops ────► GPT Vision (caption_visuals)    ──► chart_caption chunks
                │
                ▼
         OpenAI Embeddings (text-embedding-3-large)
                │
                ▼
         ChromaDB (per-thread or per-run, persisted to disk)
```

### Query Pipeline

```
User Question
      │
      ▼
 receive_question   ← validates and normalises input
      │
      ▼
   retrieve         ← embeds question, cosine search in ChromaDB
      │              ← k scales dynamically with number of indexed chunks
      │
      ▼
 generate_answer    ← builds context from retrieved chunks, streams tokens
      │              ← uses conversation history (trimmed to token budget)
      ▼
 Streamed response (UI) or full string (CLI)
```

---

## Document Types Supported

| Format | Text | Tables | Images |
|--------|------|--------|--------|
| PDF    | ✓ pdfplumber | ✓ pdfplumber `extract_tables()` | ✓ pymupdf |
| DOCX   | ✓ docx2txt | ✓ python-docx | — |
| TXT    | ✓ TextLoader | — | — |

---

## Chunk Types

Every piece of content is stored as one of three chunk types.

### `text`
Plain prose extracted page by page. Split by `RecursiveCharacterTextSplitter` at `CHUNK_SIZE=500` tokens with `CHUNK_OVERLAP=50`.

**Cited as:** `` (`filename.pdf`, p.12) ``

### `table`
Structured tables extracted cell-by-cell and formatted as markdown so column headers always stay with their data rows. Stored whole — never split — so the LLM reads the full column context.

**Cited as:** `` (`filename.pdf`, p.12) ``

**Why it matters:** naive character splitting severs table rows from their headers, leaving orphaned numbers with no column context.

### `chart_caption`
Embedded images sent to GPT Vision for structured captioning. Each caption includes chart type, axis labels, key values, and trend insight. The image is saved to `images/` and viewable inline in the UI.

**Cited as:** `` (`Chart Name`, `filename.pdf`) ``

---

## Dynamic TOP_K

`k` scales with the number of indexed chunks rather than using a fixed value:

```
k = clamp(total_chunks // 100, min=5, max=20)
```

| Indexed chunks | k  |
|----------------|----|
| < 500          | 5  |
| 655            | 6  |
| 1611           | 16 |
| 2000+          | 20 |

Configured in `settings.py` via `MIN_TOP_K`, `MAX_TOP_K`, `CHUNKS_PER_RESULT`. Logic lives in `utils/retrieval.py`.

---

## Setup

### 1. Clone and create environment
```bash
git clone https://github.com/ArjavJ123/Multimodal-RAG.git
cd multimodal-rag

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 3. Run the UI
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Each browser reload wipes all threads and starts fresh. If fails then ctrl + C and then restart streamlit 

---

## CLI Evaluation

For bulk ingestion and evaluation without the UI. Results are written to `eval_runs/` and are never wiped — each run is a permanent artifact.

### Queries input file (`eval_queries.json`)
```json
[
  { "query_id": "query_001", "question": "What was revenue in 2024?" },
  { "query_id": "query_002", "question": "Which segment grew fastest?" }
]
```

### Run
```bash
# Without streaming (prints truncated answer per query)
python eval.py --docs ./eval_docs/ --queries eval_queries.json

# With streaming (streams tokens to stdout as they are generated)
python eval.py --docs ./eval_docs/ --queries eval_queries.json --stream
```

A new `run_NNN` is created automatically on each invocation.

### Output structure
```
eval_runs/
└── run_001/
    ├── run_meta.json           # run_id, timestamp, docs, queries file, models
    ├── chunk_metrics.json      # ingest quality metrics
    ├── chroma/                 # vector store for this run
    ├── images/                 # extracted chart images
    └── query_001/
        ├── query_meta.json     # question + full answer
        ├── retrieved_chunks.json
        ├── used_chunks.json
        ├── retrieval_metrics.json
        └── usage_metrics.json
```

---

## Configuration (`settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Target chunk size in tokens |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive text chunks |
| `MIN_TOP_K` | 5 | Minimum chunks retrieved per query |
| `MAX_TOP_K` | 20 | Maximum chunks retrieved per query |
| `CHUNKS_PER_RESULT` | 100 | Controls scaling rate for dynamic TOP_K |
| `MAX_HISTORY_TOKENS` | 4000 | Token budget for conversation history |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `VISION_MODEL` | `gpt-4o-mini` | Model used for chart captioning |
| `ANSWER_MODEL` | `gpt-4o-mini` | Model used for answer generation |

---

## Chats Directory Structure (UI)

Threads are wiped on page reload. Each thread is fully self-contained.

```
chats/
└── thread_001/
    ├── messages.json                    # Full Q&A history for this thread
    ├── images/                          # Extracted chart images (PNG)
    ├── chroma/                          # Per-thread ChromaDB vector store
    ├── message_chunk_metadata/
    │   ├── uploaded_chunks.json         # All chunks created during ingestion
    │   ├── retrieved_chunks.json        # Chunks returned by vector search per message
    │   └── used_chunks.json             # Subset of retrieved chunks cited in the answer
    └── eval_metrics/
        ├── chunk_metrics.json           # Ingestion quality metrics
        ├── retrieval_metrics.json       # Per-query retrieval metrics
        └── usage_metrics.json           # Per-query citation metrics
```

---

## File Formats

### `messages.json`
```json
[
  {
    "message_id": "msg_001",
    "thread_id": "thread_001",
    "question": "What was revenue in Q4?",
    "answer": "Revenue was $1.34B ...",
    "history_context": [],
    "timestamp": "2026-04-01T00:00:00+00:00"
  }
]
```

### `message_chunk_metadata/*.json`
```json
[
  {
    "message_id": "msg_001",
    "docs": [
      {
        "source_file": "annual_report_2024.pdf",
        "chunks": [
          {
            "chunk_id": "chunk_001",
            "page_number": 12,
            "chunk_type": "text | table | chart_caption",
            "text": "...",
            "thread_id": "thread_001",
            "timestamp": "2026-04-01T00:00:00+00:00"
          }
        ]
      }
    ]
  }
]
```

### `eval_metrics/chunk_metrics.json`
```json
[
  {
    "message_id": "msg_001",
    "docs": [
      {
        "source_file": "report.pdf",
        "num_chunks": 1611,
        "num_text_chunks": 1554,
        "num_table_chunks": 43,
        "num_chart_chunks": 14,
        "doc_length_chars": 667403,
        "chunks": [
          { "chunk_id": "chunk_001", "chunk_type": "text", "length_chars": 450 }
        ]
      }
    ]
  }
]
```

### `eval_metrics/retrieval_metrics.json`
```json
[
  {
    "message_id": "msg_001",
    "total_retrieved": 16,
    "num_text_retrieved": 15,
    "num_table_retrieved": 1,
    "num_chart_retrieved": 0,
    "docs": [
      {
        "source_file": "report.pdf",
        "num_chunks_fetched": 16,
        "chunks": [{ "chunk_id": "chunk_220", "rank": 1, "chunk_type": "text" }]
      }
    ]
  }
]
```

### `eval_metrics/usage_metrics.json`
```json
[
  {
    "message_id": "msg_001",
    "total_used": 3,
    "num_text_used": 2,
    "num_table_used": 1,
    "num_chart_used": 0,
    "docs": [
      {
        "source_file": "report.pdf",
        "num_chunks_used": 3,
        "chunks": [{ "chunk_id": "chunk_220", "rank_in_retrieval": 1, "chunk_type": "text" }]
      }
    ]
  }
]
```

### `eval_runs/run_001/run_meta.json`
```json
{
  "run_id": "run_001",
  "timestamp": "2026-04-01T00:00:00+00:00",
  "docs_ingested": ["report.pdf"],
  "queries_file": "/absolute/path/to/eval_queries.json",
  "answer_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-large"
}
```

### `eval_runs/run_001/query_001/query_meta.json`
```json
{
  "query_id": "query_001",
  "run_id": "run_001",
  "question": "What was revenue in 2024?",
  "answer": "Revenue was $4.2B ...",
  "timestamp": "2026-04-01T00:00:00+00:00"
}
```

---

## Using Metrics for Quality Improvement

### 1. Answer is wrong

Check `retrieved_chunks.json` and `used_chunks.json` for the query.

- **Correct chunks were not retrieved** → retrieval issue. The right content exists in `uploaded_chunks.json` but did not appear in `retrieved_chunks.json`. Fix the chunking strategy: chunk size may be too large (diluting the signal), tables may be getting split, or chart captions may not be descriptive enough to match the query embedding.

- **Correct chunks were retrieved but not used** → prompt issue. The chunks appear in `retrieved_chunks.json` but not in `used_chunks.json`. The LLM had the right information but did not use or cite it. Tighten the system prompt or citation rules.

- **Uploaded chunks themselves look wrong** → ingestion issue. Check `uploaded_chunks.json` — if the chunk text is garbled, truncated, or missing key content, the ingestor is not extracting correctly. Fix the PDF/DOCX parser or adjust how tables and images are handled.

### 2. Answer is incomplete

Check whether all relevant chunks were retrieved.

- **Relevant chunks missing from `retrieved_chunks.json`** → `k` is too small. Lower `CHUNKS_PER_RESULT` in `settings.py` to increase `k` for your document size, or raise `MAX_TOP_K`.

- **Relevant chunks were retrieved but the answer skips them** → prompt issue. The chunks appear in `retrieved_chunks.json` but not in `used_chunks.json`. The LLM is ignoring retrieved context. Adjust the system prompt to instruct the model to be more thorough.

### 3. Answer is correct but used chunks had low retrieval rank

Check `rank_in_retrieval` in `usage_metrics.json`. If the cited chunks consistently appear near the bottom of the retrieved list (high rank numbers close to `k`), the most relevant content is being pushed down by less relevant chunks ranked above it. This is a signal to apply **reranking** — after the initial vector search, use a cross-encoder or relevance model to reorder retrieved chunks before passing them to the LLM.

---

## Project Structure

```
├── app.py                        # Streamlit UI entry point
├── eval.py                       # CLI bulk ingest + eval entry point
├── settings.py                   # Central config (models, chunking, retrieval params)
├── eval_docs/                    # Sample documents for CLI eval testing
├── eval_queries.json             # Sample queries for CLI eval testing
├── ingest/
│   ├── graph.py                  # LangGraph ingest pipeline
│   ├── nodes.py                  # load_documents, caption_visuals, chunk_and_embed
│   ├── run.py                    # run_ingest() — shared callable for UI and CLI
│   └── ingestors/
│       ├── base.py               # ingest_file() dispatcher
│       ├── pdf.py                # PDF: text + tables + images
│       ├── docx.py               # DOCX: text + tables
│       └── txt.py                # TXT: text only
├── query/
│   ├── graph.py                  # LangGraph query pipeline
│   ├── nodes.py                  # receive_question, retrieve, generate_answer
│   └── run.py                    # run_query() — shared callable, supports stream=True/False
└── utils/
    ├── dataclasses.py            # All shared data models and TypedDicts
    ├── chat_store.py             # Thread management, JSON persistence, metric writers (UI)
    ├── eval_store.py             # Run management, JSON persistence, metric writers (CLI)
    ├── citations.py              # Citation regex parsing (retrieved → used chunks)
    ├── memory.py                 # Conversation history token trimming
    ├── retrieval.py              # dynamic_top_k()
    └── visuals.py                # GPT Vision captioning
```
