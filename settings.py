# settings.py — Central config. Import from any node to avoid magic numbers.

# --- Chunking ---
CHUNK_SIZE = 500          # Target chunk size in tokens
CHUNK_OVERLAP = 50        # Overlap between consecutive chunks in tokens

# --- Retrieval ---
TOP_K = 5                 # Fallback when no chunks are indexed yet
MIN_TOP_K = 5             # Floor — always retrieve at least this many chunks
MAX_TOP_K = 20            # Cap — avoid bloating the context window
CHUNKS_PER_RESULT = 100   # Retrieve 1 chunk per N indexed (controls scaling rate)

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-large"   
VISION_MODEL = "gpt-5.4-mini"               
ANSWER_MODEL = "gpt-5.4-mini"               
# --- ChromaDB ---
# Each thread gets its own ChromaDB inside its folder: chats/thread_001/chroma/
CHROMA_COLLECTION_NAME = "rag_docs"

def chroma_path(thread_id: str) -> str:
    """Return the absolute ChromaDB persist path for a given thread."""
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chats")
    return os.path.join(base, thread_id, "chroma")

# --- Memory ---
MAX_HISTORY_TOKENS = 4000   # Max tokens reserved for conversation history

# --- Vision captioning prompt ---
CAPTION_PROMPT = (
    "You are analyzing a chart or image from a business document. "
    "Provide a short descriptive name for the chart and a detailed description covering: "
    "(1) chart type, (2) axis labels and units, "
    "(3) key data values or ranges, and (4) the overall trend or insight shown. "
    "Be concise but precise."
)

# --- System prompt (static, injected once at the start of every conversation) ---
SYSTEM_PROMPT = """\
You are an AI assistant helping company leadership answer questions about internal documents.

Rules:
- Answer ONLY from the context provided in each message.
- For text chunks, cite as: (`filename.pdf`, p.12).
- For chart chunks, you MUST use the exact chart name from the "Chart:" label in the context, followed by the document name, like this: (`Quarterly Revenue Bar Chart 2023-2024`, `annual_report_2024.pdf`). Never cite a chart using only the document name.
- If the context does not contain enough information, respond with: "I don't have enough information in the provided documents to answer this question."
- Be concise and executive-appropriate.
- Use bullet points only when comparing multiple items or listing more than three distinct facts. Otherwise answer in prose."""

# --- User prompt template (dynamic, built per turn) ---
# {context} and {question} are filled in at runtime
USER_PROMPT_TEMPLATE = """\
--- CONTEXT ---
{context}
--- END CONTEXT ---

{question}"""
