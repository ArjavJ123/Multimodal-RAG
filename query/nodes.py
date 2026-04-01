# query/nodes.py — One function per node in the query LangGraph.

import os
from typing import Dict, Any, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from utils.dataclasses import QueryState, DocumentChunk
from utils.memory import trim_history
from utils.chat_store import get_total_chunk_count
from utils.retrieval import dynamic_top_k
import settings


# ---------------------------------------------------------------------------
# Node 1: receive_question
# ---------------------------------------------------------------------------

def receive_question(state: QueryState) -> Dict[str, Any]:
    """
    Entry point for a user query. Validates and normalises the raw question
    string before it flows into retrieval.

    Returns:
        {} — question is already in state; nothing to update yet
    """
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    print(f"[receive_question] Received: {question}")
    return {"question": question}


# ---------------------------------------------------------------------------
# Node 2: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: QueryState) -> Dict[str, Any]:
    """
    Embed the question and run cosine similarity search against the thread's ChromaDB.
    k scales dynamically with total indexed chunks via dynamic_top_k().

    Returns:
        {"retrieved_chunks": List[DocumentChunk]}
    """
    chroma_dir = state.get("chroma_dir") or settings.chroma_path(state["thread_id"])
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    total_chunks = state.get("total_chunks") or get_total_chunk_count(state["thread_id"])
    k = dynamic_top_k(total_chunks) if total_chunks else settings.TOP_K
    print(f"[retrieve] total_chunks={total_chunks}, k={k}")

    results = vectorstore.similarity_search(
        query=state["question"],
        k=k,
    )

    retrieved_chunks: List[DocumentChunk] = [
        DocumentChunk(
            chunk_id=doc.metadata.get("chunk_id", ""),
            text=doc.page_content,
            source_file=os.path.basename(doc.metadata.get("source", "")),
            page_number=doc.metadata.get("page", -1),
            chunk_type=doc.metadata.get("chunk_type", "text"),
            chart_name=doc.metadata.get("chart_name"),
            chart_description=doc.metadata.get("chart_description"),
            image_path=doc.metadata.get("image_path"),
        )
        for doc in results
    ]

    print(f"[retrieve] Found {len(retrieved_chunks)} chunk(s) for query: '{state['question']}'")
    return {"retrieved_chunks": retrieved_chunks}


# ---------------------------------------------------------------------------
# Node 3: generate_answer
# ---------------------------------------------------------------------------

def generate_answer(state: QueryState) -> Dict[str, Any]:
    """
    Build a grounded prompt and call the LLM. LangGraph's stream_mode="messages"
    automatically streams tokens from this node's LLM call when the graph is
    invoked via graph.stream() — no changes needed here for streaming to work.

    Message structure:
        SystemMessage                    ← role + citation rules (static)
        [HumanMessage, AIMessage, ...]   ← trimmed conversation history
        HumanMessage                     ← retrieved context + current question

    Returns:
        {"answer": str}
    """
    def _source_line(c: DocumentChunk) -> str:
        if c.chunk_type == "chart_caption" and c.chart_name:
            return f"Chart: {c.chart_name} | Document: {os.path.basename(c.source_file)}"
        return f"Source: {os.path.basename(c.source_file)} | Page: {c.page_number}"

    context = "\n\n".join(
        f"{_source_line(c)}\n{c.text}"
        for c in state["retrieved_chunks"]
    )

    history = trim_history(state.get("chat_history") or [])

    messages = (
        [SystemMessage(content=settings.SYSTEM_PROMPT)]
        + history
        + [HumanMessage(content=settings.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=state["question"],
        ))]
    )

    llm = ChatOpenAI(model=settings.ANSWER_MODEL)
    response = llm.invoke(messages)
    answer = response.content.strip()

    print(f"[generate_answer] Answer generated ({len(answer)} chars)")
    return {"answer": answer}
