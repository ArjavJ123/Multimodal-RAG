# query/run.py — Callable query entry point used by app.py and eval.py.

from typing import Generator, List, Optional, Tuple, Union

from langchain_core.messages import BaseMessage

from query.graph import build_query_graph
from utils.dataclasses import DocumentChunk


def run_query(
    question: str,
    thread_id: str,
    chat_history: Optional[List[BaseMessage]] = None,
    stream: bool = False,
    chroma_dir: Optional[str] = None,
    total_chunks: Optional[int] = None,
) -> Union[Tuple[str, List[DocumentChunk]], Tuple[Generator, List[DocumentChunk]]]:
    """
    Run a single query through the query LangGraph pipeline.

    Args:
        question:      The user question.
        thread_id:     Thread or run identifier (used for chroma path when chroma_dir
                       is not supplied, and for dynamic-k lookup).
        chat_history:  LangChain message history for the conversation.
        stream:        If True, return a token generator instead of a full answer string.
                       The retrieved list is populated as a side effect during iteration.
        chroma_dir:    Override chroma persist directory (eval mode). None → default.
        total_chunks:  Pre-computed chunk count for dynamic k (eval mode). None → look up.

    Returns:
        stream=False: (answer: str, retrieved: List[DocumentChunk])
        stream=True:  (token_gen: Generator[str], retrieved: List[DocumentChunk])
                      retrieved is populated after the generator is fully consumed.
    """
    state = {
        "question": question,
        "thread_id": thread_id,
        "chat_history": chat_history or [],
    }
    if chroma_dir is not None:
        state["chroma_dir"] = chroma_dir
    if total_chunks is not None:
        state["total_chunks"] = total_chunks

    graph = build_query_graph()

    if not stream:
        result = graph.invoke(state)
        return result["answer"], result["retrieved_chunks"]

    retrieved: List[DocumentChunk] = []

    def _token_gen() -> Generator[str, None, None]:
        for mode, data in graph.stream(state, stream_mode=["messages", "updates"], version="v2"):
            if mode == "updates":
                for node_name, update in data.items():
                    if node_name == "retrieve":
                        retrieved.extend(update.get("retrieved_chunks", []))
            elif mode == "messages":
                msg, metadata = data
                if msg.content and metadata.get("langgraph_node") == "generate_answer":
                    yield msg.content

    return _token_gen(), retrieved
