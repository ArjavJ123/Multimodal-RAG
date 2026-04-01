# query/graph.py — Builds and compiles the query LangGraph.

from langgraph.graph import StateGraph, END

from utils.dataclasses import QueryState
from query.nodes import receive_question, retrieve, generate_answer


def build_query_graph():
    """
    Construct the query pipeline as a LangGraph StateGraph.

    Graph topology:
        receive_question
              │
           retrieve
              │
        generate_answer
              │
             END

    Streaming: call graph.stream(..., stream_mode=["messages", "updates"], version="v2")
    to receive LLM tokens from generate_answer and state updates (retrieved_chunks) from retrieve.
    """
    graph = StateGraph(QueryState)

    graph.add_node("receive_question", receive_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
