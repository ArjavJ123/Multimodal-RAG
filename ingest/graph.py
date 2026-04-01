# ingest/graph.py — Builds and compiles the ingest LangGraph.

from langgraph.graph import StateGraph, END

from utils.dataclasses import IngestState
from ingest.nodes import (
    load_documents,
    caption_visuals,
    chunk_and_embed,
    has_images,
)


def build_ingest_graph():
    """
    Construct the ingest pipeline as a LangGraph StateGraph.

    Graph topology:
        load_documents          ← dispatches to per-type ingestor (text + tables + images)
              │
        ┌─────┴──────────────────────┐
        │ (has_images conditional)   │
        │                            │
    caption_visuals          chunk_and_embed ← (no images)
        │                            │
        └──────► chunk_and_embed ────┘
                      │
                     END
    """
    graph = StateGraph(IngestState)

    graph.add_node("load_documents", load_documents)
    graph.add_node("caption_visuals", caption_visuals)
    graph.add_node("chunk_and_embed", chunk_and_embed)

    graph.set_entry_point("load_documents")

    graph.add_conditional_edges(
        "load_documents",
        has_images,
        {
            "caption_visuals": "caption_visuals",
            "chunk_and_embed": "chunk_and_embed",
        },
    )

    graph.add_edge("caption_visuals", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", END)

    return graph.compile()
