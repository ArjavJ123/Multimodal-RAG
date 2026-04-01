# utils/memory.py — Token-based conversation history trimming.

from typing import List, Union

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import settings


def _count_tokens(messages: List[BaseMessage], model: str) -> int:
    """Count total tokens across a list of LangChain messages using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback for unknown models

    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role, separators) — OpenAI convention
        total += 4 + len(enc.encode(str(msg.content)))
    return total


def trim_history(
    history: List[BaseMessage],
    max_tokens: int = settings.MAX_HISTORY_TOKENS,
    model: str = settings.ANSWER_MODEL,
) -> List[BaseMessage]:
    """
    Trim conversation history to fit within max_tokens by dropping the oldest
    Human/AI pairs first. Always keeps complete pairs to avoid orphaned messages.

    Args:
        history:    Alternating list of HumanMessage / AIMessage objects.
        max_tokens: Token budget for the history block.
        model:      Model name used to select the correct tiktoken encoding.

    Returns:
        Trimmed list of messages that fits within max_tokens.
    """
    if _count_tokens(history, model) <= max_tokens:
        return history

    # Work on pairs from oldest to newest; drop pairs until we fit
    # Pair up messages: [(human, ai), (human, ai), ...]
    pairs = []
    i = 0
    while i + 1 < len(history):
        pairs.append((history[i], history[i + 1]))
        i += 2

    # If there's an unpaired trailing message keep it
    tail = [history[-1]] if len(history) % 2 != 0 else []

    # Drop oldest pairs until under budget
    while pairs and _count_tokens(
        [m for pair in pairs for m in pair] + tail, model
    ) > max_tokens:
        pairs.pop(0)

    trimmed = [m for pair in pairs for m in pair] + tail
    dropped = (len(history) - len(trimmed)) // 2
    if dropped:
        print(f"[memory] Trimmed {dropped} oldest turn(s) to stay within {max_tokens} tokens.")

    return trimmed
