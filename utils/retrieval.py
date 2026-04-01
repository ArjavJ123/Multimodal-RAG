# utils/retrieval.py — Retrieval utilities.

import settings


def dynamic_top_k(total_chunks: int) -> int:
    """
    Scale TOP_K with the number of indexed chunks.

    Formula: clamp(total_chunks // CHUNKS_PER_RESULT, MIN_TOP_K, MAX_TOP_K)

    Examples:
        200  chunks → 5  (floor)
        655  chunks → 6  (Apple 10-K)
        1611 chunks → 16 (WEO 2025)
        2000 chunks → 20 (cap)
    """
    return max(settings.MIN_TOP_K, min(settings.MAX_TOP_K, total_chunks // settings.CHUNKS_PER_RESULT))
