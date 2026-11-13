"""Pool-overflow rule, shared by the environment and the exact solver.

When a pool cap is set (small instance: cap 2) and an arrival would exceed it, one candidate is
dropped automatically. The policy has no say. Both env.py and exact_dp.py call this one function,
so they cannot disagree.

Rule: drop the candidate with the lowest naive estimate (the newcomer is included in the
comparison). Ties: the candidate who has waited longest leaves. Remaining ties: lowest tiebreak
key (candidates that tie on all three are in the same state, so the choice does not matter).
"""
from __future__ import annotations

from typing import Optional, Sequence


def overflow_drop_index(naive_means: Sequence[float], days_waited: Sequence[int],
                        tiebreak: Optional[Sequence[int]] = None) -> int:
    """Index of the candidate to drop."""
    n = len(naive_means)
    assert n == len(days_waited) and n > 0
    tb = tiebreak if tiebreak is not None else [0] * n
    return min(range(n), key=lambda i: (naive_means[i], -days_waited[i], tb[i], i))
