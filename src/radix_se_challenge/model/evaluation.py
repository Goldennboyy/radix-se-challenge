"""Evaluation metrics."""

from __future__ import annotations


def apk(actual: list[str], predicted: list[str], k: int = 5) -> float:
    """Computes the average precision at k."""
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 1.0

    return score / min(len(actual), k)
