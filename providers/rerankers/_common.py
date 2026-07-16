def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    denom = max_val - min_val
    if denom == 0.0:
        return [1.0] * len(scores)
    return [(s - min_val) / denom for s in scores]
