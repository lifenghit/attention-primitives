"""Shared deterministic sparse-selection primitives used by DSA/MSA/QSA."""


def stable_topk(scores, topk, count=None):
    """Select descending scores with ascending-index tie breaking and -1 padding."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    count = len(scores) if count is None else count
    if count < 0 or count > len(scores):
        raise ValueError("count is outside the score row")
    order = sorted(range(count), key=lambda index: (-scores[index], index))
    selected = order[:topk]
    return selected + [-1] * (topk - len(selected)), len(selected)


def select_with_local_block(scores, topk, local_block):
    """Apply stable block top-k while retaining the query's current block."""
    selected, count = stable_topk(scores, topk)
    selected = [index for index in selected if index >= 0]
    if local_block not in selected:
        if len(selected) < topk:
            selected.append(local_block)
        else:
            selected[-1] = local_block
    return selected


__all__ = ["select_with_local_block", "stable_topk"]
