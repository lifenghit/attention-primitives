"""Dependency-free FP32 reference for MiniMax-style sparse attention."""

import math

from .sparse_common import select_with_local_block


def _check_tensor(tensor, shape, name):
    if len(tensor) != shape[0]:
        raise ValueError(f"{name} must have shape {shape}")
    if len(shape) == 1:
        return
    for item in tensor:
        _check_tensor(item, shape[1:], name)


def msa(q, k, v, q_idx, k_idx, block_size=128, topk_blocks=16,
        query_scale=None, index_scale=None, return_indices=False):
    """Evaluate MSA with one independent block selection per GQA group.

    Q and index Q use H heads; K, V, and selections use KVHeads groups. The
    returned selection has shape [T, KVHeads, topk_blocks] and uses -1 padding.
    The current query block replaces the last selected block when necessary.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be positive")
    if not isinstance(topk_blocks, int) or topk_blocks <= 0:
        raise ValueError("topk_blocks must be positive")

    t_len = len(q)
    if t_len == 0:
        return ([], []) if return_indices else []
    heads = len(q[0])
    kv_heads = len(k[0]) if k else 0
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError("QHeads must be a positive integer multiple of KVHeads")
    dim = len(q[0][0])
    v_dim = len(v[0][0]) if v else 0
    index_dim = len(q_idx[0][0]) if q_idx else 0
    if dim <= 0 or v_dim <= 0 or index_dim <= 0:
        raise ValueError("D, V, and DI must be positive")
    _check_tensor(q, (t_len, heads, dim), "Q")
    _check_tensor(k, (t_len, kv_heads, dim), "K")
    _check_tensor(v, (t_len, kv_heads, v_dim), "V")
    _check_tensor(q_idx, (t_len, heads, index_dim), "QIdx")
    _check_tensor(k_idx, (t_len, index_dim), "KIdx")

    group_size = heads // kv_heads
    q_scale = 1.0 / math.sqrt(dim) if query_scale is None else query_scale
    i_scale = 1.0 / math.sqrt(index_dim) if index_scale is None else index_scale
    output = [[[0.0] * v_dim for _ in range(heads)] for _ in range(t_len)]
    all_indices = [[[-1] * topk_blocks for _ in range(kv_heads)]
                   for _ in range(t_len)]

    for t in range(t_len):
        current_block = t // block_size
        for group in range(kv_heads):
            token_scores = []
            for s in range(t + 1):
                score = 0.0
                for head in range(group * group_size, (group + 1) * group_size):
                    score += math.fsum(
                        q_idx[t][head][i] * k_idx[s][i]
                        for i in range(index_dim)
                    ) * i_scale
                token_scores.append(score)

            block_scores = []
            for block in range(current_block + 1):
                begin = block * block_size
                end = min(begin + block_size, t + 1)
                block_scores.append(max(token_scores[begin:end]))
            selected = select_with_local_block(block_scores, topk_blocks, current_block)
            all_indices[t][group][:len(selected)] = selected

            visible = set()
            for block in selected:
                begin = block * block_size
                end = min(begin + block_size, t + 1)
                visible.update(range(begin, end))
            visible = sorted(visible)
            for head in range(group * group_size, (group + 1) * group_size):
                scores = [
                    math.fsum(q[t][head][i] * k[s][group][i]
                              for i in range(dim)) * q_scale
                    for s in visible
                ]
                maximum = max(scores)
                weights = [math.exp(score - maximum) for score in scores]
                denominator = math.fsum(weights)
                for channel in range(v_dim):
                    output[t][head][channel] = math.fsum(
                        weight * v[s][group][channel]
                        for weight, s in zip(weights, visible)
                    ) / denominator

    if return_indices:
        return output, all_indices
    return output
