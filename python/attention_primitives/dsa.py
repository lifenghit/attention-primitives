"""Dependency-free FP32 reference for DeepSeek Sparse Attention."""

import math

from .sparse_common import stable_topk


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _validate_indexer(q_idx, k_idx, w_idx):
    if len(q_idx) != len(k_idx):
        raise ValueError("q_idx and k_idx must have the same batch size")
    if w_idx is not None and len(w_idx) != len(q_idx):
        raise ValueError("w_idx must have the same batch size as q_idx")
    for b, (q_b, k_b) in enumerate(zip(q_idx, k_idx)):
        if len(q_b) != len(k_b):
            raise ValueError("q_idx and k_idx must have the same sequence length")
        if w_idx is not None and len(w_idx[b]) != len(q_b):
            raise ValueError("w_idx must have shape [B, T, HI]")
        for t, q_t in enumerate(q_b):
            if not q_t:
                raise ValueError("HI must be positive")
            index_dim = len(q_t[0])
            if index_dim == 0 or any(len(q_h) != index_dim for q_h in q_t):
                raise ValueError("q_idx must have shape [B, T, HI, DI]")
            if w_idx is not None and len(w_idx[b][t]) != len(q_t):
                raise ValueError("w_idx must have shape [B, T, HI]")
        for k_t in k_b:
            if q_b and len(k_t) != len(q_b[0][0]):
                raise ValueError("k_idx must have shape [B, T, DI]")


def dsa_indexer(q_idx, k_idx, w_idx=None, topk=2048, scale=None):
    """Select stable causal token indices, with -1 padding to ``topk``."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    _validate_indexer(q_idx, k_idx, w_idx)
    result = []
    for b, (q_b, k_b) in enumerate(zip(q_idx, k_idx)):
        batch_indices = []
        for t, q_t in enumerate(q_b):
            index_scale = scale
            if index_scale is None:
                index_scale = len(q_t[0]) ** -0.5
            scores = []
            for s, k_s in enumerate(k_b):
                if s > t:
                    scores.append(-math.inf)
                    continue
                score = 0.0
                for h, q_h in enumerate(q_t):
                    activated = max(index_scale * _dot(q_h, k_s), 0.0)
                    weight = 1.0 if w_idx is None else w_idx[b][t][h]
                    score += weight * activated
                scores.append(score)
            selected, _ = stable_topk(scores, topk, count=t + 1)
            batch_indices.append(selected)
        result.append(batch_indices)
    return result


def _validate_attention(q, k, v, indices):
    if not (len(q) == len(k) == len(v) == len(indices)):
        raise ValueError("q, k, v, and indices must have the same batch size")
    for b in range(len(q)):
        if not (len(q[b]) == len(k[b]) == len(v[b]) == len(indices[b])):
            raise ValueError("q, k, v, and indices must have the same sequence length")
        for t in range(len(q[b])):
            query_heads = len(q[b][t])
            if query_heads == 0 or not k[b][t] or len(k[b][t]) != len(v[b][t]):
                raise ValueError("HQ must be positive and K/V heads must match")
            kv_heads = len(k[b][t])
            if query_heads % kv_heads:
                raise ValueError("HQ must be divisible by H")
            head_dim = len(q[b][t][0])
            value_dim = len(v[b][t][0])
            if head_dim == 0 or value_dim == 0:
                raise ValueError("K and V dimensions must be positive")
            if any(len(q_h) != head_dim for q_h in q[b][t]):
                raise ValueError("q must have shape [B, T, HQ, K]")
            if any(len(k_h) != head_dim for k_h in k[b][t]):
                raise ValueError("k must have shape [B, T, H, K]")
            if any(len(v_h) != value_dim for v_h in v[b][t]):
                raise ValueError("v must have shape [B, T, H, V]")


def dsa(q, k, v, q_idx=None, k_idx=None, w_idx=None, indices=None,
        topk=2048, scale=None, return_indices=False):
    """Apply GQA FP32 softmax over one selection shared by all query heads."""
    if indices is None:
        if q_idx is None or k_idx is None:
            raise ValueError("q_idx and k_idx are required when indices are omitted")
        indices = dsa_indexer(q_idx, k_idx, w_idx, topk=topk)
    _validate_attention(q, k, v, indices)
    output = []
    for b in range(len(q)):
        output_b = []
        for t, q_t in enumerate(q[b]):
            sequence_length = len(q[b])
            query_heads = len(q_t)
            kv_heads = len(k[b][t])
            group_size = query_heads // kv_heads
            value_dim = len(v[b][t][0])
            attention_scale = scale
            if attention_scale is None:
                attention_scale = len(q_t[0]) ** -0.5

            selected = [False] * sequence_length
            for s in indices[b][t]:
                if s < 0:
                    continue
                if s >= sequence_length:
                    raise ValueError("selected token index is out of bounds")
                if s <= t:
                    selected[s] = True

            output_t = []
            for h, q_h in enumerate(q_t):
                kv_h = h // group_size
                scores = [
                    attention_scale * _dot(q_h, k[b][s][kv_h])
                    if selected[s] else -math.inf
                    for s in range(sequence_length)
                ]
                visible = [s for s in range(t + 1) if selected[s]]
                if not visible:
                    output_t.append([0.0] * value_dim)
                    continue
                maximum = max(scores[s] for s in visible)
                weights = [math.exp(scores[s] - maximum) for s in visible]
                denominator = sum(weights)
                probabilities = [weight / denominator for weight in weights]
                output_t.append([
                    sum(probability * v[b][s][kv_h][d]
                        for probability, s in zip(probabilities, visible))
                    for d in range(value_dim)
                ])
            output_b.append(output_t)
        output.append(output_b)
    return (output, indices) if return_indices else output


naive_dsa_indexer = dsa_indexer
naive_dsa = dsa
