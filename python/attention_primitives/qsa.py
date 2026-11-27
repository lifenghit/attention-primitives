"""Dependency-free FP32 reference implementation of Qwen sparse attention."""

import math

from .sparse_common import stable_topk


def _shape3(value, name):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(head, (list, tuple)) for head in value
    ):
        raise ValueError(f"{name} must have shape [T, H, D]")
    heads = len(value[0]) if value else 0
    dim = len(value[0][0]) if heads else 0
    if dim == 0 or any(
        len(head_rows) != heads
        or any(len(vector) != dim for vector in head_rows)
        for head_rows in value
    ):
        raise ValueError(f"{name} must be a non-empty rectangular [T, H, D] tensor")
    return len(value), heads, dim


def visible_blocks(position, sequence_length, compress_ratio):
    if compress_ratio <= 0 or position < 0 or sequence_length < 0:
        raise ValueError("position and sequence_length must be non-negative and ratio positive")
    if position >= sequence_length:
        raise ValueError("position must be smaller than sequence_length")
    return min((position + 1) // compress_ratio, sequence_length // compress_ratio)


def compress_groups(values, compress_ratio):
    """Mean-pool complete groups from [T, H, D], excluding the incomplete tail."""
    length, heads, dim = _shape3(values, "values")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    blocks = length // compress_ratio
    return [
        [
            [
                math.fsum(
                    float(values[block * compress_ratio + offset][head][axis])
                    for offset in range(compress_ratio)
                )
                / compress_ratio
                for axis in range(dim)
            ]
            for head in range(heads)
        ]
        for block in range(blocks)
    ]


def qsa_scores(index_q, compressed_k, compress_ratio, score_divisor=None,
               block_table=None, physical_block_count=None):
    """Return shared [T, C] index scores and their visible-block counts."""
    length, index_heads, dim = _shape3(index_q, "index_q")
    cache_blocks, key_heads, key_dim = _shape3(compressed_k, "compressed_k")
    if key_heads != 1 or key_dim != dim:
        raise ValueError("compressed_k must have one head and the index head dimension")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    divisor = math.sqrt(dim) if score_divisor is None else float(score_divisor)
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("score_divisor must be finite and positive")
    block_count = cache_blocks if block_table is None else len(block_table)
    if block_table is not None:
        if physical_block_count is None or physical_block_count <= 0:
            raise ValueError("physical_block_count must be positive with a block table")
        if physical_block_count > cache_blocks:
            raise ValueError("physical cache is smaller than physical_block_count")

    scores = []
    visibility = []
    for position in range(length):
        visible = visible_blocks(position, length, compress_ratio)
        if visible > block_count:
            raise ValueError("compressed_k does not contain all complete groups")
        visibility.append(visible)
        row = [-math.inf] * block_count
        for logical in range(visible):
            physical = logical if block_table is None else block_table[logical]
            if physical < 0 or physical >= cache_blocks:
                raise ValueError("physical page is out of bounds")
            key = compressed_k[physical][0]
            score = math.fsum(
                max(math.fsum(
                    float(index_q[position][head][axis]) * float(key[axis])
                    for axis in range(dim)
                ), 0.0)
                for head in range(index_heads)
            )
            row[logical] = score / divisor
        scores.append(row)
    return scores, visibility


stable_top_k = stable_topk


def qsa_reference(q, k, v, index_q, index_k=None, compress_ratio=4, top_k=1,
                  score_divisor=None, query_scale=None, block_table=None,
                  physical_block_count=None, compressed_k=None, compressed_v=None,
                  return_metadata=False):
    """Run QSA on one sequence using sequence-major [T, H, D] nested lists.

    Index selection is shared by all main heads. K/V are mean-pooled into complete
    groups, logical selections are mapped through ``block_table``, and each main
    head performs a stable FP32 softmax over only its selected physical blocks.
    """
    length, heads, dim = _shape3(q, "q")
    k_length, kv_heads, key_dim = _shape3(k, "k")
    v_length, value_heads, value_dim = _shape3(v, "v")
    if k_length != length or v_length != length or kv_heads != value_heads:
        raise ValueError("q, k and v must have matching lengths and KV heads")
    if key_dim != dim or heads % kv_heads:
        raise ValueError("main head dimensions or GQA mapping are invalid")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scale = 1.0 / math.sqrt(dim) if query_scale is None else float(query_scale)
    if not math.isfinite(scale):
        raise ValueError("query_scale must be finite")

    pooled_k = compress_groups(k, compress_ratio) if compressed_k is None else compressed_k
    pooled_v = compress_groups(v, compress_ratio) if compressed_v is None else compressed_v
    if compressed_k is not None and index_k is not None:
        raise ValueError("pass either index_k or compressed_k, not both")
    index_values = k if index_k is None else index_k
    index_pooled = (compress_groups(index_values, compress_ratio)
                    if compressed_k is None else compressed_k)
    scores, visibility = qsa_scores(
        index_q, index_pooled, compress_ratio, score_divisor,
        block_table, physical_block_count,
    )
    logical_blocks = length // compress_ratio
    physical_blocks = logical_blocks if block_table is None else physical_block_count
    if len(pooled_k) < physical_blocks or len(pooled_v) < physical_blocks:
        raise ValueError("physical compressed K/V cache is too small")
    if any(len(pooled_v[block]) != kv_heads or
           any(len(vector) != value_dim for vector in pooled_v[block])
           for block in range(physical_blocks)):
        raise ValueError("compressed_v has incompatible heads or dimensions")

    group_size = heads // kv_heads
    output = []
    all_selected = []
    all_physical = []
    all_counts = []
    for position in range(length):
        selected, count = stable_top_k(scores[position], top_k, visibility[position])
        physical = [-1] * top_k
        for slot in range(count):
            logical = selected[slot]
            physical[slot] = logical if block_table is None else block_table[logical]
        all_selected.append(selected)
        all_physical.append(physical)
        all_counts.append(count)
        head_outputs = []
        for head in range(heads):
            if count == 0:
                head_outputs.append([0.0] * value_dim)
                continue
            kv_head = head // group_size
            logits = [
                scale * math.fsum(
                    float(q[position][head][axis]) *
                    float(pooled_k[physical[slot]][kv_head][axis])
                    for axis in range(dim)
                )
                for slot in range(count)
            ]
            maximum = max(logits)
            weights = [math.exp(logit - maximum) for logit in logits]
            denominator = math.fsum(weights)
            head_outputs.append([
                math.fsum(
                    weights[slot] * float(pooled_v[physical[slot]][kv_head][axis])
                    for slot in range(count)
                ) / denominator
                for axis in range(value_dim)
            ])
        output.append(head_outputs)
    if return_metadata:
        return output, {
            "scores": scores,
            "visible_blocks": visibility,
            "selected_indices": all_selected,
            "physical_blocks": all_physical,
            "selected_counts": all_counts,
        }
    return output


qsa = qsa_reference


__all__ = [
    "compress_groups",
    "qsa",
    "qsa_reference",
    "qsa_scores",
    "stable_top_k",
    "visible_blocks",
]
