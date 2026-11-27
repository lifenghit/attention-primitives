"""Dependency-free FP32 reference for gated linear attention."""

import math


def _validate_tokens(name, values, batches, heads, dim=None):
    if len(values) != batches:
        raise ValueError(f"{name} must have shape [B, T, H, D]")
    sequence_length = None
    for batch in values:
        if sequence_length is None:
            sequence_length = len(batch)
        elif len(batch) != sequence_length:
            raise ValueError(f"{name} must have a consistent sequence length")
        for head in batch:
            if len(head) != heads:
                raise ValueError(f"{name} must have shape [B, T, H, D]")
            for vector in head:
                if dim is None:
                    dim = len(vector)
                if len(vector) != dim:
                    raise ValueError(f"{name} must have a consistent feature dimension")
    if dim is None or dim == 0:
        raise ValueError("the feature dimension must be positive")
    return sequence_length, dim


def naive_recurrent_gla(q, k, v, gk, initial_state=None, output_final_state=False):
    """Run GLA with sequence-major [B,T,H,D] inputs and [B,H,K,V] state.

    At each token, S = S * exp(gk) + k outer v and o = (q * K**-0.5) @ S.
    The returned state is None unless output_final_state is true.
    """
    if len(q) == 0:
        raise ValueError("B must be positive")
    batches = len(q)
    heads = len(q[0][0]) if q[0] else None
    if not heads:
        raise ValueError("H and T must be positive")
    sequence_length, key_dim = _validate_tokens("q", q, batches, heads)
    k_length, k_dim = _validate_tokens("k", k, batches, heads, key_dim)
    gk_length, gate_dim = _validate_tokens("gk", gk, batches, heads, key_dim)
    _, value_dim = _validate_tokens("v", v, batches, heads)
    if k_length != sequence_length or gk_length != sequence_length or len(v[0]) != sequence_length:
        raise ValueError("q, k, v and gk must have the same sequence length")
    if k_dim != key_dim or gate_dim != key_dim:
        raise ValueError("q, k and gk must have the same feature dimension")

    if initial_state is None:
        state = [
            [[[0.0 for _ in range(value_dim)] for _ in range(key_dim)] for _ in range(heads)]
            for _ in range(batches)
        ]
    else:
        if len(initial_state) != batches:
            raise ValueError("initial_state must have shape [B, H, K, V]")
        state = []
        for batch in initial_state:
            if len(batch) != heads:
                raise ValueError("initial_state must have shape [B, H, K, V]")
            state_batch = []
            for matrix in batch:
                if len(matrix) != key_dim or any(len(row) != value_dim for row in matrix):
                    raise ValueError("initial_state must have shape [B, H, K, V]")
                state_batch.append([[float(value) for value in row] for row in matrix])
            state.append(state_batch)

    output = [
        [[[0.0 for _ in range(value_dim)] for _ in range(heads)]
         for _ in range(sequence_length)]
        for _ in range(batches)
    ]
    scale = key_dim ** -0.5
    for b in range(batches):
        for t in range(sequence_length):
            for h in range(heads):
                decay = [math.exp(float(value)) for value in gk[b][t][h]]
                for i in range(key_dim):
                    k_value = float(k[b][t][h][i])
                    for j in range(value_dim):
                        state[b][h][i][j] = (
                            state[b][h][i][j] * decay[i]
                            + k_value * float(v[b][t][h][j])
                        )
                for j in range(value_dim):
                    output[b][t][h][j] = sum(
                        float(q[b][t][h][i]) * scale * state[b][h][i][j]
                        for i in range(key_dim)
                    )

    return output, state if output_final_state else None
