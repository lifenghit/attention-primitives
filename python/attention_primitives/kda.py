"""FP32 KDA token recurrence using sequence-major nested Python sequences."""

import math
import struct


def _fp32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _shape(tensor):
    shape = []
    value = tensor
    while isinstance(value, (list, tuple)):
        shape.append(len(value))
        if not value:
            break
        value = value[0]
    return tuple(shape)


def _zeros(shape):
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def naive_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
    """Evaluate q/k [B,T,H,K], v [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV]."""
    q_shape = _shape(q)
    if len(q_shape) != 4 or any(size == 0 for size in q_shape):
        raise ValueError("q must have non-empty shape [B,T,H,K]")
    batch, length, heads, key_dim = q_shape
    if _shape(k) != q_shape:
        raise ValueError("k must have the same shape as q")
    value_heads = _shape(v)[2] if len(_shape(v)) == 4 else 0
    value_dim = _shape(v)[3] if len(_shape(v)) == 4 else 0
    if _shape(v) != (batch, length, value_heads, value_dim) or not value_heads or not value_dim:
        raise ValueError("v must have non-empty shape [B,T,HV,V]")
    if value_heads % heads:
        raise ValueError("HV must be divisible by H")
    if _shape(g) != (batch, length, value_heads, key_dim):
        raise ValueError("g must have shape [B,T,HV,K]")
    if _shape(beta) != (batch, length, value_heads):
        raise ValueError("beta must have shape [B,T,HV]")
    state_shape = (batch, value_heads, key_dim, value_dim)
    if initial_state is not None and _shape(initial_state) != state_shape:
        raise ValueError("initial_state must have shape [B,HV,K,V]")
    if scale is None:
        scale = key_dim ** -0.5
    scale = _fp32(scale)
    group_size = value_heads // heads

    state = _zeros(state_shape) if initial_state is None else [
        [[[_fp32(value) for value in row] for row in head] for head in batch_state]
        for batch_state in initial_state
    ]
    output = _zeros((batch, length, value_heads, value_dim))
    for b in range(batch):
        for t in range(length):
            for hv in range(value_heads):
                h = hv // group_size
                for d in range(key_dim):
                    decay = _fp32(math.exp(g[b][t][hv][d]))
                    key = _fp32(k[b][t][h][d])
                    for n in range(value_dim):
                        state[b][hv][d][n] = _fp32(state[b][hv][d][n] * decay)
                delta = []
                for n in range(value_dim):
                    prediction = 0.0
                    for d in range(key_dim):
                        prediction = _fp32(
                            prediction + _fp32(_fp32(k[b][t][h][d]) * state[b][hv][d][n])
                        )
                    delta.append(_fp32(_fp32(v[b][t][hv][n]) - prediction))
                for d in range(key_dim):
                    update_scale = _fp32(_fp32(beta[b][t][hv]) * _fp32(k[b][t][h][d]))
                    for n in range(value_dim):
                        state[b][hv][d][n] = _fp32(
                            state[b][hv][d][n] + _fp32(update_scale * delta[n])
                        )
                for n in range(value_dim):
                    result = 0.0
                    for d in range(key_dim):
                        query = _fp32(_fp32(q[b][t][h][d]) * scale)
                        result = _fp32(result + _fp32(query * state[b][hv][d][n]))
                    output[b][t][hv][n] = result
    return output, state if output_final_state else None


__all__ = ["naive_recurrent_kda"]
