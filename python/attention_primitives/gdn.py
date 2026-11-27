"""FP32 reference implementation of the gated delta rule recurrence."""

import math
import struct
from typing import List, Optional, Sequence, Tuple


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _check_qk(name: str, value: Sequence, batch: int, length: int, heads: int, dim: int) -> None:
    if len(value) != batch:
        raise ValueError(f"{name} must have shape [B, T, H, K]")
    for b in range(batch):
        if len(value[b]) != length:
            raise ValueError(f"{name} must have shape [B, T, H, K]")
        for t in range(length):
            if len(value[b][t]) != heads:
                raise ValueError(f"{name} must have shape [B, T, H, K]")
            for h in range(heads):
                if len(value[b][t][h]) != dim:
                    raise ValueError(f"{name} must have shape [B, T, H, K]")


def _check_gates(name: str, value: Sequence, batch: int, length: int, heads: int) -> None:
    if len(value) != batch:
        raise ValueError(f"{name} must have shape [B, T, H]")
    for b in range(batch):
        if len(value[b]) != length:
            raise ValueError(f"{name} must have shape [B, T, H]")
        for t in range(length):
            if len(value[b][t]) != heads:
                raise ValueError(f"{name} must have shape [B, T, H]")


def naive_recurrent_gated_delta_rule(
    q: Sequence[Sequence[Sequence[Sequence[float]]]],
    k: Sequence[Sequence[Sequence[Sequence[float]]]],
    v: Sequence[Sequence[Sequence[Sequence[float]]]],
    beta: Sequence[Sequence[Sequence[float]]],
    g: Sequence[Sequence[Sequence[float]]],
    scale: Optional[float] = None,
    initial_state: Optional[Sequence[Sequence[Sequence[Sequence[float]]]]] = None,
    output_final_state: bool = False,
) -> Tuple[List[List[List[List[float]]]], Optional[List[List[List[List[float]]]]]]:
    """Evaluate q/k/v [B,T,H,K or V], beta/g [B,T,H], state [B,H,K,V]."""
    if not q or not q[0] or not q[0][0] or not q[0][0][0]:
        raise ValueError("B, T, H and K must be positive")
    batch = len(q)
    length = len(q[0])
    heads = len(q[0][0])
    key_dim = len(q[0][0][0])
    _check_qk("q", q, batch, length, heads, key_dim)
    _check_qk("k", k, batch, length, heads, key_dim)
    _check_gates("beta", beta, batch, length, heads)
    _check_gates("g", g, batch, length, heads)

    if len(v) != batch:
        raise ValueError("v must have shape [B, T, H, V]")
    value_dim = 0
    for b in range(batch):
        if len(v[b]) != length:
            raise ValueError("v must have shape [B, T, H, V]")
        for t in range(length):
            if len(v[b][t]) != heads:
                raise ValueError("v must have shape [B, T, H, V]")
            for h in range(heads):
                if not v[b][t][h]:
                    raise ValueError("V must be positive")
                if value_dim == 0:
                    value_dim = len(v[b][t][h])
                if len(v[b][t][h]) != value_dim:
                    raise ValueError("v must have shape [B, T, H, V]")

    q_scale = _f32(1.0 / math.sqrt(float(key_dim))) if scale is None else _f32(float(scale))
    if initial_state is None:
        state = [
            [[[0.0 for _ in range(value_dim)] for _ in range(key_dim)] for _ in range(heads)]
            for _ in range(batch)
        ]
    else:
        if len(initial_state) != batch:
            raise ValueError("initial_state must have shape [B, H, K, V]")
        state = []
        for b in range(batch):
            if len(initial_state[b]) != heads:
                raise ValueError("initial_state must have shape [B, H, K, V]")
            state.append([])
            for h in range(heads):
                if len(initial_state[b][h]) != key_dim:
                    raise ValueError("initial_state must have shape [B, H, K, V]")
                state[b].append([])
                for i in range(key_dim):
                    if len(initial_state[b][h][i]) != value_dim:
                        raise ValueError("initial_state must have shape [B, H, K, V]")
                    state[b][h].append([_f32(float(x)) for x in initial_state[b][h][i]])

    output = [
        [
            [[0.0 for _ in range(value_dim)] for _ in range(heads)]
            for _ in range(length)
        ]
        for _ in range(batch)
    ]
    for b in range(batch):
        for h in range(heads):
            for t in range(length):
                decay = _f32(math.exp(_f32(float(g[b][t][h]))))
                for i in range(key_dim):
                    for j in range(value_dim):
                        state[b][h][i][j] = _f32(decay * state[b][h][i][j])
                for j in range(value_dim):
                    prediction = 0.0
                    for i in range(key_dim):
                        product = _f32(_f32(float(k[b][t][h][i])) * state[b][h][i][j])
                        prediction = _f32(prediction + product)
                    delta = _f32(_f32(float(v[b][t][h][j])) - prediction)
                    delta = _f32(_f32(float(beta[b][t][h])) * delta)
                    for i in range(key_dim):
                        update = _f32(_f32(float(k[b][t][h][i])) * delta)
                        state[b][h][i][j] = _f32(state[b][h][i][j] + update)
                for j in range(value_dim):
                    readout = 0.0
                    for i in range(key_dim):
                        scaled_q = _f32(_f32(float(q[b][t][h][i])) * q_scale)
                        readout = _f32(readout + _f32(scaled_q * state[b][h][i][j]))
                    output[b][t][h][j] = readout

    return output, state if output_final_state else None


gated_delta_rule = naive_recurrent_gated_delta_rule


__all__ = ["gated_delta_rule", "naive_recurrent_gated_delta_rule"]
