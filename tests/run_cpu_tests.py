#!/usr/bin/env python3
"""Dependency-free CPU reference checks for the attention primitive contracts."""

import argparse
import math
import unittest


def block_scores(q, k, q_heads, kv_heads, dim, causal=False):
    if not q_heads or not kv_heads or q_heads % kv_heads:
        raise ValueError("QHeads must be a positive integer multiple of KVHeads")
    if dim <= 0:
        raise ValueError("D must be positive")
    q_blocks = len(q)
    kv_blocks = len(k)
    scores = []
    group_size = q_heads // kv_heads
    for qb in range(q_blocks):
        head_rows = []
        for qh in range(q_heads):
            kh = qh // group_size
            q_vec = [float(value) for value in q[qb][qh]]
            row = []
            for kb in range(kv_blocks):
                if causal and kb > qb:
                    row.append(-math.inf)
                    continue
                k_vec = [float(value) for value in k[kb][kh]]
                if len(q_vec) != dim or len(k_vec) != dim:
                    raise ValueError("Q and K vectors must have length D")
                q_mean = math.fsum(q_vec) / dim
                k_mean = math.fsum(k_vec) / dim
                row.append(q_mean * k_mean / math.sqrt(dim))
            head_rows.append(row)
        scores.append(head_rows)
    return scores


def stable_top_k(scores, top_k):
    if top_k <= 0:
        raise ValueError("topK must be positive")
    all_indices = []
    all_counts = []
    for block in scores:
        block_indices = []
        block_counts = []
        for row in block:
            candidates = [
                (score, index)
                for index, score in enumerate(row)
                if score != -math.inf
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [index for _, index in candidates[:top_k]]
            block_counts.append(len(selected))
            block_indices.append(selected + [-1] * (top_k - len(selected)))
        all_indices.append(block_indices)
        all_counts.append(block_counts)
    return all_indices, all_counts


def select_blocks(indices, counts, block_table=None):
    physical = []
    positions = []
    for qb, block in enumerate(indices):
        physical_block = []
        position_block = []
        for qh, row in enumerate(block):
            physical_row = [-1] * len(row)
            position_row = [-1] * len(row)
            for position in range(counts[qb][qh]):
                logical = row[position]
                if block_table is not None and logical >= len(block_table):
                    raise ValueError("logical block index is outside the block table")
                mapped = logical if block_table is None else block_table[logical]
                if mapped < 0:
                    raise ValueError("physical block indices must be non-negative")
                physical_row[position] = mapped
                position_row[position] = position
            physical_block.append(physical_row)
            position_block.append(position_row)
        physical.append(physical_block)
        positions.append(position_block)
    return physical, counts, positions


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(left, right):
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("incompatible matrix shapes")
    right_t = transpose(right)
    return [
        [math.fsum(a * b for a, b in zip(row, column)) for column in right_t]
        for row in left
    ]


def cumulative_decay(decay, delta):
    result = []
    total = 0.0
    for value in delta:
        total += decay * value
        result.append(total)
    return result


def chunk_scan(x, b, c, delta, state_in, decay):
    length = len(delta)
    channels = len(x)
    state_dim = len(b)
    if len(c) != state_dim or any(len(row) != length for row in b + c):
        raise ValueError("B and C must both have shape [N, L]")
    if any(len(row) != length for row in x):
        raise ValueError("X must have shape [P, L]")
    if len(state_in) != channels or any(len(row) != state_dim for row in state_in):
        raise ValueError("StateIn must have shape [P, N]")

    cumsum = cumulative_decay(decay, delta)
    c_t = transpose(c)
    m = matmul(c_t, b)
    w = [[0.0] * length for _ in range(length)]
    for i in range(length):
        for j in range(i + 1):
            w[i][j] = m[i][j] * math.exp(cumsum[i] - cumsum[j]) * delta[j]
    y_intra = matmul(x, transpose(w))
    state_c = matmul(state_in, c)
    y_state = [
        [math.exp(cumsum[t]) * state_c[p][t] for t in range(length)]
        for p in range(channels)
    ]
    y = [
        [y_intra[p][t] + y_state[p][t] for t in range(length)]
        for p in range(channels)
    ]
    return y, cumsum


def state_update(x, b, delta, cumsum, state_in):
    length = len(delta)
    final_decay = math.exp(cumsum[-1])
    b_tilde = [
        [b[n][j] * delta[j] * math.exp(cumsum[-1] - cumsum[j]) for j in range(length)]
        for n in range(len(b))
    ]
    contribution = matmul(x, transpose(b_tilde))
    return [
        [final_decay * state_in[p][n] + contribution[p][n] for n in range(len(b))]
        for p in range(len(x))
    ]


def assert_matrix_almost_equal(test_case, actual, expected, places=12):
    test_case.assertEqual(len(actual), len(expected))
    for actual_row, expected_row in zip(actual, expected):
        test_case.assertEqual(len(actual_row), len(expected_row))
        for actual_value, expected_value in zip(actual_row, expected_row):
            test_case.assertAlmostEqual(actual_value, expected_value, places=places)


class SparsePrimitiveTests(unittest.TestCase):
    def test_pooling_scaling_causal_and_gqa(self):
        q = [[[1.0, 3.0], [2.0, 4.0]]]
        k = [
            [[2.0, 4.0], [10.0, 10.0]],
            [[1.0, 1.0], [20.0, 20.0]],
        ]
        scores = block_scores(q, k, q_heads=2, kv_heads=1, dim=2, causal=True)
        self.assertAlmostEqual(scores[0][0][0], 6.0 / math.sqrt(2.0))
        self.assertAlmostEqual(scores[0][1][0], 9.0 / math.sqrt(2.0))
        self.assertEqual(scores[0][0][1], -math.inf)
        self.assertEqual(scores[0][1][1], -math.inf)

    def test_top_k_ties_empty_rows_and_padding(self):
        scores = [[[2.0, 2.0, 1.0], [-math.inf, -math.inf, -math.inf]]]
        indices, counts = stable_top_k(scores, top_k=5)
        self.assertEqual(indices[0][0], [0, 1, 2, -1, -1])
        self.assertEqual(counts[0], [3, 0])
        self.assertEqual(indices[0][1], [-1] * 5)
        self.assertEqual(stable_top_k(scores, top_k=1)[0][0][0], [0])
        with self.assertRaises(ValueError):
            stable_top_k(scores, top_k=0)

    def test_identity_and_physical_block_selection(self):
        scores = [[[0.0, 3.0, 2.0]]]
        indices, counts = stable_top_k(scores, top_k=2)
        physical, _, positions = select_blocks(indices, counts, [4, 7, 9])
        self.assertEqual(physical, [[[7, 9]]])
        self.assertEqual(positions, [[[0, 1]]])
        identity, _, _ = select_blocks(indices, counts)
        self.assertEqual(identity, indices)
        with self.assertRaises(ValueError):
            select_blocks(indices, counts, [4, -1, 9])


class LinearPrimitiveTests(unittest.TestCase):
    def test_single_token_zero_state(self):
        x = [[2.0]]
        b = [[3.0]]
        c = [[4.0]]
        delta = [0.5]
        state = [[0.0]]
        y, cumsum = chunk_scan(x, b, c, delta, state, decay=-1.0)
        self.assertEqual(cumsum, [-0.5])
        assert_matrix_almost_equal(self, y, [[12.0]])
        updated = state_update(x, b, delta, cumsum, state)
        assert_matrix_almost_equal(self, updated, [[3.0]])

    def test_nonzero_initial_state_and_negative_decay(self):
        x = [[1.0, 2.0]]
        b = [[1.0, 0.5]]
        c = [[2.0, 1.0]]
        delta = [0.25, 0.5]
        state = [[3.0]]
        y, cumsum = chunk_scan(x, b, c, delta, state, decay=-2.0)
        expected_cumsum = [-0.5, -1.5]
        assert_matrix_almost_equal(self, [cumsum], [expected_cumsum])
        expected_y_0 = 1.0 * 2.0 * 1.0 * 0.25 + math.exp(-0.5) * 6.0
        w_10 = 1.0 * 1.0 * math.exp(-1.0) * 0.25
        w_11 = 1.0 * 0.5 * 0.5
        expected_y_1 = w_10 + 2.0 * w_11 + math.exp(-1.5) * 3.0
        assert_matrix_almost_equal(self, y, [[expected_y_0, expected_y_1]])
        updated = state_update(x, b, delta, cumsum, state)
        expected_state = math.exp(-1.5) * 3.0 + 1.0 * math.exp(-1.0) * 0.25 + 0.5
        assert_matrix_almost_equal(self, updated, [[expected_state]])

    def test_multiple_chunks_preserve_state(self):
        x_chunks = [[[1.0]], [[2.0, 1.0]]]
        b_chunks = [[[1.0]], [[0.5, 2.0]]]
        c_chunks = [[[1.0]], [[1.0, 0.5]]]
        deltas = [[1.0], [0.25, 0.5]]
        state = [[0.5]]
        all_y = []
        for x, b, c, delta in zip(x_chunks, b_chunks, c_chunks, deltas):
            y, cumsum = chunk_scan(x, b, c, delta, state, decay=-0.5)
            all_y.extend(y[0])
            state = state_update(x, b, delta, cumsum, state)
        self.assertEqual(len(all_y), 3)
        self.assertTrue(all(math.isfinite(value) for value in all_y + state[0]))
        replay_y, replay_cumsum = chunk_scan(
            x_chunks[-1], b_chunks[-1], c_chunks[-1], deltas[-1], [[0.5]], decay=-0.5
        )
        replay_state = state_update(
            x_chunks[-1], b_chunks[-1], deltas[-1], replay_cumsum, [[0.5]]
        )
        self.assertNotEqual(replay_y, all_y[-2:])
        self.assertNotEqual(replay_state, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", default="all", choices=("all", "sparse", "linear"))
    args = parser.parse_args()
    classes = {
        "sparse": (SparsePrimitiveTests,),
        "linear": (LinearPrimitiveTests,),
        "all": (SparsePrimitiveTests, LinearPrimitiveTests),
    }[args.suite]
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for test_class in classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
