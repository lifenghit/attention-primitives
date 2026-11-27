#!/usr/bin/env python3
"""Differential checks against the pinned FLA upstream naive implementations."""

import os
import sys
import unittest

import torch


UPSTREAM = os.environ.get("FLA_UPSTREAM_ROOT", "/storage/lf02381138/ling_pr/output/fla_upstream_main")
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fla.ops.kda.naive import naive_recurrent_kda as upstream_kda
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule as upstream_gdn
from fla.ops.gla.naive import naive_recurrent_gla as upstream_gla
from fla.ops.dsa.naive import naive_dsa as upstream_dsa
from attention_primitives.kda import naive_recurrent_kda
from attention_primitives.gdn import naive_recurrent_gated_delta_rule
from attention_primitives.gla import naive_recurrent_gla
from attention_primitives.dsa import naive_dsa


class FLAAlignmentTest(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-5):
        if expected is None:
            self.assertIsNone(actual)
            return
        torch.testing.assert_close(torch.tensor(actual), expected, atol=atol, rtol=atol)

    def test_kda_with_gva(self):
        torch.manual_seed(11)
        q, k = torch.randn(1, 3, 1, 2), torch.randn(1, 3, 1, 2)
        v = torch.randn(1, 3, 2, 2)
        g = -torch.rand(1, 3, 2, 2)
        beta = torch.rand(1, 3, 2)
        initial = torch.randn(1, 2, 2, 2)
        expected_o, expected_s = upstream_kda(q, k, v, g, beta, initial_state=initial, output_final_state=True)
        actual_o, actual_s = naive_recurrent_kda(q.tolist(), k.tolist(), v.tolist(), g.tolist(), beta.tolist(),
                                                  initial_state=initial.tolist(), output_final_state=True)
        self.assert_close(actual_o, expected_o)
        self.assert_close(actual_s, expected_s)

    def test_gdn(self):
        torch.manual_seed(12)
        q, k, v = torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2)
        g = -torch.rand(1, 3, 2)
        beta = torch.rand(1, 3, 2)
        expected_o, expected_s = upstream_gdn(q, k, v, beta, g, output_final_state=True)
        actual_o, actual_s = naive_recurrent_gated_delta_rule(q.tolist(), k.tolist(), v.tolist(), beta.tolist(), g.tolist(),
                                                               output_final_state=True)
        self.assert_close(actual_o, expected_o)
        self.assert_close(actual_s, expected_s)

    def test_gla(self):
        torch.manual_seed(13)
        q, k, v = torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2)
        gk = -torch.rand(1, 3, 2, 2)
        expected_o, expected_s = upstream_gla(q, k, v, gk, output_final_state=True)
        actual_o, actual_s = naive_recurrent_gla(q.tolist(), k.tolist(), v.tolist(), gk.tolist(), output_final_state=True)
        self.assert_close(actual_o, expected_o)
        self.assert_close(actual_s, expected_s)

    def test_dsa(self):
        torch.manual_seed(14)
        q, k, v = torch.randn(1, 5, 2, 2), torch.randn(1, 5, 1, 2), torch.randn(1, 5, 1, 2)
        q_idx, k_idx = torch.randn(1, 5, 2, 2), torch.randn(1, 5, 2)
        w_idx = torch.rand(1, 5, 2)
        expected_o, expected_indices = upstream_dsa(q, k, v, q_idx, k_idx, w_idx, topk=1, return_indices=True)
        actual_o, actual_indices = naive_dsa(q.tolist(), k.tolist(), v.tolist(), q_idx.tolist(), k_idx.tolist(),
                                             w_idx.tolist(), topk=1, return_indices=True)
        for actual_row, expected_row in zip(actual_indices[0], expected_indices[0].tolist()):
            self.assertEqual(set(actual_row), set(expected_row))
        self.assert_close(actual_o, expected_o)


if __name__ == "__main__":
    unittest.main()
