import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from relfx.bidirectional import (
    orient_fusion_for_regression,
    swap_ordered_pairs,
)
from relfx.model import FusionModule, ProjectionHead


class ModelVariantTest(unittest.TestCase):
    def test_base_diff_gate_keeps_concatenated_gate_input(self):
        fusion = FusionModule(
            embed_dim=8,
            fusion_type="diff_gate",
            model_variant="base",
        )

        self.assertEqual(fusion.gate_proj[0].in_features, 16)

    def test_bidirectional_fusion_is_strictly_antisymmetric(self):
        torch.manual_seed(7)
        fusion = FusionModule(
            embed_dim=8,
            fusion_type="diff_gate",
            model_variant="bidirectional",
        )
        branch_a = torch.randn(4, 8)
        branch_b = torch.randn(4, 8)

        forward = fusion(branch_a, branch_b)
        reverse = fusion(branch_b, branch_a)

        torch.testing.assert_close(forward, -reverse, rtol=0, atol=0)
        torch.testing.assert_close(
            fusion(branch_a, branch_a),
            torch.zeros_like(branch_a),
            rtol=0,
            atol=0,
        )
        self.assertEqual(fusion.gate_proj[0].in_features, 8)

    def test_bidirectional_projection_uses_tanh_with_learned_biases(self):
        base = ProjectionHead(8, 4, model_variant="base")
        bidirectional = ProjectionHead(8, 4, model_variant="bidirectional")

        self.assertIsInstance(base[1], nn.ReLU)
        self.assertIsInstance(bidirectional[1], nn.Tanh)
        self.assertIsNotNone(bidirectional[0].bias)
        self.assertIsNotNone(bidirectional[2].bias)

    def test_projection_biases_prevent_a_strict_end_to_end_guarantee(self):
        projection = ProjectionHead(2, 2, model_variant="bidirectional")
        with torch.no_grad():
            projection[0].weight.copy_(torch.eye(2))
            projection[0].bias.copy_(torch.tensor([0.2, -0.1]))
            projection[2].weight.copy_(torch.eye(2))
            projection[2].bias.copy_(torch.tensor([0.3, 0.1]))
        fusion = torch.tensor([[0.4, -0.7]])

        forward = F.normalize(projection(fusion), dim=-1)
        reverse = F.normalize(projection(-fusion), dim=-1)

        self.assertFalse(torch.allclose(reverse, -forward))

    def test_bidirectional_variant_rejects_other_fusions(self):
        with self.assertRaisesRegex(ValueError, "requires fusion_type='diff_gate'"):
            FusionModule(
                embed_dim=8,
                fusion_type="diff",
                model_variant="bidirectional",
            )

    def test_pair_swaps_and_regression_orientation_share_the_mask(self):
        first = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        second = torch.tensor([[[4.0]], [[5.0]], [[6.0]]])
        reverse_mask = torch.tensor([False, True, False])

        ordered_first, ordered_second = swap_ordered_pairs(
            first, second, reverse_mask
        )
        fusion = (ordered_second - ordered_first).reshape(3, 1)
        regression_fusion = orient_fusion_for_regression(
            fusion, reverse_mask
        )

        torch.testing.assert_close(
            ordered_first,
            torch.tensor([[[1.0]], [[5.0]], [[3.0]]]),
        )
        torch.testing.assert_close(
            ordered_second,
            torch.tensor([[[4.0]], [[2.0]], [[6.0]]]),
        )
        torch.testing.assert_close(
            regression_fusion,
            torch.tensor([[3.0], [3.0], [3.0]]),
        )


if __name__ == "__main__":
    unittest.main()
