import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from relfx.model import DualBranchFxEncoder


class ModelRepresentationApiTest(unittest.TestCase):
    def test_forward_names_the_paper_embedding_and_fusion_explicitly(self):
        branch_a = torch.tensor([[1.0, 2.0, 3.0]])
        branch_b = torch.tensor([[4.0, 5.0, 6.0]])
        fusion = branch_b - branch_a
        projected = F.normalize(fusion[:, :2], dim=-1)
        model = SimpleNamespace(
            forward_dual_backbone=lambda original, processed: (branch_a, branch_b),
            fusion=lambda emb_a, emb_b: emb_b - emb_a,
            forward_projection=lambda value: F.normalize(value[:, :2], dim=-1),
        )

        output = DualBranchFxEncoder.forward(model, None, None)

        torch.testing.assert_close(output["fusion"], fusion)
        torch.testing.assert_close(output["embedding"], projected)
        self.assertIs(output["embedding"], output["z"])

    def test_get_embedding_returns_projected_z(self):
        projected = torch.tensor([[0.6, 0.8]])
        model = SimpleNamespace(
            forward=lambda original, processed: {"z": projected},
        )

        output = DualBranchFxEncoder.get_embedding(model, None, None)

        self.assertIs(output, projected)

    def test_get_fusion_representation_returns_raw_e_fx_by_default(self):
        branch_a = torch.tensor([[1.0, 1.0, 1.0]])
        branch_b = torch.tensor([[4.0, 5.0, 1.0]])
        model = SimpleNamespace(
            forward_dual_backbone=lambda original, processed: (branch_a, branch_b),
            fusion=lambda emb_a, emb_b: emb_b - emb_a,
        )

        output = DualBranchFxEncoder.get_fusion_representation(model, None, None)

        torch.testing.assert_close(output, torch.tensor([[3.0, 4.0, 0.0]]))

    def test_get_fusion_representation_can_normalize_e_fx(self):
        branch_a = torch.tensor([[1.0, 1.0, 1.0]])
        branch_b = torch.tensor([[4.0, 5.0, 1.0]])
        model = SimpleNamespace(
            forward_dual_backbone=lambda original, processed: (branch_a, branch_b),
            fusion=lambda emb_a, emb_b: emb_b - emb_a,
        )

        output = DualBranchFxEncoder.get_fusion_representation(
            model, None, None, normalized=True
        )

        torch.testing.assert_close(output, torch.tensor([[0.6, 0.8, 0.0]]))


if __name__ == "__main__":
    unittest.main()
