import sys
import unittest
from unittest.mock import patch

import torch

from scripts.embed import parse_args, select_representation


class EmbedScriptTest(unittest.TestCase):
    def test_projected_is_the_cli_default(self):
        argv = [
            "embed.py",
            "--checkpoint",
            "model.safetensors",
            "--reference",
            "reference.wav",
            "--processed",
            "processed.wav",
            "--output",
            "embedding.npy",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.representation, "projected")

    def test_selects_projected_representation(self):
        projected = torch.tensor([[0.6, 0.8]])
        output = {
            "z": projected,
            "embedding": projected,
            "fusion": torch.tensor([[3.0, 4.0, 0.0]]),
        }

        selected = select_representation(output, "projected")

        self.assertIs(selected, projected)
        self.assertEqual(tuple(selected.shape), (1, 2))

    def test_selects_raw_fusion_representation(self):
        fusion = torch.tensor([[3.0, 4.0, 0.0]])
        output = {
            "z": torch.tensor([[0.6, 0.8]]),
            "embedding": torch.tensor([[0.6, 0.8]]),
            "fusion": fusion,
        }

        selected = select_representation(output, "fusion")

        self.assertIs(selected, fusion)
        self.assertEqual(tuple(selected.shape), (1, 3))

    def test_rejects_unknown_representation(self):
        with self.assertRaisesRegex(ValueError, "Unknown representation"):
            select_representation({}, "unknown")


if __name__ == "__main__":
    unittest.main()
