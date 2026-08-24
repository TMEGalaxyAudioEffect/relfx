import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from relfx.checkpoint import (
    RELFX_METADATA_KEY,
    load_safetensors_artifact,
    resolve_model_variant,
)


class SafetensorsCheckpointTest(unittest.TestCase):
    def test_loads_tensors_and_json_metadata(self):
        tensors = {
            "layer.bias": torch.tensor([1.0, -2.0]),
            "layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        }
        metadata = {"artifact_format": "relfx-safetensors-v1", "epoch": 199}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={RELFX_METADATA_KEY: json.dumps(metadata)},
            )
            loaded_tensors, loaded_metadata = load_safetensors_artifact(path)

        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(loaded_tensors.keys(), tensors.keys())
        for name, tensor in tensors.items():
            self.assertTrue(torch.equal(loaded_tensors[name], tensor))

    def test_rejects_pickle_checkpoint_suffix(self):
        with self.assertRaisesRegex(ValueError, "safetensors"):
            load_safetensors_artifact("model.pt")

    def test_resolves_existing_v6_artifacts_as_base(self):
        metadata = {"version": "v6", "model_config": {"fusion_type": "diff_gate"}}

        self.assertEqual(resolve_model_variant(metadata), "base")

    def test_resolves_original_v8_artifacts_as_bidirectional(self):
        metadata = {"version": "v8", "model_config": {"fusion_type": "diff_gate"}}

        self.assertEqual(resolve_model_variant(metadata), "bidirectional")

    def test_explicit_model_variant_takes_precedence(self):
        metadata = {
            "version": "v8",
            "model_config": {"model_variant": "base"},
        }

        self.assertEqual(resolve_model_variant(metadata), "base")


if __name__ == "__main__":
    unittest.main()
