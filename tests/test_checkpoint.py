import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from relfx.checkpoint import RELFX_METADATA_KEY, load_safetensors_artifact


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


if __name__ == "__main__":
    unittest.main()
