import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_training_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_training_manifest", SCRIPT_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class TrainingManifestBuilderTest(unittest.TestCase):
    @staticmethod
    def _write_audio(path, seconds=21):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(seconds * 100, dtype=np.float32), 100)

    def test_unannotated_audio_has_no_structure_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "moisesdb"
            audio_path = root / "song" / "vocals.wav"
            self._write_audio(audio_path)

            entries, eligible = BUILDER.unlabeled_ranges(root, "full_audio")

            self.assertEqual(eligible, 1)
            entry = entries["moisesdb/song/vocals.wav"]
            self.assertEqual(entry["sampling_scope"], "full_audio")
            self.assertNotIn("label", entry["segments"][0])

    def test_only_structural_sources_keep_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mixes"
            audio_path = root / "001.wav"
            source_path = Path(temp_dir) / "sections.json"
            self._write_audio(audio_path)
            source_path.write_text(
                json.dumps({
                    "001": {
                        "segments": [
                            {"label": "verse", "start": 0, "end": 21}
                        ]
                    }
                }),
                encoding="utf-8",
            )

            entries, eligible, _ = BUILDER.structural_ranges(root, source_path)

            self.assertEqual(eligible, 1)
            entry = entries["mixes/001.wav"]
            self.assertEqual(entry["sampling_scope"], "structural_section")
            self.assertEqual(entry["segments"][0]["label"], "verse")


if __name__ == "__main__":
    unittest.main()
