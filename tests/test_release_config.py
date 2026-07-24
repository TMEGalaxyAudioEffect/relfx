import unittest

from relfx import config
from relfx.dataset import AudioSegmentDataset


class ReleaseConfigTest(unittest.TestCase):
    def test_defaults_contain_no_machine_paths(self):
        self.assertEqual(config.AUDIO_DIRS, [])
        self.assertIsNone(config.STRUCTURE_SEGMENT_JSON)
        self.assertEqual(config.TRAIN_CONFIG["lr"], 3e-4)
        self.assertEqual(config.TRAIN_CONFIG["epochs"], 200)

    def test_song_level_split_is_disjoint(self):
        files = [
            "/audio/001_take_a.wav",
            "/audio/001_take_b.wav",
            "/audio/002_take_a.wav",
            "/audio/003_take_a.wav",
            "/audio/004_take_a.wav",
        ]
        train = set(
            AudioSegmentDataset._split_by_song_id(
                files, split="train", val_ratio=0.25, seed=42
            )
        )
        val = set(
            AudioSegmentDataset._split_by_song_id(
                files, split="val", val_ratio=0.25, seed=42
            )
        )
        self.assertTrue(train.isdisjoint(val))
        self.assertEqual(train | val, set(files))


if __name__ == "__main__":
    unittest.main()
