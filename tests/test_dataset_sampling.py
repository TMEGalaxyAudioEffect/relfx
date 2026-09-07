import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from relfx.dataset import AudioSegmentDataset


class PaperCrossSegmentSamplingTest(unittest.TestCase):
    def _make_dataset(self, manifest, files, density_filter=False, stems=False):
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        audio_root = root / "audio"
        audio_root.mkdir()
        manifest_path = root / "sections.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        scanner = patch.object(
            AudioSegmentDataset,
            "_scan_audio_files",
            return_value=[str(audio_root / name) for name in files],
        )
        try:
            with scanner:
                dataset = AudioSegmentDataset(
                    audio_dir=str(audio_root),
                    segment_samples=100,
                    sample_rate=10,
                    cross_segment=True,
                    structure_segment_json=str(manifest_path),
                    density_filter_audio_dirs=(
                        {str(audio_root)} if density_filter else set()
                    ),
                    stem_audio_dirs={str(audio_root)} if stems else set(),
                )
        except Exception:
            temporary_directory.cleanup()
            raise
        self.addCleanup(temporary_directory.cleanup)
        return dataset

    def test_requires_structural_manifest(self):
        with patch.object(
            AudioSegmentDataset,
            "_scan_audio_files",
            return_value=["/audio/001_take.wav"],
        ):
            with self.assertRaisesRegex(RuntimeError, "requires a structural"):
                AudioSegmentDataset(
                    audio_dir="/audio",
                    cross_segment=True,
                    structure_segment_json=None,
                )

    def test_keeps_only_sections_long_enough_for_both_clips(self):
        dataset = self._make_dataset(
            {
                "001": {
                    "segments": [
                        {"label": "verse", "start": 0, "end": 19},
                        {"label": "chorus", "start": 25, "end": 48},
                        {"label": "bridge", "start": 50, "end": 80},
                    ]
                }
            },
            ["001_take.wav"],
        )

        self.assertEqual(
            dataset._structure_segments["001"],
            [{"label": "chorus", "start": 25.0, "end": 48.0}],
        )

    def test_rejects_incomplete_manifest_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "does not cover 1 audio files"):
            self._make_dataset(
                {
                    "001": {
                        "segments": [
                            {"label": "verse", "start": 0, "end": 25}
                        ]
                    }
                },
                ["001_take.wav", "002_take.wav"],
            )

    def test_samples_adjacent_non_overlapping_clips_from_one_section(self):
        dataset = self._make_dataset(
            {
                "001": {
                    "segments": [
                        {"label": "verse", "start": 5, "end": 30}
                    ]
                }
            },
            ["001_take.wav"],
        )
        filepath = dataset.audio_files[0]
        calls = []

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return np.zeros((2, dataset.segment_samples), dtype=np.float32)

        with patch("relfx.dataset.sf.info") as info, patch.object(
            dataset, "_load_segment", side_effect=fake_load
        ):
            info.return_value = SimpleNamespace(samplerate=10, frames=400)
            first, second = dataset._load_two_structured_segments(filepath)

        self.assertEqual(first.shape, (2, 100))
        self.assertEqual(second.shape, (2, 100))
        self.assertEqual(len(calls), 2)
        first_start = calls[0][1]["start_hint"]
        second_start = calls[1][1]["start_hint"]
        self.assertGreaterEqual(first_start, 50)
        self.assertLessEqual(second_start + 100, 300)
        self.assertEqual(second_start, first_start + 100)
        self.assertTrue(calls[0][1]["strict_start"])
        self.assertTrue(calls[1][1]["strict_start"])

    def test_song_id_manifest_can_be_shared_by_multiple_stems(self):
        dataset = self._make_dataset(
            {
                "001": {
                    "segments": [
                        {"label": "verse", "start": 0, "end": 25}
                    ]
                }
            },
            ["001_vocals.wav", "001_drums.wav"],
        )

        self.assertEqual(len(dataset.audio_files), 2)
        self.assertEqual(
            {dataset._file_structure_keys[path] for path in dataset.audio_files},
            {"001"},
        )

    def test_full_mix_positive_observations_use_different_songs(self):
        dataset = self._make_dataset(
            {
                song_id: {
                    "segments": [
                        {"label": "verse", "start": 0, "end": 25}
                    ]
                }
                for song_id in ("001", "002")
            },
            ["001_a.wav", "001_b.wav", "002_a.wav"],
        )

        with patch("relfx.dataset.random.randint", side_effect=[1, 2]):
            self.assertEqual(dataset._sample_second_track_index(0), 2)

    def test_stems_may_come_from_the_same_song(self):
        dataset = self._make_dataset(
            {
                "001": {
                    "segments": [
                        {"label": "verse", "start": 0, "end": 25}
                    ]
                }
            },
            ["001_vocals.wav", "001_drums.wav"],
            density_filter=True,
            stems=True,
        )

        with patch("relfx.dataset.random.randint", return_value=1):
            self.assertEqual(dataset._sample_second_track_index(0), 1)

    def test_sampling_metadata_records_policy_and_provenance(self):
        dataset = self._make_dataset(
            {
                "001": {
                    "segments": [
                        {"label": "verse", "start": 0, "end": 25}
                    ]
                }
            },
            ["001_take.wav"],
        )

        metadata = dataset.sampling_metadata()
        self.assertEqual(metadata["policy"], "same_section_adjacent")
        self.assertEqual(metadata["segment_samples"], 100)
        self.assertEqual(metadata["section_labels"], ["chorus", "verse"])
        self.assertEqual(len(metadata["structural_manifest_sha256"]), 64)
        self.assertEqual(len(metadata["sampler_source_sha256"]), 64)
        self.assertEqual(metadata["density_filtered_files"], 0)
        self.assertEqual(metadata["stem_files"], 0)
        self.assertEqual(metadata["minimum_content_ratio"], 0.7)
        self.assertNotIn("path", metadata)


if __name__ == "__main__":
    unittest.main()
