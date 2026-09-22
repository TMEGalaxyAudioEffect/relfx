import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "convert_checkpoint_to_safetensors.py"
)
SPEC = importlib.util.spec_from_file_location("release_converter", SCRIPT_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONVERTER)


class ReleaseConversionTest(unittest.TestCase):
    def _metadata(self):
        return {
            "version": "v6",
            "cross_segment": True,
            "cross_segment_policy": "same_section_adjacent",
            "cross_segment_recipe_version": "ismir2026-camera-ready-v1",
            "model_config": {"fusion_type": "diff_gate"},
            "sampling_config": {
                "eligible_files_before_split": 2585,
                "section_labels": [
                    "chorus",
                    "presegmented_section",
                    "unsegmented_stem",
                    "verse",
                ],
                "pairing_scope_counts": {
                    "sq_presegmented_section": 4406,
                    "moises_adjacent_within_stem": 2585,
                    "quku_structural_section": 2000,
                },
            },
        }

    def test_same_section_recipe_preserves_source_provenance(self):
        source = self._metadata()
        converted = CONVERTER.apply_release_recipe(
            source,
            recipe="same-section-adjacent-v1",
            training_data="moisesdb-only",
        )

        self.assertEqual(converted["version"], "v6-fix")
        self.assertEqual(converted["source_checkpoint_version"], "v6")
        self.assertEqual(
            converted["source_cross_segment_recipe_version"],
            "ismir2026-camera-ready-v1",
        )
        self.assertEqual(converted["model_config"]["model_variant"], "base")
        sampling = converted["sampling_config"]
        self.assertEqual(sampling["dataset"], "MoisesDB")
        self.assertEqual(sampling["section_labels"], [])
        self.assertEqual(sampling["pairing_scope_counts"], {"full_audio": 2585})
        self.assertEqual(
            sampling["source_manifest_pairing_scope_counts"],
            source["sampling_config"]["pairing_scope_counts"],
        )
        self.assertEqual(source["version"], "v6")

    def test_recipe_rejects_wrong_sampling_policy(self):
        source = self._metadata()
        source["cross_segment_policy"] = "distinct_section"

        with self.assertRaisesRegex(ValueError, "same-section adjacent"):
            CONVERTER.apply_release_recipe(
                source,
                recipe="same-section-adjacent-v1",
                training_data="moisesdb-only",
            )


if __name__ == "__main__":
    unittest.main()
