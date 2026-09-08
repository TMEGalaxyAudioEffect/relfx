#!/usr/bin/env python3
"""Build a source-aware manifest without requiring synthetic labels."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import soundfile as sf

from relfx.dataset import AudioSegmentDataset


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def manifest_key(root, path):
    return f"{root.name}/{path.relative_to(root).as_posix()}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_files(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        try:
            info = sf.info(path)
        except Exception:
            continue
        duration = info.frames / info.samplerate
        if duration >= 2.0:
            yield path, duration


def unlabeled_ranges(root, scope):
    entries = {}
    eligible = 0
    for path, duration in audio_files(root):
        entries[manifest_key(root, path)] = {
            "sampling_scope": scope,
            "segments": [{"start": 0.0, "end": duration}],
        }
        eligible += duration >= 20.0
    return entries, eligible


def structural_ranges(root, source_manifest):
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    entries = {}
    eligible = 0
    missing = []
    for path, _ in audio_files(root):
        candidates = [
            path.stem,
            AudioSegmentDataset._extract_song_id(str(path)),
        ]
        source_key = next((key for key in candidates if key in source), None)
        if source_key is None:
            missing.append(path)
            continue
        item = copy.deepcopy(source[source_key])
        item["sampling_scope"] = "structural_section"
        entries[manifest_key(root, path)] = item
        eligible += any(
            str(segment.get("label", "")).lower() in {"verse", "chorus"}
            and float(segment.get("end", 0)) - float(segment.get("start", 0))
            >= 20.0
            for segment in item.get("segments", [])
        )
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise RuntimeError(
            f"Structural manifest misses {len(missing)} files: {examples}"
        )
    return entries, eligible, sha256(source_manifest)


def merge(target, source, source_name):
    overlap = set(target).intersection(source)
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise RuntimeError(f"Manifest key collision from {source_name}: {examples}")
    target.update(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structural-source", nargs=2, action="append", default=[],
        metavar=("AUDIO_DIR", "SECTION_JSON"),
    )
    parser.add_argument("--presegmented-dir", action="append", default=[])
    parser.add_argument("--full-audio-dir", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {}
    stats = []
    for root_value, source_value in args.structural_source:
        root, source = Path(root_value), Path(source_value)
        entries, eligible, source_hash = structural_ranges(root, source)
        merge(manifest, entries, root.name)
        stats.append({"root_name": root.name, "scope": "structural_section",
                      "files": len(entries), "eligible": eligible,
                      "section_manifest_sha256": source_hash})
    for root_value, scope in [
        *((value, "presegmented_audio") for value in args.presegmented_dir),
        *((value, "full_audio") for value in args.full_audio_dir),
    ]:
        root = Path(root_value)
        entries, eligible = unlabeled_ranges(root, scope)
        merge(manifest, entries, root.name)
        stats.append({"root_name": root.name, "scope": scope,
                      "files": len(entries), "eligible": eligible})

    if not manifest:
        parser.error("provide at least one audio source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {"entries": len(manifest), "sources": stats}
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
