# Data Format

## Audio roots

Each `--audio-dir` is scanned recursively for WAV, FLAC, MP3, OGG, and M4A
files. Files shorter than two seconds or unreadable files are skipped.

Train/validation splitting is performed by inferred song ID so variants or
stems belonging to one detected song remain in one split. For a custom
collection, arrange each song under a stable parent directory or use a stable
numeric filename prefix.

## Sampling manifest

The adjacent-pair sampler requires a manifest describing the valid range for
each audio file. Set:

```bash
export RELFX_STRUCTURE_SEGMENTS=/path/to/segments.json
```

Use `sampling_scope: "structural_section"` when structure annotations are
available. These entries require `verse` or `chorus` labels:

```json
{
  "track_001": {
    "sampling_scope": "structural_section",
    "segments": [
      {"label": "verse", "start": 12.0, "end": 36.0, "duration": 24.0},
      {"label": "chorus", "start": 43.0, "end": 68.0, "duration": 25.0}
    ]
  }
}
```

For audio files already cut to one structural section, use
`presegmented_audio`. For unannotated stems such as MoisesDB, use `full_audio`.
Neither scope uses or requires a synthetic structure label:

```json
{
  "source-name/song/stem.wav": {
    "sampling_scope": "full_audio",
    "segments": [{"start": 0.0, "end": 184.2}]
  }
}
```

Every range must be at least 20 seconds for two paper-configured 10-second
clips. The second clip begins exactly where the first ends, so each pair is
adjacent and non-overlapping. For `structural_section`, only eligible verse and
chorus ranges are used.

Every scanned audio file must match a manifest entry. Source-relative keys of
the form `audio-root-name/relative/path.wav` are preferred because they safely
support duplicate filenames in different directories. Filename-stem and
inferred-song-ID keys remain supported for simple layouts. Files with no
eligible range are excluded before the song-level split; missing coverage is
an error.

Generate a combined manifest without assigning labels to unannotated audio:

```bash
python scripts/build_training_manifest.py \
  --structural-source /path/to/structured-audio /path/to/sections.json \
  --presegmented-dir /path/to/presegmented-audio \
  --full-audio-dir /path/to/moisesdb \
  --output outputs/training-manifest.json
```

## Density filtering

Pass `--density-filter-audio-dir /path/to/moisesdb` to apply the paper's 70%
non-silent-frame filter to that root. Repeat the option for multiple roots.
`RELFX_DENSITY_FILTER_AUDIO_DIRS` provides the equivalent environment setting;
paths are separated by `:` on Unix-like systems and `;` on Windows. Both clips
in an adjacent pair must pass.

## Data rights

No training audio is included. Users are responsible for acquiring datasets
and processing only material for which they have the required rights.
