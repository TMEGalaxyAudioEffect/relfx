# Data Format

## Audio roots

Each `--audio-dir` is scanned recursively for WAV, FLAC, MP3, OGG, and M4A
files. Files shorter than two seconds or unreadable files are skipped.

Train/validation splitting is performed by inferred song ID so variants or
stems belonging to one detected song remain in one split. For a custom
collection, arrange each song under a stable parent directory or use a stable
numeric filename prefix.

## Structural-segment manifest

Structural metadata is optional. Set:

```bash
export RELFX_STRUCTURE_SEGMENTS=/path/to/segments.json
export RELFX_STRUCTURED_AUDIO_DIRS=/path/to/full-mix-audio
```

The JSON maps a filename stem to a list of segments:

```json
{
  "track_001": {
    "segments": [
      {"label": "verse", "start": 12.0, "end": 28.0, "duration": 16.0},
      {"label": "chorus", "start": 43.0, "end": 59.0, "duration": 16.0}
    ]
  }
}
```

Only `verse` and `chorus` entries at least ten seconds long are candidates by
default. Two distinct candidate segments are sampled. They are not guaranteed
to share a label or be adjacent.

## Density filtering

Set `RELFX_DENSITY_FILTER_AUDIO_DIRS` to apply the paper's 70% non-silent-frame
filter to selected roots. Multiple paths in environment variables are
separated by `:` on Unix-like systems and `;` on Windows.

## Data rights

No training audio is included. Users are responsible for acquiring datasets
and processing only material for which they have the required rights.
