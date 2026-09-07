# Data Format

## Audio roots

Each `--audio-dir` is scanned recursively for WAV, FLAC, MP3, OGG, and M4A
files. Files shorter than two seconds or unreadable files are skipped.

Train/validation splitting is performed by inferred song ID so variants or
stems belonging to one detected song remain in one split. For a custom
collection, arrange each song under a stable parent directory or use a stable
numeric filename prefix.

## Structural-section manifest

The paper-aligned cross-segment sampler requires structural metadata. Set:

```bash
export RELFX_STRUCTURE_SEGMENTS=/path/to/segments.json
```

The JSON maps either a filename stem or an inferred song ID to a list of
sections. A song-level key can therefore be shared by all MoisesDB stems from
that song:

```json
{
  "track_001": {
    "segments": [
      {"label": "verse", "start": 12.0, "end": 36.0, "duration": 24.0},
      {"label": "chorus", "start": 43.0, "end": 68.0, "duration": 25.0}
    ]
  }
}
```

Only `verse` and `chorus` sections long enough to contain two complete clips
are candidates: 20 seconds for the paper's 10-second clips. The sampler first
chooses one eligible section, then draws two consecutive 10-second clips from
that section. The second clip begins exactly where the first ends, so the pair
is adjacent and non-overlapping.

Every scanned audio file must match a filename-level or song-level manifest
entry. Files whose matched entry has no eligible section are excluded before
the song-level train/validation split. Training stops on missing manifest
coverage or if no eligible files remain. There is no random-position or
cross-section fallback in paper mode.

Mark stem collections with `--stem-audio-dir` (or `RELFX_STEM_AUDIO_DIRS`).
Positive observations from full-mix roots use different songs; observations
from a stem root may use different stems of the same song.

## Density filtering

Set `RELFX_DENSITY_FILTER_AUDIO_DIRS` to apply the paper's 70% non-silent-frame
filter to selected roots. Both clips in an adjacent pair must pass the filter.
Multiple paths in environment variables are separated by `:` on Unix-like
systems and `;` on Windows.

## Data rights

No training audio is included. Users are responsible for acquiring datasets
and processing only material for which they have the required rights.
