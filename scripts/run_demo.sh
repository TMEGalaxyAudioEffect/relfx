#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 REFERENCE_AUDIO PROCESSED_AUDIO [OUTPUT_NPY]" >&2
  exit 2
fi

: "${RELFX_CHECKPOINT_PATH:?Set RELFX_CHECKPOINT_PATH to an approved checkpoint}"

reference_audio="$1"
processed_audio="$2"
output_path="${3:-outputs/demo/embedding.npy}"

mkdir -p "$(dirname "$output_path")"

python scripts/embed.py \
  --checkpoint "$RELFX_CHECKPOINT_PATH" \
  --reference "$reference_audio" \
  --processed "$processed_audio" \
  --output "$output_path"
