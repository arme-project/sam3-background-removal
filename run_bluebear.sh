#!/usr/bin/env bash
# Submits the full SAM3 background removal pipeline as three chained
# Slurm jobs on BlueBEAR: decode (CPU) -> segment (GPU) -> postprocess (CPU).
#
# Usage:
#   ./run_bluebear.sh [--config path/to/config.json]
#
# Default: config.json

set -e

CONFIG="config.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p logs

echo "Using config: $CONFIG"

DECODE_ID=$(sbatch --parsable jobs/decode.sh "$CONFIG")
echo "Submitted decode job: $DECODE_ID"

SEGMENT_ID=$(sbatch --parsable --dependency=afterok:$DECODE_ID jobs/segment.sh "$CONFIG")
echo "Submitted segment job: $SEGMENT_ID (after $DECODE_ID)"

POSTPROCESS_ID=$(sbatch --parsable --dependency=afterok:$SEGMENT_ID jobs/postprocess.sh "$CONFIG")
echo "Submitted postprocess job: $POSTPROCESS_ID (after $SEGMENT_ID)"

echo
echo "Pipeline submitted. Track with: squeue -u \$USER"