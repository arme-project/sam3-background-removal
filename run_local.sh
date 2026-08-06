#!/usr/bin/env bash
# Runs the full SAM3 background removal pipeline sequentially, locally (macOS/MPS).
#
# Usage:
#   ./run_local.sh [--config path/to/config.json] [--from-stage STAGE]
#
# STAGE is one of: setup, decode, segment, merge, remove_fg_noise,
#                   fill_small_holes, opening, erosion, smooth_gaussian,
#                   composite, cleanup
# Defaults: config.json, from-stage setup (i.e. runs everything).
#
# Example - already have segmented masks, just want to retune opening/erosion:
#   ./run_local.sh --from-stage opening

set -e

CONFIG="config.json"
FROM_STAGE="setup"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

STAGES=(setup decode segment merge remove_fg_noise fill_small_holes opening erosion smooth_gaussian composite cleanup)

# Validate --from-stage and find its index
START_INDEX=-1
for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$FROM_STAGE" ]]; then
        START_INDEX=$i
        break
    fi
done
if [[ $START_INDEX -eq -1 ]]; then
    echo "Unknown stage: $FROM_STAGE"
    echo "Valid stages: ${STAGES[*]}"
    exit 1
fi

echo "Using config: $CONFIG"
echo "Starting from stage: $FROM_STAGE"
echo

run_stage() {
    local name="$1"
    local script="$2"
    echo "== $name =="
    python "$script" --config "$CONFIG"
}

for i in "${!STAGES[@]}"; do
    if [[ $i -lt $START_INDEX ]]; then
        continue
    fi
    case "${STAGES[$i]}" in
        setup)             run_stage "setup_dirs"        "src/setup_dirs.py" ;;
        decode)             run_stage "decode_video"       "src/decode_video.py" ;;
        segment)             run_stage "segment_frame"      "src/segment_frame.py" ;;
        merge)               run_stage "merge_masks"        "src/merge_masks.py" ;;
        remove_fg_noise)     run_stage "remove_fg_noise"    "src/remove_fg_noise.py" ;;
        fill_small_holes)    run_stage "fill_small_holes"   "src/fill_small_holes.py" ;;
        opening)             run_stage "opening"            "src/opening.py" ;;
        erosion)             run_stage "erosion"             "src/erosion.py" ;;
        smooth_gaussian)     run_stage "smooth_gaussian"    "src/smooth_gaussian.py" ;;
        composite)           run_stage "composite_and_stitch" "src/composite_and_stitch.py" ;;
        cleanup)             run_stage "cleanup_tmp"        "src/cleanup_tmp.py" ;;
    esac
done

echo
echo "Pipeline complete."