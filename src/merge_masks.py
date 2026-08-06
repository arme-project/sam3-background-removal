"""
Bitwise-OR merge per-frame masks across all prompt folders in
segmented_frames/, driven by config["prompts"] (not hardcoded).

Adding/removing/renaming prompts in config.json is picked up automatically -
no code changes needed here.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def merge_frame(mask_paths):
    """OR-merge a list of single-frame mask paths into one binary uint8 mask."""
    if not mask_paths:
        raise ValueError("merge_frame() called with no mask paths.")

    merged = None
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask > 0
        merged = mask if merged is None else (merged | mask)

    assert merged is not None
    return (merged * 255).astype(np.uint8)


def merge_masks(video_tmp: Path, prompts: list[str]):
    """OR-merge masks across all prompt subfolders under segmented_frames/.

    Reads: video_tmp/segmented_frames/<prompt>/*.png  for each prompt in `prompts`
    Writes: video_tmp/merged_frames/*.png

    A frame is merged only if it exists in at least one prompt folder; prompts
    missing that particular frame are simply skipped for it (rather than
    failing), since segmentation confidence/coverage can differ by prompt.
    Returns the merged_frames directory.
    """
    segmented_root = video_tmp / "segmented_frames"
    merged_dir = video_tmp / "merged_frames"
    merged_dir.mkdir(parents=True, exist_ok=True)

    prompt_dirs = {}
    for prompt in prompts:
        prompt_dir = segmented_root / prompt
        if not prompt_dir.is_dir():
            print(f"Warning: no segmented_frames folder found for prompt '{prompt}' "
                  f"(expected {prompt_dir}) - skipping this prompt.")
            continue
        prompt_dirs[prompt] = prompt_dir

    if not prompt_dirs:
        raise RuntimeError(
            f"No prompt folders found under {segmented_root}. "
            f"Expected one subfolder per prompt in config['prompts']: {prompts}"
        )

    # Union of frame filenames across all found prompt folders, so a frame
    # missing from one prompt (e.g. bow undetected in that frame) doesn't
    # block merging the others.
    all_frames = set()
    for prompt_dir in prompt_dirs.values():
        all_frames.update(
            f.name for f in prompt_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    frame_files = sorted(all_frames)
    print(f"Found {len(frame_files)} frames across {len(prompt_dirs)} prompt folders: "
          f"{list(prompt_dirs.keys())}")

    for i, frame_file in enumerate(frame_files):
        mask_paths = [
            prompt_dir / frame_file
            for prompt_dir in prompt_dirs.values()
            if (prompt_dir / frame_file).exists()
        ]
        merged = merge_frame(mask_paths)
        Image.fromarray(merged).save(merged_dir / frame_file)

        if (i + 1) % 100 == 0:
            print(f"Merged {i + 1}/{len(frame_files)} frames.")

    print("Finished.")
    return merged_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OR-merge per-prompt masks, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_name = config["video_name"]
    paths = config["paths"]
    prompts = config["prompts"]

    video_tmp = Path(paths["tmp_dir"]) / video_name

    merge_masks(video_tmp, prompts)