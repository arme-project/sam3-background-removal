"""
Bitwise-OR merge per-frame masks across all prompt folders in
segmented_frames/, driven by config["prompts"] (not hardcoded).

Adding/removing/renaming prompts in config.json is picked up automatically -
no code changes needed here.
"""

import numpy as np
from PIL import Image

from paths import merge_dir, segment_dir


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


def merge_masks(config: dict):
    """OR-merge masks across all prompt folders, one folder per prompt at its
    own configured (threshold, mask_threshold).

    Reads: paths.segment_dir(config, prompt)/*.png  for each prompt in config["prompts"]
    Writes: paths.merge_dir(config)/*.png

    A frame is merged only if it exists in at least one prompt folder; prompts
    missing that particular frame are simply skipped for it (rather than
    failing), since segmentation confidence/coverage can differ by prompt.
    Returns the merge output directory.
    """
    prompts = config["prompts"]
    merged_dir = merge_dir(config)
    merged_dir.mkdir(parents=True, exist_ok=True)

    prompt_dirs = {}
    for prompt in prompts:
        prompt_dir = segment_dir(config, prompt)
        if not prompt_dir.is_dir():
            print(f"Warning: no segmented_frames folder found for prompt '{prompt}' "
                  f"(expected {prompt_dir}) - skipping this prompt.")
            continue
        prompt_dirs[prompt] = prompt_dir

    if not prompt_dirs:
        raise RuntimeError(
            f"No prompt folders found for prompts: {list(prompts.keys())}. "
            f"Run segment_frame.py first."
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
    import argparse
    import json

    from paths import decoded_frame_count, is_stage_done

    parser = argparse.ArgumentParser(description="OR-merge per-prompt masks, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    output_dir = merge_dir(config)
    if is_stage_done(output_dir, expected_count=decoded_frame_count(config)):
        print(f"Merge already done at {output_dir}, skipping.")
    else:
        merge_masks(config)