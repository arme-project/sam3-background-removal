"""
Fill small enclosed holes in binary masks, leaving larger holes
(e.g. violin f-holes) untouched (foreground = white/255, background = black/0).
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """
    Fill enclosed background holes with area <= max_hole_area.
    Holes touching the image border are never filled (they're not enclosed).
    """
    binary = mask > 0
    background = ~binary

    structure = np.ones((3, 3), dtype=int)
    labeled, num_components = label(background, structure=structure)  # type: ignore

    if num_components == 0:
        return mask

    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | \
                    set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)

    sizes = np.bincount(labeled.ravel())

    filled = mask.copy()
    fg_value = mask.max() if mask.max() > 0 else 255

    for comp_label in range(1, num_components + 1):
        if comp_label in border_labels:
            continue
        if sizes[comp_label] <= max_hole_area:
            filled[labeled == comp_label] = fg_value

    return filled


def process_folder(input_dir: Path, output_dir: Path, max_hole_area: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No .png files found in {input_dir}")
        return

    for path in png_files:
        img = Image.open(path).convert("L")
        mask = np.array(img)

        filled = fill_small_holes(mask, max_hole_area)
        n_filled_pixels = int(np.sum(filled > 0) - np.sum(mask > 0))  # type: ignore

        Image.fromarray(filled).save(output_dir / path.name)

        if n_filled_pixels > 0:
            print(f"{path.name}: filled {n_filled_pixels} pixels")

    print(f"\nProcessed {len(png_files)} frames (max_hole_area={max_hole_area}). Output in {output_dir}")


if __name__ == "__main__":
    import argparse
    import json

    from paths import decoded_frame_count, is_stage_done, postprocess_dir

    parser = argparse.ArgumentParser(description="Fill small holes, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    input_dir = postprocess_dir(config, "remove_fg_noise")
    output_dir = postprocess_dir(config, "fill_small_holes")

    max_hole_area = config["postprocess"]["fill_small_holes"]["max_hole_area"]

    if is_stage_done(output_dir, expected_count=decoded_frame_count(config)):
        print(f"fill_small_holes already done at {output_dir}, skipping.")
    else:
        process_folder(input_dir, output_dir, max_hole_area)