"""
Remove foreground noise from binary masks by keeping only the largest
connected component (foreground = white/255, background = black/0).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def remove_fg_noise(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected foreground component.

    mask: 2D array, foreground = nonzero (255), background = 0
    returns: cleaned mask, same dtype/shape as input
    """
    binary = mask > 0

    structure = np.ones((3, 3), dtype=int)
    labeled, num_components = label(binary, structure=structure)  # type: ignore

    if num_components <= 1:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)

    cleaned = np.where(labeled == largest_label, mask, 0).astype(mask.dtype)
    return cleaned


def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No .png files found in {input_dir}")
        return

    for path in png_files:
        img = Image.open(path).convert("L")
        mask = np.array(img)

        before = label(mask > 0, structure=np.ones((3, 3), dtype=int))[1]  # type: ignore
        cleaned = remove_fg_noise(mask)
        after = label(cleaned > 0, structure=np.ones((3, 3), dtype=int))[1]  # type: ignore

        Image.fromarray(cleaned).save(output_dir / path.name)

        if before != after:
            print(f"{path.name}: {before} components -> {after} (removed {before - after})")

    print(f"\nProcessed {len(png_files)} frames. Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove foreground noise, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_tmp = Path(config["paths"]["tmp_dir"]) / config["video_name"]
    input_dir = video_tmp / "merged_frames"
    output_dir = video_tmp / "postprocessed_frames" / "01_remove_fg_noise"

    process_folder(input_dir, output_dir)