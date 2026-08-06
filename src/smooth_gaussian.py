"""
Apply Gaussian blur + rethreshold to binary masks as a light smoothing pass
on the foreground boundary (foreground = white/255, background = black/0).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def smooth_gaussian(mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Blur the binary mask with a Gaussian kernel, then rethreshold at 0.5.

    mask: 2D array, foreground = nonzero (255), background = 0
    returns: smoothed mask, same dtype/shape as input
    """
    fg_value = mask.max() if mask.max() > 0 else 255
    binary_float = (mask > 0).astype(np.float32)
    blurred = gaussian_filter(binary_float, sigma=sigma)
    return (blurred >= 0.5).astype(mask.dtype) * fg_value


def process_folder(input_dir: Path, output_dir: Path, sigma: float):
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No .png files found in {input_dir}")
        return

    for path in png_files:
        img = Image.open(path).convert("L")
        mask = np.array(img)

        smoothed = smooth_gaussian(mask, sigma)
        Image.fromarray(smoothed).save(output_dir / path.name)

    print(f"\nProcessed {len(png_files)} frames (sigma={sigma}). Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian smoothing, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_tmp = Path(config["paths"]["tmp_dir"]) / config["video_name"]
    input_dir = video_tmp / "postprocessed_frames" / "04_erosion"
    output_dir = video_tmp / "postprocessed_frames" / "05_smooth_gaussian"

    sigma = config["postprocess"]["smooth_gaussian"]["sigma"]
    process_folder(input_dir, output_dir, sigma)