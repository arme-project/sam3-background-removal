"""
Apply binary morphological erosion to binary masks. Shrinks the foreground
boundary inward, trimming thin protrusions (foreground = white/255,
background = black/0).

Note: erosion_iterations should stay low (e.g. 1) - high iteration counts
are destructive to thin structures like the bow.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def apply_erosion(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """
    Apply binary erosion with a square structuring element of side kernel_size,
    repeated `iterations` times.

    mask: 2D array, foreground = nonzero (255), background = 0
    returns: eroded mask, same dtype/shape as input
    """
    binary = mask > 0
    fg_value = mask.max() if mask.max() > 0 else 255

    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    eroded = binary_erosion(binary, structure=structure, iterations=iterations)

    return (eroded * fg_value).astype(mask.dtype)


def process_folder(input_dir: Path, output_dir: Path, kernel_size: int, iterations: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No .png files found in {input_dir}")
        return

    for path in png_files:
        img = Image.open(path).convert("L")
        mask = np.array(img)

        eroded = apply_erosion(mask, kernel_size, iterations)
        Image.fromarray(eroded).save(output_dir / path.name)

    print(f"\nProcessed {len(png_files)} frames "
          f"(kernel_size={kernel_size}, iterations={iterations}). Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply binary erosion, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_tmp = Path(config["paths"]["tmp_dir"]) / config["video_name"]
    input_dir = video_tmp / "postprocessed_frames" / "03_opening"
    output_dir = video_tmp / "postprocessed_frames" / "04_erosion"

    erosion_cfg = config["postprocess"]["erosion"]
    process_folder(input_dir, output_dir, erosion_cfg["kernel_size"], erosion_cfg["iterations"])