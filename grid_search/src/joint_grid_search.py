"""
Joint grid search across BOTH segmentation params (threshold, mask_threshold
per prompt) AND postprocess morphological params (opening, erosion), to
check for tradeoffs/interactions between the two stages rather than tuning
each in isolation against the other's default settings.

Structured to avoid re-running SAM3 unnecessarily, since segmentation is the
expensive part and postprocessing is cheap:

    for each segmentation combo:
        run SAM3 once per frame -> raw merged mask (no postprocess yet)
        for each postprocess combo:
            apply opening, then erosion, to that raw mask
            score against ground truth
            log one row: [segmentation params, postprocess params, per-frame
                          IoU, mean iou/precision/recall/f0.5]

So total SAM3 calls = (segmentation combos) x (frames) - NOT multiplied by
postprocess combos. Postprocess combos are just cheap array ops on top.

Supports the same multi-frame manifest as grid_search.py:
    [{"name": "violin_01", "frame": "...", "gt_mask": "..."}, ...]
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compute_iou import compute_metrics
from erosion import apply_erosion
from opening import apply_opening
from segment_frame import get_device, load_sam3, segment_frame


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a multi-frame manifest JSON (see module docstring). Paths are
    resolved relative to the manifest file's own directory."""
    with open(manifest_path) as f:
        entries = json.load(f)
    manifest_dir = manifest_path.parent
    return [
        {"name": e["name"], "frame": manifest_dir / e["frame"], "gt_mask": manifest_dir / e["gt_mask"]}
        for e in entries
    ]


def build_segmentation_grid(seg_param_grid: dict) -> list[dict]:
    """Same joint cartesian product as grid_search.py's build_grid():
    seg_param_grid: {prompt_name: {"threshold": [...], "mask_threshold": [...]}}
    Returns a list of {prompt: {"threshold": x, "mask_threshold": y}, ...} combos."""
    prompt_names = list(seg_param_grid.keys())
    per_prompt_combos = {
        prompt: list(itertools.product(seg_param_grid[prompt]["threshold"],
                                        seg_param_grid[prompt]["mask_threshold"]))
        for prompt in prompt_names
    }
    all_combos = itertools.product(*(per_prompt_combos[p] for p in prompt_names))

    grid = []
    for combo in all_combos:
        entry = {}
        for prompt, (threshold, mask_threshold) in zip(prompt_names, combo):
            entry[prompt] = {"threshold": threshold, "mask_threshold": mask_threshold}
        grid.append(entry)
    return grid


def build_postprocess_grid(postprocess_param_grid: dict) -> list[dict]:
    """
    postprocess_param_grid: {"opening": {"kernel_size": [...], "iterations": [...]},
                              "erosion": {"kernel_size": [...], "iterations": [...]}}
    Returns a list of {"opening": {"kernel_size": x, "iterations": y},
                        "erosion": {"kernel_size": x, "iterations": y}} combos.
    """
    opening_combos = list(itertools.product(
        postprocess_param_grid["opening"]["kernel_size"],
        postprocess_param_grid["opening"]["iterations"],
    ))
    erosion_combos = list(itertools.product(
        postprocess_param_grid["erosion"]["kernel_size"],
        postprocess_param_grid["erosion"]["iterations"],
    ))

    grid = []
    for (o_kernel, o_iter), (e_kernel, e_iter) in itertools.product(opening_combos, erosion_combos):
        grid.append({
            "opening": {"kernel_size": o_kernel, "iterations": o_iter},
            "erosion": {"kernel_size": e_kernel, "iterations": e_iter},
        })
    return grid


def merge_segmentation_combo(image, prompt_params: dict, model, processor, device) -> np.ndarray:
    """Run segment_frame() for every prompt at this combo's params, OR-merge
    into one raw binary mask (before any postprocessing)."""
    merged = None
    for prompt, params in prompt_params.items():
        mask = segment_frame(
            image, prompt, model, processor, device,
            threshold=params["threshold"],
            mask_threshold=params["mask_threshold"],
        )
        mask_bool = mask > 0
        merged = mask_bool if merged is None else (merged | mask_bool)
    assert merged is not None
    return (merged * 255).astype(np.uint8)


def apply_postprocess_combo(raw_mask: np.ndarray, postprocess_params: dict) -> np.ndarray:
    """Opening then erosion (pipeline order), using the same apply_opening()/
    apply_erosion() functions the real pipeline stages call.

    Note: scipy's binary_opening/binary_erosion treat iterations=0 as "repeat
    until the mask stops changing" (NOT "apply zero times") - left unguarded,
    that silently erases the mask rather than skipping the stage. So
    iterations=0 here is treated as an explicit no-op/skip instead.
    """
    mask = raw_mask
    if postprocess_params["opening"]["iterations"] > 0:
        mask = apply_opening(
            mask,
            postprocess_params["opening"]["kernel_size"],
            postprocess_params["opening"]["iterations"],
        )
    if postprocess_params["erosion"]["iterations"] > 0:
        mask = apply_erosion(
            mask,
            postprocess_params["erosion"]["kernel_size"],
            postprocess_params["erosion"]["iterations"],
        )
    return mask


def run_joint_grid_search(frames: list[dict], seg_param_grid: dict, postprocess_param_grid: dict,
                           output_csv: Path):
    seg_grid = build_segmentation_grid(seg_param_grid)
    postprocess_grid = build_postprocess_grid(postprocess_param_grid)

    total_sam3_calls = len(seg_grid) * len(frames)
    total_rows = len(seg_grid) * len(postprocess_grid)
    print(f"Segmentation combos: {len(seg_grid)}, postprocess combos: {len(postprocess_grid)}, frames: {len(frames)}")
    print(f"SAM3 forward passes: {total_sam3_calls} (segmentation combos x frames - NOT multiplied by postprocess combos)")
    print(f"Total CSV rows: {total_rows} (segmentation combos x postprocess combos)")
    if total_rows > 2000:
        print(f"WARNING: {total_rows} rows is a lot - consider narrowing the postprocess or segmentation grids.")

    loaded_frames = []
    for entry in frames:
        image = Image.open(entry["frame"]).convert("RGB")
        gt_mask = np.array(Image.open(entry["gt_mask"]).convert("L"))
        loaded_frames.append({"name": entry["name"], "image": image, "gt_mask": gt_mask})

    device = get_device()
    print(f"Using device: {device}")
    model, processor = load_sam3(device)

    fieldnames = []
    for prompt in seg_param_grid:
        fieldnames += [f"{prompt}_threshold", f"{prompt}_mask_threshold"]
    fieldnames += ["opening_kernel_size", "opening_iterations", "erosion_kernel_size", "erosion_iterations"]
    for lf in loaded_frames:
        fieldnames += [f"iou_{lf['name']}"]
    fieldnames += ["mean_iou", "mean_precision", "mean_recall", "mean_f0.5"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row_num = 0
        for seg_i, prompt_params in enumerate(seg_grid):
            # Expensive step - once per segmentation combo, reused across every postprocess combo below.
            raw_masks = {}
            for lf in loaded_frames:
                raw_masks[lf["name"]] = merge_segmentation_combo(lf["image"], prompt_params, model, processor, device)

            for postprocess_params in postprocess_grid:
                row = {}
                for prompt, params in prompt_params.items():
                    row[f"{prompt}_threshold"] = params["threshold"]
                    row[f"{prompt}_mask_threshold"] = params["mask_threshold"]
                row["opening_kernel_size"] = postprocess_params["opening"]["kernel_size"]
                row["opening_iterations"] = postprocess_params["opening"]["iterations"]
                row["erosion_kernel_size"] = postprocess_params["erosion"]["kernel_size"]
                row["erosion_iterations"] = postprocess_params["erosion"]["iterations"]

                per_frame_metrics = []
                for lf in loaded_frames:
                    final_mask = apply_postprocess_combo(raw_masks[lf["name"]], postprocess_params)
                    metrics = compute_metrics(final_mask, lf["gt_mask"])
                    per_frame_metrics.append(metrics)
                    row[f"iou_{lf['name']}"] = round(metrics["iou"], 4)

                row["mean_iou"] = round(float(np.mean([m["iou"] for m in per_frame_metrics])), 4)
                row["mean_precision"] = round(float(np.mean([m["precision"] for m in per_frame_metrics])), 4)
                row["mean_recall"] = round(float(np.mean([m["recall"] for m in per_frame_metrics])), 4)
                row["mean_f0.5"] = round(float(np.mean([m["f0.5"] for m in per_frame_metrics])), 4)
                writer.writerow(row)

                row_num += 1
                print(f"[{row_num}/{total_rows}] (seg combo {seg_i + 1}/{len(seg_grid)}) {row}")

    print(f"\nDone. Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint grid search across segmentation AND postprocess (opening, erosion) params together."
    )
    parser.add_argument("--manifest", default=None,
                         help='Path to a JSON manifest of frames, e.g. [{"name": "...", "frame": "...", "gt_mask": "..."}, ...]')
    parser.add_argument("--frame", default=None, help="Path to a single frame (shorthand for a one-entry manifest)")
    parser.add_argument("--gt-mask", default=None, help="Path to a single ground-truth mask (used with --frame)")
    parser.add_argument("--grid-config", required=True,
                         help='Path to JSON with "segmentation" and "postprocess" sections, e.g. '
                              '{"segmentation": {"person": {"threshold": [0.3], "mask_threshold": [0.3,0.4]}, ...}, '
                              '"postprocess": {"opening": {"kernel_size": [3], "iterations": [1]}, '
                              '"erosion": {"kernel_size": [3], "iterations": [1,2]}}}')
    parser.add_argument("--output", default="joint_grid_search_results.csv", help="Path to write results CSV")
    args = parser.parse_args()

    if args.manifest and (args.frame or args.gt_mask):
        parser.error("--manifest and --frame/--gt-mask are mutually exclusive.")
    if not args.manifest and not (args.frame and args.gt_mask):
        parser.error("Provide either --manifest, or both --frame and --gt-mask.")

    if args.manifest:
        frames = load_manifest(Path(args.manifest))
    else:
        frames = [{"name": Path(args.frame).stem, "frame": Path(args.frame), "gt_mask": Path(args.gt_mask)}]

    with open(args.grid_config) as f:
        grid_config = json.load(f)

    run_joint_grid_search(
        frames, grid_config["segmentation"], grid_config["postprocess"], Path(args.output),
    )
