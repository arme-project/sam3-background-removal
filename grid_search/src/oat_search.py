"""
One-at-a-time (OAT) sweep over segmentation threshold/mask_threshold, per
prompt. Unlike grid_search.py/joint_grid_search.py (full joint/factorial -
every combination of every parameter), this holds ALL parameters at a fixed
baseline except one, sweeps that one across a range of test values, then
moves to the next parameter - so cost grows LINEARLY with the number of
values tested, not multiplicatively.

Tradeoff: OAT can't catch interactions between parameters (e.g. "violin
mask_threshold=0.5 is only good when person mask_threshold=0.3") - it
assumes the effect of each parameter is independent of the others' values.
That's a real limitation, but a reasonable one once a joint sweep has
already shown a given parameter has little interaction with the rest (as
your medium sweep's postprocess-vs-segmentation analysis is checking for).

Use this to explore a WIDER range per parameter than a joint sweep could
practically afford, now that you have a baseline worth exploring around.

Baseline + sweep ranges come from a JSON config, e.g.:
    {
      "baseline": {
        "person": {"threshold": 0.3, "mask_threshold": 0.3},
        "violin": {"threshold": 0.3, "mask_threshold": 0.3},
        "violin_bow": {"threshold": 0.3, "mask_threshold": 0.3}
      },
      "sweep_values": {
        "person": {"threshold": [0.1,0.15,0.2,...], "mask_threshold": [0.1,0.15,...]},
        "violin": {"threshold": [...], "mask_threshold": [...]},
        "violin_bow": {"threshold": [...], "mask_threshold": [...]}
      }
    }

For each (prompt, param) pair in sweep_values, every listed value is tried
with all other prompt/param values held at baseline - one CSV row per test
value. The baseline combo itself is included once at the top, labelled
accordingly, so you can see the reference point alongside every sweep.

Supports the same multi-frame manifest as grid_search.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from compute_iou import compute_metrics
from segment_frame import get_device, load_sam3, segment_frame


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a multi-frame manifest JSON (see grid_search.py for format).
    Paths resolved relative to the manifest file's own directory."""
    with open(manifest_path) as f:
        entries = json.load(f)
    manifest_dir = manifest_path.parent
    return [
        {"name": e["name"], "frame": manifest_dir / e["frame"], "gt_mask": manifest_dir / e["gt_mask"]}
        for e in entries
    ]


def merge_with_params(image, params: dict, model, processor, device) -> np.ndarray:
    """OR-merge every prompt's segmentation at the given per-prompt params."""
    merged = None
    for prompt, p in params.items():
        mask = segment_frame(
            image, prompt, model, processor, device,
            threshold=p["threshold"], mask_threshold=p["mask_threshold"],
        )
        mask_bool = mask > 0
        merged = mask_bool if merged is None else (merged | mask_bool)
    assert merged is not None
    return (merged * 255).astype(np.uint8)


def build_oat_plan(baseline: dict, sweep_values: dict) -> list[dict]:
    """
    Returns a list of test points, each a dict:
        {"varying_prompt": str, "varying_param": str, "varying_value": float,
         "params": {prompt: {"threshold": x, "mask_threshold": y}, ...}}
    plus one entry at the start with varying_prompt=None for the baseline itself.
    """
    plan = [{
        "varying_prompt": None, "varying_param": None, "varying_value": None,
        "params": {p: dict(baseline[p]) for p in baseline},
    }]

    for prompt in sweep_values:
        for param_name in sweep_values[prompt]:
            for value in sweep_values[prompt][param_name]:
                params = {p: dict(baseline[p]) for p in baseline}
                params[prompt] = dict(params[prompt])
                params[prompt][param_name] = value
                plan.append({
                    "varying_prompt": prompt, "varying_param": param_name, "varying_value": value,
                    "params": params,
                })
    return plan


def run_oat_search(frames: list[dict], baseline: dict, sweep_values: dict, output_csv: Path):
    plan = build_oat_plan(baseline, sweep_values)

    total_tests = sum(len(sweep_values[p][k]) for p in sweep_values for k in sweep_values[p])
    print(f"Baseline: {baseline}")
    print(f"OAT test points: {total_tests} (+ 1 baseline row) across {len(frames)} frame(s)")
    print(f"SAM3 forward passes: {len(plan) * len(frames)} (linear in test points, not multiplicative)")
    sys.stdout.flush()

    loaded_frames = []
    for entry in frames:
        image = Image.open(entry["frame"]).convert("RGB")
        gt_mask = np.array(Image.open(entry["gt_mask"]).convert("L"))
        loaded_frames.append({"name": entry["name"], "image": image, "gt_mask": gt_mask})

    device = get_device()
    print(f"Using device: {device}")
    sys.stdout.flush()
    model, processor = load_sam3(device)

    prompts = list(baseline.keys())
    fieldnames = ["varying_prompt", "varying_param", "varying_value"]
    for prompt in prompts:
        fieldnames += [f"{prompt}_threshold", f"{prompt}_mask_threshold"]
    for lf in loaded_frames:
        fieldnames += [f"iou_{lf['name']}"]
    fieldnames += ["mean_iou", "mean_precision", "mean_recall", "mean_f0.5"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for i, point in enumerate(plan):
            row = {
                "varying_prompt": point["varying_prompt"] or "baseline",
                "varying_param": point["varying_param"] or "",
                "varying_value": point["varying_value"] if point["varying_value"] is not None else "",
            }
            for prompt in prompts:
                row[f"{prompt}_threshold"] = point["params"][prompt]["threshold"]
                row[f"{prompt}_mask_threshold"] = point["params"][prompt]["mask_threshold"]

            per_frame_metrics = []
            for lf in loaded_frames:
                merged_mask = merge_with_params(lf["image"], point["params"], model, processor, device)
                metrics = compute_metrics(merged_mask, lf["gt_mask"])
                per_frame_metrics.append(metrics)
                row[f"iou_{lf['name']}"] = round(metrics["iou"], 4)

            row["mean_iou"] = round(float(np.mean([m["iou"] for m in per_frame_metrics])), 4)
            row["mean_precision"] = round(float(np.mean([m["precision"] for m in per_frame_metrics])), 4)
            row["mean_recall"] = round(float(np.mean([m["recall"] for m in per_frame_metrics])), 4)
            row["mean_f0.5"] = round(float(np.mean([m["f0.5"] for m in per_frame_metrics])), 4)
            writer.writerow(row)
            f.flush()

            print(f"[{i + 1}/{len(plan)}] {row}")
            sys.stdout.flush()

    print(f"\nDone. Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OAT sweep over segmentation threshold/mask_threshold, one parameter at a time, "
                     "all others held at baseline."
    )
    parser.add_argument("--manifest", default=None, help="Path to a JSON manifest of frames")
    parser.add_argument("--frame", default=None, help="Path to a single frame (shorthand for a one-entry manifest)")
    parser.add_argument("--gt-mask", default=None, help="Path to a single ground-truth mask (used with --frame)")
    parser.add_argument("--oat-config", required=True,
                         help='Path to JSON with "baseline" and "sweep_values" sections (see module docstring)')
    parser.add_argument("--output", default="oat_search_results.csv", help="Path to write results CSV")
    args = parser.parse_args()

    if args.manifest and (args.frame or args.gt_mask):
        parser.error("--manifest and --frame/--gt-mask are mutually exclusive.")
    if not args.manifest and not (args.frame and args.gt_mask):
        parser.error("Provide either --manifest, or both --frame and --gt-mask.")

    if args.manifest:
        frames = load_manifest(Path(args.manifest))
    else:
        frames = [{"name": Path(args.frame).stem, "frame": Path(args.frame), "gt_mask": Path(args.gt_mask)}]

    with open(args.oat_config) as f:
        oat_config = json.load(f)

    run_oat_search(frames, oat_config["baseline"], oat_config["sweep_values"], Path(args.output))