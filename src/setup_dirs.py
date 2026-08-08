import argparse
import json
from pathlib import Path

from paths import (
    decoded_frames_dir,
    merge_dir,
    segment_dir,
)


def setup_dirs(config: dict) -> Path:
    """Create the nested, param-aware tmp/<video_name>/... scaffold plus
    Input_Videos/ and Output_Videos/.

    Directory structure created (see paths.py for the full shape/rationale):
        tmp/<video_name>/decoded_frames/
        tmp/<video_name>/segmented_frames/<prompt>/t{th}_mt{mth}/   (per prompt in config["prompts"])
        tmp/<video_name>/merged_frames/<merge_key>/
        <input_dir>/
        <output_dir>/

    Deliberately does NOT pre-create the postprocess chain
    (remove_fg_noise/fill_small_holes/opening/erosion/smooth_gaussian/) -
    each of those stage scripts creates its own output folder on demand when
    it actually runs (via process_folder()'s own mkdir). Pre-creating them
    here would leave empty scaffold folders that is_stage_done() could later
    be confused by, and there's no benefit to having them exist before their
    stage has actually produced anything.

    Only creates the folders for the *current* config's param combo - other
    combos' folders (from past runs with different params) are left alone,
    which is exactly the reuse/coexistence behavior the nested tree is for.

    Returns the video's tmp root (tmp/<video_name>).
    """
    paths = config["paths"]
    prompts = config["prompts"]

    decoded_frames_dir(config).mkdir(parents=True, exist_ok=True)

    for prompt in prompts:
        segment_dir(config, prompt).mkdir(parents=True, exist_ok=True)

    merge_dir(config).mkdir(parents=True, exist_ok=True)

    Path(paths["input_dir"]).mkdir(exist_ok=True)
    Path(paths["output_dir"]).mkdir(exist_ok=True)

    video_tmp = decoded_frames_dir(config).parent
    return video_tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tmp/ directory scaffold for a video, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_tmp = setup_dirs(config)
    print(f"Created directory structure at {video_tmp}")