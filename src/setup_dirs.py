import argparse
import json
from pathlib import Path


def setup_dirs(config: dict) -> Path:
    """Create the tmp/<video_name>/... scaffold plus Input_Videos/ and Output_Videos/.

    Directory structure created:
        tmp/<video_name>/decoded_frames/
        tmp/<video_name>/segmented_frames/<prompt>/   (one per prompt in config["prompts"])
        tmp/<video_name>/merged_frames/
        tmp/<video_name>/postprocessed_frames/01_remove_fg_noise/
        tmp/<video_name>/postprocessed_frames/02_fill_small_holes/
        tmp/<video_name>/postprocessed_frames/03_opening/
        tmp/<video_name>/postprocessed_frames/04_erosion/
        tmp/<video_name>/postprocessed_frames/05_smooth_gaussian/
        <input_dir>/
        <output_dir>/

    Returns the video's tmp root (tmp/<video_name>).
    """
    video_name = config["video_name"]
    prompts = config["prompts"]
    paths = config["paths"]

    video_tmp = Path(paths["tmp_dir"]) / video_name

    (video_tmp / "decoded_frames").mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        (video_tmp / "segmented_frames" / prompt).mkdir(parents=True, exist_ok=True)
    (video_tmp / "merged_frames").mkdir(parents=True, exist_ok=True)

    postprocess_stages = [
        "01_remove_fg_noise",
        "02_fill_small_holes",
        "03_opening",
        "04_erosion",
        "05_smooth_gaussian",
    ]
    for stage in postprocess_stages:
        (video_tmp / "postprocessed_frames" / stage).mkdir(parents=True, exist_ok=True)

    Path(paths["input_dir"]).mkdir(exist_ok=True)
    Path(paths["output_dir"]).mkdir(exist_ok=True)

    return video_tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tmp/ directory scaffold for a video, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_tmp = setup_dirs(config)
    print(f"Created directory structure at {video_tmp}")