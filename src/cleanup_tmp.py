"""
Delete the tmp/<video_name>/ working directory once the pipeline has
completed, if config["cleanup"]["delete_tmp_after_run"] is true.

Intended to run last in the chain, after composite_and_stitch.py has
written the final output to Output_Videos/. Does nothing (and does not
error) if the flag is false - safe to always include in submit_pipeline.sh.
"""

import argparse
import json
import shutil
from pathlib import Path


def cleanup_tmp(config: dict):
    video_name = config["video_name"]
    paths = config["paths"]
    delete_after_run = config.get("cleanup", {}).get("delete_tmp_after_run", False)

    video_tmp = Path(paths["tmp_dir"]) / video_name

    if not delete_after_run:
        print(f"cleanup.delete_tmp_after_run is false - leaving {video_tmp} in place.")
        return

    if not video_tmp.exists():
        print(f"{video_tmp} does not exist - nothing to clean up.")
        return

    shutil.rmtree(video_tmp)
    print(f"Deleted {video_tmp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete tmp/<video_name>/ if configured to, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    cleanup_tmp(config)