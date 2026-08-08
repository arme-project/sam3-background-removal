"""
Delete only the tmp/ folders belonging to the current config's specific
param combo, once the pipeline has completed, if
config["cleanup"]["delete_tmp_after_run"] is true.

Deletes:
    - segmented_frames/<prompt>/<this combo's t/mt params>/  for each prompt
    - merged_frames/<this config's merge_key>/  (which recursively includes
      the entire postprocess chain nested under it - remove_fg_noise/,
      fill_small_holes/, etc.)

Does NOT delete:
    - decoded_frames/  - shared across all param combos for this video, so
      deleting it here would break other combos still relying on it
    - other prompts'/combos' segmented_frames or merged_frames folders -
      this is scoped to only what the current config touched

Intended to run last in the chain, after composite_and_stitch.py has
written the final output to Output_Videos/. Does nothing (and does not
error) if the flag is false - safe to always include in submit_pipeline.sh.
"""

import argparse
import json
import shutil

from paths import merge_base, segment_dir


def cleanup_tmp(config: dict):
    delete_after_run = config.get("cleanup", {}).get("delete_tmp_after_run", False)

    if not delete_after_run:
        print("cleanup.delete_tmp_after_run is false - leaving tmp/ in place.")
        return

    prompts = config["prompts"]
    for prompt in prompts:
        prompt_dir = segment_dir(config, prompt)
        if prompt_dir.exists():
            shutil.rmtree(prompt_dir)
            print(f"Deleted {prompt_dir}")
        else:
            print(f"{prompt_dir} does not exist - nothing to clean up for prompt '{prompt}'.")

    combo_dir = merge_base(config)
    if combo_dir.exists():
        shutil.rmtree(combo_dir)
        print(f"Deleted {combo_dir}")
    else:
        print(f"{combo_dir} does not exist - nothing to clean up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete this config's param combo from tmp/ if configured to, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    cleanup_tmp(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete tmp/<video_name>/ if configured to, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    cleanup_tmp(config)