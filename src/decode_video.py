import argparse
import json
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg


def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_path,
        "-i", video_path,
        "-vsync", "0",
        os.path.join(frame_dir, "frame_%06d.png"),
    ], check=True)

    return frame_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video using FFmpeg, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_name = config["video_name"]
    paths = config["paths"]

    video_path = Path(paths["input_dir"]) / f"{video_name}.mov"
    frame_dir = Path(paths["tmp_dir"]) / video_name / "decoded_frames"

    extract_frames(str(video_path), str(frame_dir))