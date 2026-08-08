"""
Composite original color frames with binary masks into RGBA frames
(mask -> alpha channel), stitch into a ProRes 4444 .mov that preserves
transparency, then mux the original audio back in via stream copy.

fps is computed from the source video itself (exact average, via PyAV)
rather than hardcoded/rounded, to avoid audio/video drift - iPhone
footage is mildly VFR so a rounded fps value will drift over a long clip.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image

from paths import decoded_frames_dir, postprocess_dir, run_stamp


def get_exact_fps(video_path: Path) -> Fraction:
    """
    Compute the exact average fps of a video as a Fraction (e.g. 31008/517),
    using total frame count / total duration rather than the container's
    (possibly rounded) declared frame rate.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frame_count = stream.frames
    duration_seconds = float(stream.duration * stream.time_base)  # type: ignore

    if not frame_count:
        # Some containers don't populate stream.frames; fall back to counting.
        frame_count = sum(1 for _ in container.decode(stream))
        container.close()
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        duration_seconds = float(stream.duration * stream.time_base)  # type: ignore

    container.close()

    return Fraction(frame_count, 1) / Fraction(duration_seconds).limit_denominator(100000)


def composite_rgba(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    frame: HxWx3 RGB array
    mask: HxW array, foreground = nonzero, background = 0
    returns: HxWx4 RGBA array with mask as alpha channel
    """
    alpha = (mask > 0).astype(np.uint8) * 255
    rgba = np.dstack([frame, alpha])
    return rgba


def build_frame_pairs(frames_dir: Path, masks_dir: Path):
    """Match frames to masks by filename, erroring loudly on any mismatch."""
    frame_files = sorted(frames_dir.glob("*.png"))
    mask_files = {p.name: p for p in masks_dir.glob("*.png")}

    if not frame_files:
        raise FileNotFoundError(f"No .png files found in {frames_dir}")

    pairs = []
    missing = []
    for frame_path in frame_files:
        mask_path = mask_files.get(frame_path.name)
        if mask_path is None:
            missing.append(frame_path.name)
        else:
            pairs.append((frame_path, mask_path))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frame(s) have no matching mask, e.g.: {missing[:5]}"
        )

    return pairs


def stitch_video(pairs, silent_output_path: Path, fps: Fraction, codec: str):
    if codec not in ("prores", "vp9"):
        raise ValueError(f"Unknown codec: {codec}")

    writer = None
    first = True
    for i, (frame_path, mask_path) in enumerate(pairs):
        frame = np.array(Image.open(frame_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if frame.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Size mismatch at {frame_path.name}: frame {frame.shape[:2]} vs mask {mask.shape[:2]}"
            )

        rgba = composite_rgba(frame, mask)

        if first:
            h, w = rgba.shape[:2]
            writer = imageio_ffmpeg.write_frames(
                str(silent_output_path),
                size=(w, h),
                fps=float(fps),
                codec="prores_ks" if codec == "prores" else "libvpx-vp9",
                pix_fmt_in="rgba",
                pix_fmt_out="yuva444p10le" if codec == "prores" else "yuva420p",
                output_params=(["-profile:v", "4"] if codec == "prores"
                                else ["-b:v", "0", "-crf", "30"]),
            )
            writer.send(None)
            first = False

        writer.send(rgba.tobytes())  # type: ignore

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)} frames composited")

    writer.close()  # type: ignore
    print(f"\nWrote {len(pairs)} silent frames to {silent_output_path}")


def mux_audio(silent_video_path: Path, source_video_path: Path, final_output_path: Path):
    """
    Mux the audio track from source_video_path onto silent_video_path via
    stream copy (no re-encoding), writing final_output_path.
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    subprocess.run([
        ffmpeg_path,
        "-y",
        "-i", str(silent_video_path),
        "-i", str(source_video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(final_output_path),
    ], check=True)
    print(f"Muxed audio -> {final_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Composite frames+masks into an RGBA video, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--codec", type=str, choices=["prores", "vp9"], default="prores",
                         help="prores -> .mov (recommended, best NLE compatibility); "
                              "vp9 -> .webm (smaller, web-friendly)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_name = config["video_name"]
    paths = config["paths"]
    stamp = run_stamp(config)

    video_tmp = decoded_frames_dir(config).parent
    frames_dir = decoded_frames_dir(config)
    masks_dir = postprocess_dir(config, "smooth_gaussian")

    source_video_path = Path(paths["input_dir"]) / f"{video_name}.mov"
    final_output_path = Path(paths["output_dir"]) / f"{video_name}_{stamp}.mov"
    silent_output_path = video_tmp / f"{video_name}_{stamp}_silent.mov"

    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing exact source fps...")
    fps = get_exact_fps(source_video_path)
    print(f"Using fps = {fps} ({float(fps):.6f})")

    pairs = build_frame_pairs(frames_dir, masks_dir)
    print(f"Found {len(pairs)} matched frame/mask pairs")

    stitch_video(pairs, silent_output_path, fps, args.codec)
    mux_audio(silent_output_path, source_video_path, final_output_path)


if __name__ == "__main__":
    main()