"""
Centralized path resolution for the nested, param-aware tmp/<video_name>/ tree.

Every stage script's __main__ block should resolve its input/output directories
through this module rather than hand-building paths. This is the single place
that knows the tree shape, float formatting, and shorthand abbreviations -
change it here once, every stage picks it up.

Tree shape:
    tmp/<video_name>/
        decoded_frames/                                        (no params)
        segmented_frames/<prompt>/t{th}_mt{mth}/                (per-prompt params)
        merged_frames/<prompt_short>-t{th}-mt{mth}__.../         (joined, sorted by prompt name)
            remove_fg_noise/                                    (no params today)
                fill_small_holes/mha{max_hole_area}/
                    opening/k{kernel_size}_i{iterations}/
                        erosion/k{kernel_size}_i{iterations}/
                            smooth_gaussian/s{sigma}/

Nothing in this module reads/writes files - it only computes Path objects.
"""

import hashlib
from pathlib import Path
from typing import Optional


def _fmt_float(value: float) -> str:
    """Normalize float formatting so 0.3 and 0.30 in config.json never diverge
    into separate folders. Always 2 decimal places."""
    return f"{float(value):.2f}"


def video_tmp_dir(config: dict) -> Path:
    """tmp/<video_name>/"""
    return Path(config["paths"]["tmp_dir"]) / config["video_name"]


def decoded_frames_dir(config: dict) -> Path:
    """tmp/<video_name>/decoded_frames/  (no params - single shared folder)"""
    return video_tmp_dir(config) / "decoded_frames"


def _prompt_param_folder(prompt_cfg: dict) -> str:
    """t{threshold}_mt{mask_threshold}, e.g. t0.30_mt0.30"""
    return f"t{_fmt_float(prompt_cfg['threshold'])}_mt{_fmt_float(prompt_cfg['mask_threshold'])}"


def segment_dir(config: dict, prompt: str) -> Path:
    """tmp/<video_name>/segmented_frames/<prompt>/t{th}_mt{mth}/"""
    prompt_cfg = config["prompts"][prompt]
    return (
        video_tmp_dir(config)
        / "segmented_frames"
        / prompt
        / _prompt_param_folder(prompt_cfg)
    )


def merge_key(config: dict) -> str:
    """Joined, sorted-by-prompt-name shorthand key describing every prompt's
    params, e.g. p-t0.30-mt0.30__v-t0.30-mt0.30__vb-t0.30-mt0.40"""
    prompts = config["prompts"]
    parts = []
    for prompt_name in sorted(prompts.keys()):
        prompt_cfg = prompts[prompt_name]
        short = prompt_cfg["short"]
        parts.append(f"{short}-t{_fmt_float(prompt_cfg['threshold'])}-mt{_fmt_float(prompt_cfg['mask_threshold'])}")
    return "__".join(parts)


def merge_base(config: dict) -> Path:
    """tmp/<video_name>/merged_frames/<merge_key>/

    Nesting anchor for the postprocess chain - remove_fg_noise/ sits directly
    under this. Actual merged output frames do NOT live here directly (see
    merge_dir) - they live in a "frames" subfolder, kept separate so merged
    .png output is never mixed at the same folder level as remove_fg_noise/.
    """
    return video_tmp_dir(config) / "merged_frames" / merge_key(config)


def merge_dir(config: dict) -> Path:
    """tmp/<video_name>/merged_frames/<merge_key>/frames/

    Where the OR-merged mask .png frames actually live. Kept in a "frames"
    subfolder so they're not sitting at the same folder level as
    remove_fg_noise/ (the first postprocess stage nested under this combo).
    """
    return merge_base(config) / "frames"


# Registry of postprocess stages in pipeline order, each with:
#   - the config key under config["postprocess"] holding its params
#   - a function that formats those params into a folder name (None if no params)
_POSTPROCESS_STAGES = {
    "remove_fg_noise": None,  # no params today
    "fill_small_holes": lambda p: f"mha{p['max_hole_area']}",
    "opening": lambda p: f"k{p['kernel_size']}_i{p['iterations']}",
    "erosion": lambda p: f"k{p['kernel_size']}_i{p['iterations']}",
    "smooth_gaussian": lambda p: f"s{_fmt_float(p['sigma'])}",
}

POSTPROCESS_STAGE_ORDER = list(_POSTPROCESS_STAGES.keys())


def _postprocess_stage_base(config: dict, stage_name: str) -> Path:
    """
    Internal: tmp/<video_name>/merged_frames/<merge_key>/<stage_1>/<stage_1_params>/.../<stage_name>/<stage_name_params>/

    This is the nesting anchor for stage_name - the next stage's base sits
    directly under it. Actual output frames for stage_name do NOT live here
    directly (see postprocess_dir) - they live in a "frames" subfolder, kept
    separate so a stage's png output is never mixed at the same folder level
    as the next stage's subfolder.
    """
    if stage_name not in _POSTPROCESS_STAGES:
        raise ValueError(
            f"Unknown postprocess stage '{stage_name}'. "
            f"Expected one of {POSTPROCESS_STAGE_ORDER}."
        )

    path = merge_base(config)
    postprocess_cfg = config["postprocess"]

    for stage in POSTPROCESS_STAGE_ORDER:
        path = path / stage
        folder_fn = _POSTPROCESS_STAGES[stage]
        if folder_fn is not None:
            path = path / folder_fn(postprocess_cfg[stage])
        if stage == stage_name:
            break

    return path


def postprocess_dir(config: dict, stage_name: str) -> Path:
    """
    tmp/<video_name>/merged_frames/<merge_key>/.../<stage_name>/<stage_name_params>/frames/

    Where a stage's actual output .png frames live. Kept in a "frames"
    subfolder (rather than directly in the stage's param folder) so frames
    are never sitting at the same folder level as the next stage's
    subfolder - e.g. remove_fg_noise/frames/*.png and
    remove_fg_noise/fill_small_holes/ are siblings under remove_fg_noise/,
    instead of *.png files and fill_small_holes/ being mixed together.
    """
    return _postprocess_stage_base(config, stage_name) / "frames"


def is_stage_done(output_dir: Path, expected_count: Optional[int] = None) -> bool:
    """
    True if this stage's work is already done for the current param combo
    and can be reused/skipped.

    If expected_count is given, requires the number of .png files in
    output_dir to match it exactly - this is the recommended usage, since a
    stale/partial folder (e.g. left over from an interrupted run, or from a
    prior decode with a different frame count) can otherwise have "at least
    one file" but not the full set, silently producing incomplete/mismatched
    output downstream. Pass expected_count = number of frames in
    decoded_frames/ wherever that's available.

    If expected_count is omitted, falls back to the weaker "at least one
    .png file present" check - kept for callers where the expected count
    genuinely isn't known.
    """
    if not output_dir.is_dir():
        return False

    png_files = list(output_dir.glob("*.png"))

    if expected_count is not None:
        return len(png_files) == expected_count

    return len(png_files) > 0


def decoded_frame_count(config: dict) -> int:
    """Number of decoded .png frames for this video - the expected_count to
    pass to is_stage_done() for every downstream stage, since every stage
    should produce exactly one output frame per decoded input frame."""
    return len(list(decoded_frames_dir(config).glob("*.png")))


def run_stamp(config: dict, length: int = 8) -> str:
    """
    Short deterministic hash identifying the exact param combo that produced
    a given run - every prompt's (threshold, mask_threshold) plus every
    postprocess stage's params. Same config -> same stamp, always; different
    params (even a single value) -> a different stamp.

    Deliberately a hash of params, not a timestamp: keeps the final output
    filename consistent with the rest of the tree's reuse philosophy (same
    config -> same identifying string), rather than getting a new, unrelated
    filename on every rerun of an unchanged config.

    Used to disambiguate final output files in Output_Videos/ once multiple
    param combos are being run/compared, so they don't overwrite each other.
    """
    # The final postprocess stage's base path already encodes merge_key plus
    # every postprocess stage's params via nesting - reuse it as the single
    # source of truth for "what params produced this" rather than
    # re-deriving/duplicating that logic here.
    identifying_string = str(_postprocess_stage_base(config, POSTPROCESS_STAGE_ORDER[-1]))
    digest = hashlib.sha1(identifying_string.encode("utf-8")).hexdigest()
    return digest[:length]