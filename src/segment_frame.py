import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def get_device():
    """Pick the best available device: CUDA (BlueBEAR) > MPS (local Mac) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam3(device):
    """Load the SAM3 model and processor onto the given device."""
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    return model, processor


def segment_frame(image, prompt, model, processor, device,
                   threshold=0.3, mask_threshold=0.3):
    """Run SAM3 on a single PIL image, return a binary mask as uint8 array."""
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    instance_masks = results["masks"].cpu().numpy().astype(bool)
    if instance_masks.shape[0] == 0:
        h, w = image.size[1], image.size[0]
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = (instance_masks.any(axis=0) * 255).astype(np.uint8)

    del inputs, outputs, results
    return mask


def segment_frames(frame_dir, mask_dir, prompt, device=None,
                    threshold=0.3, mask_threshold=0.3, model=None,
                    processor=None, progress_every=100):
    """Segment every frame in frame_dir for a given text prompt.

    Writes masks to mask_dir, skipping frames already processed.
    If model/processor are not provided, they are loaded here.
    Returns mask_dir.
    """
    if device is None:
        device = get_device()

    os.makedirs(mask_dir, exist_ok=True)

    if model is None or processor is None:
        model, processor = load_sam3(device)

    frame_files = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Found {len(frame_files)} frames.")

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        mask_path = os.path.join(mask_dir, frame_file)

        if os.path.exists(mask_path):
            continue

        image = Image.open(frame_path).convert("RGB")
        mask = segment_frame(
            image, prompt, model, processor, device,
            threshold=threshold, mask_threshold=mask_threshold,
        )
        Image.fromarray(mask).save(mask_path)

        del image, mask

        if (i + 1) % progress_every == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames.")

    print("Finished.")
    return mask_dir


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Segment frames with SAM3, driven by config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    video_name = config["video_name"]
    paths = config["paths"]
    segment_cfg = config["segment"]

    video_tmp = Path(paths["tmp_dir"]) / video_name
    frame_dir = video_tmp / "decoded_frames"

    prompts = config["prompts"]

    device = get_device()
    print(f"Using device: {device}")
    model, processor = load_sam3(device)

    for prompt in prompts:
        mask_dir = video_tmp / "segmented_frames" / prompt
        print(f"Segmenting prompt '{prompt}'...")
        segment_frames(
            str(frame_dir), str(mask_dir), prompt,
            device=device,
            threshold=segment_cfg["threshold"],
            mask_threshold=segment_cfg["mask_threshold"],
            model=model, processor=processor,
        )