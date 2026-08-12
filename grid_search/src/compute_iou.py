"""
Compute IoU (intersection-over-union) between a predicted mask and a
ground-truth mask. Standalone helper used by grid_search.py, but also
runnable directly for a one-off comparison.
"""

import argparse

import numpy as np
from PIL import Image


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, beta: float = 0.5) -> dict:
    """
    pred_mask, gt_mask: 2D arrays, foreground = nonzero, background = 0.

    Returns a dict with iou, precision, recall, and f_beta (default beta=0.5,
    which weights precision more heavily than recall - i.e. penalizes
    background bleeding through more than it penalizes losing a bit of true
    foreground at the edge). All values are floats in [0, 1].

    Edge case: if pred_mask is completely empty, precision is undefined
    (0/0) - defined here as 1.0 if gt is also empty (nothing to disagree on)
    else 0.0, matching compute_iou's convention. Recall is 0.0 whenever gt
    is non-empty and pred is empty (correctly punishing an empty/degenerate
    prediction rather than rewarding it).
    """
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0

    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    union = int(np.logical_or(pred_binary, gt_binary).sum())
    pred_area = int(pred_binary.sum())
    gt_area = int(gt_binary.sum())

    iou = 1.0 if union == 0 else intersection / union

    if pred_area == 0:
        precision = 1.0 if gt_area == 0 else 0.0
    else:
        precision = intersection / pred_area

    if gt_area == 0:
        recall = 1.0 if pred_area == 0 else 0.0
    else:
        recall = intersection / gt_area

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    f_beta = 0.0 if denom == 0 else (1 + beta_sq) * precision * recall / denom

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        f"f{beta}": f_beta,
    }


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    pred_mask, gt_mask: 2D arrays, foreground = nonzero, background = 0.
    Returns IoU as a float in [0, 1]. Returns 1.0 if both masks are
    completely empty (nothing to disagree on), 0.0 if union is empty
    but one mask is non-empty (shouldn't happen, but guards divide-by-zero).
    """
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def compute_iou_from_paths(pred_path, gt_path) -> float:
    """Load two mask PNGs and compute IoU. Errors loudly if dimensions
    don't match, since a silent resize/mismatch would corrupt the metric."""
    pred = np.array(Image.open(pred_path).convert("L"))
    gt = np.array(Image.open(gt_path).convert("L"))

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} ({pred_path}) vs "
            f"gt {gt.shape} ({gt_path}). Masks must be the same size."
        )

    return compute_iou(pred, gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IoU between two mask PNGs.")
    parser.add_argument("pred", help="Path to predicted mask PNG")
    parser.add_argument("gt", help="Path to ground-truth mask PNG")
    args = parser.parse_args()

    iou = compute_iou_from_paths(args.pred, args.gt)
    print(f"IoU: {iou:.4f}")