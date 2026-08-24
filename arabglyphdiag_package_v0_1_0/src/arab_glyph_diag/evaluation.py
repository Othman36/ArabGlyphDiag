"""Functional sensitivity utilities used by ArabGlyphDiag validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import correlate2d
from scipy.stats import binomtest, wilcoxon

from .priors import binarize_ink


def mask_iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def translated_mask_bank(mask, max_iou: float = 0.25) -> dict:
    """Enumerate in-bounds spatial translations of an exact binary mask shape."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("Cannot translate an empty mask.")

    miny, maxy = int(ys.min()), int(ys.max())
    minx, maxx = int(xs.min()), int(xs.max())
    crop = mask[miny : maxy + 1, minx : maxx + 1].copy()
    ch, cw = crop.shape

    preferred, fallback = [], []
    for top in range(0, mask.shape[0] - ch + 1):
        for left in range(0, mask.shape[1] - cw + 1):
            if top == miny and left == minx:
                continue
            moved = np.zeros_like(mask)
            moved[top : top + ch, left : left + cw] = crop
            iou = mask_iou(mask, moved)
            item = {"top": top, "left": left, "iou": iou, "mask": moved}
            fallback.append(item)
            if iou <= max_iou:
                preferred.append(item)

    pool = preferred if preferred else fallback
    if not pool:
        raise RuntimeError("No valid translated mask controls could be constructed.")
    return {
        "crop": crop,
        "original_top": miny,
        "original_left": minx,
        "pool": pool,
        "used_low_iou_pool": bool(preferred),
    }


def choose_foreground_ink_matched_masks(
    image_or_foreground,
    expected_mask,
    bank=None,
    max_controls: int = 20,
    min_controls: int = 3,
    base_tolerance_frac: float = 0.05,
    max_iou: float = 0.25,
) -> tuple[list[np.ndarray], dict]:
    """Choose translated controls matched to foreground occupancy.

    The same binary mask shape is translated to alternative positions. Candidate
    placements are ranked first by absolute foreground-pixel mismatch relative
    to the expected region and then by lower IoU with the expected region.
    """
    arr = np.asarray(image_or_foreground)
    if arr.dtype == bool and arr.ndim == 2:
        foreground = arr
    else:
        foreground = binarize_ink(arr)

    expected_mask = np.asarray(expected_mask, dtype=bool)
    if bank is None:
        bank = translated_mask_bank(expected_mask, max_iou=max_iou)

    target_ink = int(np.logical_and(foreground, expected_mask).sum())
    overlap_map = correlate2d(
        foreground.astype(np.float32),
        bank["crop"].astype(np.float32),
        mode="valid",
    )
    pool = bank["pool"]
    overlaps = np.array(
        [int(round(float(overlap_map[item["top"], item["left"]]))) for item in pool],
        dtype=int,
    )
    ious = np.array([item["iou"] for item in pool], dtype=float)
    abs_diff = np.abs(overlaps - target_ink)
    order = np.lexsort((ious, abs_diff))

    base_tol = max(1, int(round(base_tolerance_frac * max(target_ink, 1))))
    tolerance = base_tol
    eligible = order[abs_diff[order] <= tolerance]
    max_diff = int(abs_diff.max()) if len(abs_diff) else 0
    while len(eligible) < min_controls and tolerance < max_diff:
        tolerance += 1
        eligible = order[abs_diff[order] <= tolerance]
    if len(eligible) == 0:
        eligible = order[:1]

    take = eligible[: min(max_controls, len(eligible))]
    masks = [pool[int(i)]["mask"].copy() for i in take]
    selected_ink = overlaps[take]
    selected_ious = ious[take]
    selected_diff = abs_diff[take]

    stats = {
        "expected_ink_pixels": target_ink,
        "control_ink_pixels_mean": float(np.mean(selected_ink)),
        "control_ink_pixels_min": int(np.min(selected_ink)),
        "control_ink_pixels_max": int(np.max(selected_ink)),
        "control_mean_abs_ink_diff": float(np.mean(selected_diff)),
        "control_max_abs_ink_diff": int(np.max(selected_diff)),
        "control_exact_ink_match_fraction": float(np.mean(selected_diff == 0)),
        "control_within1_ink_pixel_fraction": float(np.mean(selected_diff <= 1)),
        "control_within2_ink_pixel_fraction": float(np.mean(selected_diff <= 2)),
        "control_mean_iou": float(np.mean(selected_ious)),
        "control_max_iou": float(np.max(selected_ious)),
        "control_pool_size": int(len(pool)),
        "control_reps_used": int(len(take)),
        "match_tolerance_pixels_used": int(tolerance),
        "match_base_tolerance_pixels": int(base_tol),
        "used_low_iou_pool": bool(bank["used_low_iou_pool"]),
    }
    return masks, stats


def occlude_images(images, mask, fill_value: float = 0.0) -> np.ndarray:
    X = np.asarray(images).copy()
    mask = np.asarray(mask, dtype=bool)
    if X.ndim == 3:
        X[:, mask] = fill_value
    elif X.ndim == 4 and X.shape[-1] == 1:
        plane = X[..., 0]
        plane[:, mask] = fill_value
        X[..., 0] = plane
    else:
        raise ValueError("images must have shape [N,H,W] or [N,H,W,1].")
    return X


def occlude_one_image_many_masks(image, masks: Sequence[np.ndarray], fill_value: float = 0.0) -> np.ndarray:
    image = np.asarray(image)
    out = np.repeat(image[None, ...], len(masks), axis=0)
    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if out.ndim == 4 and out.shape[-1] == 1:
            out[i, ..., 0][mask] = fill_value
        elif out.ndim == 3:
            out[i][mask] = fill_value
        else:
            raise ValueError("image must be HxW or HxWx1.")
    return out


def bootstrap_mean_ci(values, reps: int = 10000, seed: int = 2026, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for b in range(reps):
        idx = rng.integers(0, len(values), size=len(values))
        means[b] = values[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def class_level_effect_summary(values, bootstrap_reps: int = 10000, seed: int = 2026) -> dict:
    """Summarize one effect value per class using the paper-facing unit."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "n_classes": 0,
            "positive_classes": 0,
            "negative_classes": 0,
            "zero_classes": 0,
            "mean_class_delta": np.nan,
            "median_class_delta": np.nan,
            "bootstrap95_low_mean_class_delta": np.nan,
            "bootstrap95_high_mean_class_delta": np.nan,
            "class_wilcoxon_W": np.nan,
            "class_wilcoxon_p": np.nan,
            "class_sign_test_p_two_sided": np.nan,
        }

    nonzero = vals[np.abs(vals) > 1e-12]
    if len(nonzero):
        w = wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
        n_pos = int(np.sum(nonzero > 0))
        sign_p = float(binomtest(n_pos, len(nonzero), 0.5, alternative="two-sided").pvalue)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    else:
        sign_p, w_stat, w_p = np.nan, np.nan, np.nan
    lo, hi = bootstrap_mean_ci(vals, reps=bootstrap_reps, seed=seed)
    return {
        "n_classes": int(len(vals)),
        "positive_classes": int(np.sum(vals > 0)),
        "negative_classes": int(np.sum(vals < 0)),
        "zero_classes": int(np.sum(np.abs(vals) <= 1e-12)),
        "mean_class_delta": float(np.mean(vals)),
        "median_class_delta": float(np.median(vals)),
        "bootstrap95_low_mean_class_delta": lo,
        "bootstrap95_high_mean_class_delta": hi,
        "class_wilcoxon_W": w_stat,
        "class_wilcoxon_p": w_p,
        "class_sign_test_p_two_sided": sign_p,
    }
