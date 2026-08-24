"""Secondary-component extraction and empirical spatial priors."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import ndimage

from .metadata import GlyphMetadata, normalize_metadata


def binarize_ink(image, target_density: float = 0.15) -> np.ndarray:
    """Return a foreground mask while tolerating either image polarity."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError("image must be 2D or HxWx1.")
    threshold = float(img.mean())
    high = img > threshold
    low = img < threshold
    return high if abs(high.mean() - target_density) <= abs(low.mean() - target_density) else low


def detect_secondary_components(image, min_size: int = 2, max_size: int = 40) -> dict:
    """Extract small connected components relative to the largest ink component.

    The default size thresholds reproduce the 32x32 experimental setting and
    should be re-calibrated for materially different image resolutions.
    """
    binary = binarize_ink(image)
    if int(binary.sum()) < 5:
        return {"centroids": [], "sizes": [], "skeleton_centroid": None}

    labeled, num = ndimage.label(binary)
    if num <= 0:
        return {"centroids": [], "sizes": [], "skeleton_centroid": None}

    sizes = np.asarray(ndimage.sum(binary, labeled, range(1, num + 1)), dtype=float)
    main_label = int(np.argmax(sizes)) + 1
    skeleton_centroid = tuple(float(v) for v in ndimage.center_of_mass(binary, labeled, main_label))

    centroids, kept_sizes = [], []
    for label in range(1, num + 1):
        if label == main_label:
            continue
        size = float(sizes[label - 1])
        if min_size <= size <= max_size:
            cy, cx = ndimage.center_of_mass(binary, labeled, label)
            centroids.append((float(cy), float(cx)))
            kept_sizes.append(size)

    return {
        "centroids": centroids,
        "sizes": kept_sizes,
        "skeleton_centroid": skeleton_centroid,
    }


def describe_secondary_components(image, min_size: int = 2, max_size: int = 40) -> dict:
    """Return simple visible-component features from the lightweight extractor.

    ``position`` is based on the median vertical offset of retained small
    components relative to the largest-component centroid. A merged two-dot
    component can therefore have a correct region/position while still yielding
    an incorrect exact component count.
    """
    det = detect_secondary_components(image, min_size=min_size, max_size=max_size)
    centroids = det["centroids"]
    skeleton = det["skeleton_centroid"]
    if not centroids or skeleton is None:
        position = "none"
    else:
        dy = np.asarray([cy - skeleton[0] for cy, _ in centroids], dtype=float)
        position = "above" if float(np.median(dy)) < 0 else "below"
    return {
        **det,
        "component_count": int(len(centroids)),
        "component_present": bool(centroids),
        "position": position,
    }


def synthetic_spatial_prior(dot_position: str, size: int = 32, sigma_frac: float = 0.22) -> np.ndarray:
    """Construct a simple above/below Gaussian prior.

    This utility is provided as an explicit fallback. The published evaluation
    used empirical priors for all 15 dotted classes, so synthetic fallback is
    disabled by default in ``build_empirical_priors``.
    """
    if dot_position == "none":
        return np.zeros((size, size), dtype=np.float32)
    if dot_position not in {"above", "below"}:
        raise ValueError("dot_position must be 'above', 'below', or 'none'.")
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2.0
    cy = size * (0.22 if dot_position == "above" else 0.78)
    sigma = size * sigma_frac
    heat = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    return (heat / (heat.max() + 1e-8)).astype(np.float32)


def build_empirical_priors(
    images,
    labels,
    metadata: Mapping[int, GlyphMetadata] | None = None,
    n_samples_per_class: int = 250,
    sigma: float = 1.8,
    seed: int = 42,
    fallback: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate class-level secondary-component priors from handwriting data.

    Parameters
    ----------
    fallback:
        ``"none"`` leaves classes without enough detected evidence empty.
        ``"synthetic"`` uses a coarse above/below Gaussian fallback.
    """
    X = np.asarray(images)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(X) != len(y):
        raise ValueError("images and labels must have the same length.")
    if X.ndim not in {3, 4}:
        raise ValueError("images must have shape [N,H,W] or [N,H,W,1].")
    h, w = X.shape[1], X.shape[2]
    if h != w:
        raise ValueError("Current empirical-prior implementation expects square images.")
    if fallback not in {"none", "synthetic"}:
        raise ValueError("fallback must be 'none' or 'synthetic'.")

    meta = normalize_metadata(metadata)
    n_classes = max(meta) + 1
    priors = np.zeros((n_classes, h, w), dtype=np.float32)
    source = np.array(["none"] * n_classes, dtype=object)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)

    for class_id, m in meta.items():
        if m.dot_position == "none":
            continue
        idxs = np.where(y == class_id)[0]
        if len(idxs) > n_samples_per_class:
            idxs = rng.choice(idxs, n_samples_per_class, replace=False)
        heat = np.zeros((h, w), dtype=np.float32)
        hits = 0
        for idx in idxs:
            for cy, cx in detect_secondary_components(X[idx])["centroids"]:
                heat += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
                hits += 1
        threshold = max(5, int(0.10 * len(idxs)))
        if hits >= threshold and heat.max() > 0:
            priors[class_id] = heat / (heat.max() + 1e-8)
            source[class_id] = "empirical"
        elif fallback == "synthetic":
            priors[class_id] = synthetic_spatial_prior(m.dot_position, size=h)
            source[class_id] = "synthetic_fallback"

    return priors, source


def top_area_mask(prior, area_frac: float) -> np.ndarray:
    """Select the highest-prior pixels occupying ``area_frac`` of an image."""
    prior = np.asarray(prior, dtype=float)
    if prior.ndim != 2:
        raise ValueError("prior must be 2D.")
    if not (0 < area_frac <= 1):
        raise ValueError("area_frac must be in (0, 1].")
    k = max(1, int(round(prior.size * float(area_frac))))
    flat = prior.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros_like(flat, dtype=bool)
    out[idx] = True
    return out.reshape(prior.shape)
