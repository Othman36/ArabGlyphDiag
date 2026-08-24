import numpy as np

from arab_glyph_diag.priors import binarize_ink, detect_secondary_components, top_area_mask


def test_binarize_ink_handles_simple_image():
    img = np.zeros((32, 32), dtype=float)
    img[10:20, 10:20] = 1.0
    mask = binarize_ink(img)
    assert mask.shape == (32, 32)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_component_detector_finds_small_component():
    img = np.zeros((32, 32), dtype=float)
    img[12:20, 10:22] = 1.0
    img[5:7, 15:17] = 1.0
    det = detect_secondary_components(img)
    assert det["skeleton_centroid"] is not None
    assert len(det["centroids"]) >= 1


def test_top_area_mask_exact_pixel_count():
    prior = np.arange(100, dtype=float).reshape(10, 10)
    mask = top_area_mask(prior, 0.10)
    assert mask.sum() == 10


def test_describe_secondary_components_position():
    from arab_glyph_diag.priors import describe_secondary_components

    img = np.zeros((32, 32), dtype=float)
    img[14:24, 10:22] = 1.0
    img[5:7, 15:17] = 1.0
    desc = describe_secondary_components(img)
    assert desc["component_present"] is True
    assert desc["component_count"] >= 1
    assert desc["position"] == "above"
