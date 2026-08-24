import numpy as np

from arab_glyph_diag.evaluation import (
    choose_foreground_ink_matched_masks,
    class_level_effect_summary,
    translated_mask_bank,
)


def test_translated_masks_preserve_area():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:4, 6:9] = True
    bank = translated_mask_bank(mask, max_iou=0.25)
    assert bank["pool"]
    assert all(item["mask"].sum() == mask.sum() for item in bank["pool"])


def test_ink_matched_control_prefers_same_foreground_count():
    expected = np.zeros((16, 16), dtype=bool)
    expected[2:4, 2:5] = True
    foreground = np.zeros((16, 16), dtype=bool)
    foreground[2:4, 2:5] = True
    foreground[10:12, 10:13] = True
    bank = translated_mask_bank(expected, max_iou=0.25)
    masks, stats = choose_foreground_ink_matched_masks(
        foreground,
        expected,
        bank=bank,
        max_controls=5,
        min_controls=1,
    )
    assert len(masks) >= 1
    assert stats["control_mean_abs_ink_diff"] == 0.0
    assert stats["control_exact_ink_match_fraction"] == 1.0


def test_class_level_summary():
    vals = np.array([0.1, 0.2, 0.05, -0.01, 0.12])
    out = class_level_effect_summary(vals, bootstrap_reps=1000, seed=1)
    assert out["n_classes"] == 5
    assert out["positive_classes"] == 4
    assert out["negative_classes"] == 1
    assert out["bootstrap95_high_mean_class_delta"] > out["bootstrap95_low_mean_class_delta"]
