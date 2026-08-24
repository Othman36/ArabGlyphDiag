import numpy as np

from arab_glyph_diag import GlyphDiagnostic, failure_mode_for_pair


def test_failure_taxonomy():
    assert failure_mode_for_pair(2, 3) == "dot_count_confusion"
    assert failure_mode_for_pair(16, 15) == "dot_count_confusion"
    assert failure_mode_for_pair(8, 10) == "skeleton_confusion"
    assert failure_mode_for_pair(1, 1) == "correct"


def test_diagnostic_summary_and_fingerprint():
    y_true = np.array([2, 3, 16, 8, 1])
    y_pred = np.array([3, 2, 15, 10, 1])
    diag = GlyphDiagnostic.from_predictions(y_true, y_pred=y_pred)
    summary = diag.summary()
    assert summary["n_samples"] == 5
    assert summary["total_errors"] == 4
    assert summary["within_skeleton_errors"] == 3
    fp = diag.structural_fingerprint().set_index("failure_mode")
    assert int(fp.loc["dot_count_confusion", "count"]) == 3
    assert int(fp.loc["skeleton_confusion", "count"]) == 1


def test_probs_can_generate_predictions():
    y_true = np.array([0, 1, 2])
    probs = np.zeros((3, 28), dtype=float)
    probs[0, 0] = 1
    probs[1, 1] = 1
    probs[2, 3] = 1
    diag = GlyphDiagnostic.from_predictions(y_true, probs=probs)
    assert np.array_equal(diag.y_pred, np.array([0, 1, 3]))
    assert abs(diag.accuracy - 2 / 3) < 1e-12


def test_family_error_burden():
    y_true = np.array([1, 2, 3, 1, 2, 3])
    y_pred = np.array([2, 2, 3, 10, 3, 1])
    diag = GlyphDiagnostic.from_predictions(y_true, y_pred=y_pred)
    fam = diag.family_error_burden()
    beh = fam[fam.skeleton_family == "beh"].iloc[0]
    assert beh.n == 6
    assert beh.family_total_errors == 4
    assert beh.within_family_errors == 3
