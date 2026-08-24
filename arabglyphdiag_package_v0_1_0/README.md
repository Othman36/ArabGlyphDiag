# ArabGlyphDiag

ArabGlyphDiag is an open-source toolkit for **structural error diagnosis in Arabic handwritten character recognition**.

It is designed to answer a different question from a recognition benchmark: once a classifier has produced predictions, **what kinds of Arabic-script-specific errors remain?** The core API works from reference labels plus predicted labels or class probabilities. It does not require access to a particular model architecture.

## What the package provides

- hierarchical error taxonomy for skeleton, dot-count, and dot-position confusions;
- confusion-pair tables with Arabic structural labels;
- dataset-level structural failure fingerprints;
- per-class accuracy and family-level error burden;
- lightweight connected-component secondary-region extraction;
- empirical class-specific secondary-component priors;
- same-shape translated and foreground-ink-matched occlusion controls;
- class-level Wilcoxon, sign-test, and bootstrap summaries;
- optional experimental Keras Grad-CAM adapter.

## Important scope

ArabGlyphDiag is **diagnostic software**, not a new classifier and not a causal explanation method. The structural taxonomy is operational and hierarchical. Exact connected-component count should be treated as an auxiliary descriptive signal. In the published validation, spatial localization was more reliable than exact component counting.

The core prediction-based diagnostic interface is architecture-independent when predictions are supplied externally. The final empirical evaluation used one frozen convolutional backbone family on AHCD and Hijja2; systematic multi-architecture validation remains future work.

## Installation

From the source directory:

```bash
pip install .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

For the optional Keras Grad-CAM adapter:

```bash
pip install '.[keras]'
```

## Python quick start

```python
import numpy as np
from arab_glyph_diag import GlyphDiagnostic

# Integer class ids in the 0..27 AHCD/Hijja2 ordering.
y_true = np.array([1, 2, 3, 16, 20])
y_pred = np.array([1, 3, 2, 15, 19])

diag = GlyphDiagnostic.from_predictions(y_true, y_pred=y_pred)

print(diag.summary())
print(diag.structural_fingerprint())
print(diag.confusion_pairs())
print(diag.family_error_burden())

diag.write_report('arabglyphdiag_report')
```

You can also supply class probabilities:

```python
diag = GlyphDiagnostic.from_predictions(y_true, probs=probabilities)
```

## CLI

For a CSV containing `y_true` and `y_pred`:

```bash
arabglyphdiag diagnose predictions.csv --output-dir report
```

For probability columns named `p_0`, `p_1`, ..., `p_27`:

```bash
arabglyphdiag diagnose predictions.csv --proba-prefix p_ --output-dir report
```

The report directory contains:

- `summary.json`
- `confusion_pairs.csv`
- `structural_fingerprint.csv`
- `per_class_accuracy.csv`
- `family_error_burden.csv`

## Empirical secondary-component priors

```python
from arab_glyph_diag import build_empirical_priors, top_area_mask

priors, source = build_empirical_priors(
    train_images,
    train_labels,
    fallback='none',
)

mask = top_area_mask(priors[2], area_frac=0.025)
```

The default connected-component thresholds reproduce the **32x32** experimental setting. Re-calibrate them for materially different resolutions.

## Foreground-ink-matched control

```python
from arab_glyph_diag.evaluation import (
    translated_mask_bank,
    choose_foreground_ink_matched_masks,
)

bank = translated_mask_bank(expected_mask, max_iou=0.25)
controls, match_stats = choose_foreground_ink_matched_masks(
    image,
    expected_mask,
    bank=bank,
    max_controls=20,
    min_controls=3,
)
```

This preserves the mask shape and area while choosing translated locations whose foreground occupancy is as close as possible to the expected region on the same image.

## Validation snapshot

The frozen study reported:

- AHCD primary single-model accuracy: **98.442% +/- 0.075 pp**;
- Hijja2 primary single-model accuracy: **93.926% +/- 0.137 pp**;
- residual AHCD errors: 55.77% skeleton, 44.23% dot-count;
- residual Hijja2 errors: 86.64% skeleton, 13.16% dot-count, 0.20% dot-position;
- blinded 300-image human comparison: 76.74% position agreement, 58.57% exact component-count agreement, 78.75% presence agreement;
- at 2.5% mask area, expected-region occlusion exceeded foreground-ink-matched translated controls by a class-level mean of **0.1496** on AHCD and **0.1422** on Hijja2;
- the 2.5% effect was positive for 13/15 AHCD dotted classes and 15/15 Hijja2 dotted classes.

See `docs/validation.md` for the full scope and caveats.

## Reproducibility and test-set history

The final Accuracy-v2 model selection and epoch selection used development partitions only, and the test set was reporting-only within that locked run. Earlier exploratory project iterations had already evaluated the official benchmark test sets, so the final numbers should be interpreted as **validation-driven benchmark evaluations**, not historically pristine one-shot test estimates.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
