# Changelog

## 0.1.0 - 2026-08-24

Initial research release.

- Prediction/probability-based structural diagnostics.
- Arabic 28-class metadata and hierarchical error taxonomy.
- Structural fingerprints and family error burden.
- Empirical secondary-component priors.
- Foreground-ink-matched translated occlusion utilities.
- Optional experimental Keras Grad-CAM adapter.
- CLI, tests, validation notes, and provenance guidance.

### Documentation refresh - 2026-10-05

- Documented cross-architecture validation on residual CNN, HOG+linear-SVM, and ImageNet-pretrained ViT-B/16.
- Documented complete-source HMBD v1 supplementary results from the original repository.
- Added dependence-aware Grad-CAM null-control summary.
- Added matched random-control results for the diagnostic-guided class-weighting case study.
- Clarified that the image-level audit targets the 15 classes with nonzero secondary components while the prediction taxonomy covers all 28 classes.
- No public API or package-version change.

## Neurocomputing resubmission snapshot - NEUCOM-D-26-18891

- Froze manuscript-facing documentation after the reviewer-response validation runs.
- Documented complete-source HMBD v1 results from the original `HossamBalaha/HMBD-v1` repository.
- Documented the cross-architecture HOG/SVM and ImageNet-pretrained ViT-B/16 checks.
- Documented the dependence-aware Grad-CAM null analysis and the matched-random intervention control.
- Confirmed the source test suite: 12 tests pass with `PYTHONPATH=src pytest -q`.
- Recommended immutable repository tag for the reviewed snapshot: `v0.1.0-neucom-r1`.
