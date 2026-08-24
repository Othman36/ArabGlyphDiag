# Provenance notes

The research notebooks used fixed manifests, frozen model weights, saved prediction arrays, exact split reconstruction, and SHA-256 hashes.

## Human-audit frozen files

- `PASS1_blinded_annotation_completed.csv`
  - SHA-256: `e66fbae82a76b443d8a43332831635e471c87417aaab470e8daaee517dc7a251`
- `PASS2_repeat_blinded_annotation_completed.csv`
  - SHA-256: `28bdfa02bf22ffd8129e939a3a2e74d2fb8b8df734db41ceef3a2e8c4da90b1a`
- `DETECTOR_REVIEW_60_annotation_completed.csv`
  - SHA-256: `2e10732f92ab2fc7a3b4351330ed49112a821618d59386747a0c86a487a2846b`

The audit manifest recorded the Pass 1, Pass 2, human-audit, detector-review, and all-stage states as frozen.

## Research notebook lineage

1. `ArabGlyphDiag_Accuracy_v2.ipynb`
   - locked recognition protocol and model artifacts.
2. `ArabGlyphDiag_Blinded_Human_Audit_UI_v3.ipynb`
   - blinded 300-image human audit, repeat subset, and secondary overlay review.
3. `ArabGlyphDiag_PostHoc_Validation_v5.ipynb`
   - frozen-recognizer structural diagnostics and foreground-ink-matched occlusion control.

The packaged library intentionally does not include the trained model weights or benchmark datasets. It provides the reusable diagnostic implementation.
