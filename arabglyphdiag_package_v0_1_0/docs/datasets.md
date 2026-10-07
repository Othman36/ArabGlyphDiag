# Dataset sources used in validation

ArabGlyphDiag does not redistribute benchmark datasets.

- AHCD: use the original AHCD source referenced in the manuscript.
- Hijja/Hijja2: use the original Hijja/Hijja2 source referenced in the manuscript.
- HMBD v1 supplementary isolated-letter experiment: `https://github.com/HossamBalaha/HMBD-v1`.

The HMBD experiment uses all 13,533 decodable images in the 28 isolated-letter folders. The fixed 70/15/15 image-level split contains 9,473 training, 2,030 fit-validation, and 2,030 test images. The repository layout used for this experiment does not expose writer identities for a writer-disjoint reconstruction, so HMBD is treated as a supplementary software-generalization setting.
