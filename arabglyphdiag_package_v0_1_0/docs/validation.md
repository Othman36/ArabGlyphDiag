# Validation summary

This document records the empirical scope used to validate ArabGlyphDiag. It is not a claim that every module has identical reliability, that structural-error distributions are architecture-invariant, or that a diagnostic finding automatically yields an effective corrective intervention.

## Primary frozen recognition results

Primary locked single-model results:

| Dataset | Frozen variant | Accuracy | SD | Earlier project baseline | Delta |
|---|---|---:|---:|---:|---:|
| AHCD | letter_only | 98.442% | 0.075 pp | 92.85% | +5.592 pp |
| Hijja2 | structural_multitask | 93.926% | 0.137 pp | 87.21% | +6.716 pp |

Three-model ensemble results:

| Dataset | Raw | TTA |
|---|---:|---:|
| AHCD | 98.452% | 98.512% |
| Hijja2 | 94.606% | 94.694% |

The structural auxiliary losses did not show a statistically established accuracy advantage. AHCD was tied in mean development accuracy; Hijja2 selected structural multitask numerically, but the three-seed paired comparison was non-significant. The recognition improvement therefore belongs to the training recipe as a whole and is not attributed specifically to the structural loss.

## Primary structural fingerprints

Using frozen CNN ensemble predictions:

| Dataset | Total errors | Skeleton | Count | Position |
|---|---:|---:|---:|---:|
| AHCD | 52 | 29 (55.77%) | 23 (44.23%) | 0 |
| Hijja2 | 494 | 428 (86.64%) | 65 (13.16%) | 1 (0.20%) |

Across nine multi-letter skeleton families, AHCD contains only 23 within-family errors in total. The cross-dataset Spearman correlation of within-family error rate is rho=0.3235, p=0.3958. The family comparison is therefore underpowered and descriptive; the paper reports raw counts beside rates.

## Cross-architecture validation

The prediction-level metadata and deterministic hierarchy were applied unchanged to three recognizer families.

| Dataset | Recognizer | Accuracy | Errors | Skeleton share | Count share | Position share |
|---|---|---:|---:|---:|---:|---:|
| AHCD | Frozen CNN ensemble | 98.45% | 52 | 55.77% | 44.23% | 0.00% |
| AHCD | HOG + linear SVM | 76.76% | 781 | 73.37% | 25.86% | 0.77% |
| AHCD | Pretrained ViT-B/16 ensemble | 97.53% | 83 | 59.04% | 39.76% | 1.20% |
| Hijja2 | Frozen CNN ensemble | 94.61% | 494 | 86.64% | 13.16% | 0.20% |
| Hijja2 | HOG + linear SVM | 42.24% | 5,290 | 89.79% | 9.72% | 0.49% |
| Hijja2 | Pretrained ViT-B/16 ensemble | 89.08% | 1,000 | 88.80% | 11.10% | 0.10% |
| HMBD | HOG + linear SVM | 64.04% | 730 | 79.86% | 19.18% | 0.96% |
| HMBD | Pretrained ViT-B/16 ensemble | 94.14% | 119 | 71.43% | 28.57% | 0.00% |

For ViT-B/16, 32x32 grayscale glyphs are resized to 224x224, replicated to three channels, and normalized with the ImageNet mean and standard deviation. The classification head is adapted to 28 classes; after a head-only warm-up, the final four transformer blocks, final encoder normalization, and head are fine-tuned. Seeds 42, 123, and 777 are used, with fit-validation-only checkpoint selection.

These results show that the same software contract can be applied to substantially different recognizers. They do not establish architecture-invariant error distributions.

## HMBD supplementary dataset

HMBD uses the 28 isolated-letter folders from the original repository:

`https://github.com/HossamBalaha/HMBD-v1`

All 13,533 isolated-letter images decode successfully. Exact decoded-image hashes found no cross-class identical-image conflicts and no within-class duplicates. A fixed stratified 70/15/15 image-level split contains 9,473 training, 2,030 fit-validation, and 2,030 test images, with hashes disjoint across partitions.

The isolated-letter folder organization does not expose writer identities for reconstructing a writer-disjoint split. HMBD is therefore a supplementary software-generalization setting rather than a benchmark directly comparable to the official AHCD/Hijja2 tests.

## Blinded human audit

A single Arabic-literate annotator completed a fixed 300-image blinded audit: 10 examples from each of the 15 classes whose metadata contain nonzero secondary components in both AHCD and Hijja2. The prediction-level taxonomy still covers all 28 classes; the audit is restricted to these 15 classes because it validates the optional image-level secondary-component estimator.

Full 300-image human-versus-automatic agreement:

| Measure | Agreement | 95% CI |
|---|---:|---:|
| Position | 76.74% | 71.52-81.24% |
| Exact component count | 58.57% | 52.72-64.19% |
| Presence / identifiability | 78.75% | 73.52-83.19% |

Dataset-specific agreement:

| Measure | AHCD | Hijja2 |
|---|---:|---:|
| Position | 84.67% | 68.12% |
| Exact component count | 68.00% | 47.69% |
| Presence | 85.81% | 70.40% |

Intra-annotator repeatability on the 60-image repeat subset:

| Label | Raw agreement | Cohen kappa | Bootstrap 95% CI |
|---|---:|---:|---:|
| Diacritic identifiable | 93.33% | 0.640 | 0.209-0.916 |
| Dot count | 91.67% | 0.850 | 0.716-0.965 |
| Dot position | 95.00% | 0.876 | 0.723-1.000 |

Direct overlay review:

- 45/60 correct region;
- 3/60 partially correct;
- 2/60 incorrect;
- 8/60 no detector mark;
- 2/60 unclear;
- among 50 marked cases, 90.0% were fully correct and 96.0% were at least partially aligned with a genuine secondary-component region.

These results support region localization as a useful diagnostic signal. Exact component count is less reliable and should remain auxiliary.

## Functional occlusion validation

The final specificity check compared expected class-specific secondary-component regions with translated masks matched for exact shape, area, and per-image foreground occupancy.

At 2.5% image area:

| Dataset | Positive classes | Mean class delta | Bootstrap 95% CI | Wilcoxon p | Sign p |
|---|---:|---:|---:|---:|---:|
| AHCD | 13/15 | 0.149619 | 0.097214-0.202415 | 0.000427 | 0.007385 |
| Hijja2 | 15/15 | 0.142220 | 0.111705-0.171306 | 0.000061 | 0.000061 |

Foreground matching at 2.5% was close:

| Dataset | Expected ink pixels | Control ink pixels | Mean absolute mismatch | Within 1 pixel |
|---|---:|---:|---:|---:|
| AHCD | 6.036 | 6.004 | 0.250 | 99.93% |
| Hijja2 | 5.420 | 5.333 | 0.357 | 98.91% |

The appropriate interpretation is spatially specific functional sensitivity beyond generic foreground removal. This does not establish a causal model-internal dot representation.

## Exploratory Grad-CAM analysis

Valid dot-count-confusion cases numbered 5 on AHCD (4 ordered confusion pairs) and 29 on Hijja2 (10 ordered pairs). Correct-prior alignment did not reliably exceed flipped or wrong-class controls. Holm adjustment across four sample-level dataset-by-control contrasts produced adjusted p-values of 0.303, 0.625, 0.116, and 0.303. Pair-level sensitivity checks were likewise non-significant. Grad-CAM is therefore retained only as an exploratory adapter and is not evidence of structural reasoning.

## Diagnostic-guided intervention case study

A HOG+linear-SVM case study used the dominant diagnostic mode and the five most frequent diagnostic-validation confusion pairs to select classes for a fixed 2x training weight. The dominant mode was cross-skeleton confusion on both AHCD and Hijja2, so the experiment did not produce dataset-specific broad-category selection.

The intervention reduced accuracy from 76.76% to 75.77% on AHCD and from 42.24% to 40.71% on Hijja2. It also increased target-mode errors from 573 to 593 and from 4,750 to 4,873. Against 20 deterministic random class sets per dataset matched on the number of up-weighted classes and the same 2x multiplier, the diagnostic-guided rule did not perform better. The experiment therefore demonstrates an auditable diagnostic-to-action workflow, not an effective or uniquely diagnostic-derived remedy.

## Runtime

Probability-to-report median overhead, excluding classifier inference and optional perturbation analyses:

- AHCD, 3,360 samples: 0.001786 s.
- Hijja2, 9,159 samples: 0.002438 s.

## Test-set history

Within the locked Accuracy-v2 protocol, model and epoch selection used development partitions and the official test split was reporting-only. Earlier exploratory iterations of the broader project had already evaluated the official AHCD and Hijja2 test sets. The final benchmark values should therefore be described as validation-driven evaluations rather than historically pristine one-shot estimates.
