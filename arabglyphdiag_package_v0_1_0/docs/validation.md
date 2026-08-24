# Validation summary

This document records the empirical scope used to validate ArabGlyphDiag. It is not a claim that every module has identical reliability.

## Frozen recognition results

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

The structural auxiliary losses did not show a statistically established accuracy advantage. AHCD was tied in mean development accuracy; Hijja2 selected structural multitask numerically, but the three-seed paired comparison was non-significant. The recognition improvement therefore belongs to the revised training recipe as a whole and is not attributed specifically to the structural loss.

## Structural fingerprints

Using frozen ensemble predictions:

| Dataset | Total errors | Skeleton | Dot count | Dot position |
|---|---:|---:|---:|---:|
| AHCD | 52 | 29 (55.77%) | 23 (44.23%) | 0 |
| Hijja2 | 494 | 428 (86.64%) | 65 (13.16%) | 1 (0.20%) |

Across nine multi-letter skeleton families, the cross-dataset Spearman correlation of within-family error rate was rho=0.3235, p=0.3958. Family difficulty should therefore be treated as dataset-dependent rather than universal.

## Blinded human audit

A single Arabic-literate author completed a fixed 300-image blinded audit: 10 examples from each of 15 dotted classes in both AHCD and Hijja2. A fixed 60-image subset was independently re-annotated, and a separate fixed 60-image subset was used for direct detector-overlay review.

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

The effect remained positive at 5%, 7.5%, and 10%, but 2.5% and 5% are the cleanest specificity checks. Larger masks, particularly Hijja2 at 10%, had poorer foreground-match balance and are best treated as sensitivity analyses.

The appropriate interpretation is that the estimated regions show spatially specific functional sensitivity beyond generic foreground removal. This does not prove that the classifier reasons causally in terms of Arabic dots.

## Grad-CAM

Grad-CAM was explored but did not pass the spatial null-control check consistently. It should be treated as exploratory or supplementary rather than as validated structural-reasoning evidence.

## Runtime

Probability-to-report median overhead, excluding classifier inference and optional perturbation analyses:

- AHCD, 3,360 samples: 0.001786 s.
- Hijja2, 9,159 samples: 0.002438 s.

## Test-set history

Within the locked Accuracy-v2 protocol, model and epoch selection used development partitions and the official test split was reporting-only. Earlier exploratory iterations of the broader project had already evaluated the official AHCD and Hijja2 test sets. The final benchmark values should therefore be described as validation-driven evaluations rather than historically pristine one-shot estimates.
