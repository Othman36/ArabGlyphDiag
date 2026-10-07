# Reviewed software snapshot

Manuscript: **NEUCOM-D-26-18891**  
Software: **ArabGlyphDiag 0.1.0**  
Recommended immutable Git tag: **`v0.1.0-neucom-r1`**

This snapshot preserves the ArabGlyphDiag 0.1.0 API and taxonomy. The manuscript-response work added validation evidence and documentation; it did not change the public diagnostic semantics.

## Verification

The exact source tree in this archive was checked with:

```bash
PYTHONPATH=src pytest -q
```

Result: **12 passed**.

## Dataset provenance

The supplementary HMBD experiment uses the original repository:

`https://github.com/HossamBalaha/HMBD-v1`

All 13,533 isolated-letter images in the 28-class subset decoded successfully in the final source audit. The fixed image-level split contains 9,473 training, 2,030 fit-validation, and 2,030 test images. Writer identities are not exposed by the isolated-letter layout, so no writer-disjoint claim is made for HMBD.

## Archiving

For resubmission reproducibility, create the Git tag above on the public repository. If a DOI-backed archive is desired, archive that exact tag/release (for example through Zenodo) and add the DOI to the repository and manuscript only after the DOI exists.
