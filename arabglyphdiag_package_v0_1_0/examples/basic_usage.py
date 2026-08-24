import numpy as np

from arab_glyph_diag import GlyphDiagnostic


y_true = np.array([1, 2, 3, 16, 20, 9, 10])
y_pred = np.array([1, 3, 2, 15, 19, 10, 9])

diag = GlyphDiagnostic.from_predictions(y_true, y_pred=y_pred)

print("Summary")
print(diag.summary())
print("\nStructural fingerprint")
print(diag.structural_fingerprint().to_string(index=False))
print("\nConfusion pairs")
print(diag.confusion_pairs().to_string(index=False))
