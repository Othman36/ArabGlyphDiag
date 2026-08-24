"""Probability/prediction-based structural diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metadata import GlyphMetadata, normalize_metadata


def failure_mode_for_pair(true_id: int, pred_id: int, metadata: Mapping[int, GlyphMetadata] | None = None) -> str:
    """Return the hierarchical structural relation for a true/predicted pair.

    Hierarchy:
    1. different skeleton group -> ``skeleton_confusion``
    2. same skeleton, different dot count -> ``dot_count_confusion``
    3. same skeleton/count, different dot position -> ``dot_position_confusion``
    4. otherwise -> ``same_structure_confusion``

    This is an operational taxonomy, not a causal explanation of why a model
    made an error.
    """
    meta = normalize_metadata(metadata)
    true_id = int(true_id)
    pred_id = int(pred_id)
    if true_id == pred_id:
        return "correct"
    if true_id not in meta or pred_id not in meta:
        raise KeyError("Both class ids must be present in metadata.")
    t, p = meta[true_id], meta[pred_id]
    if t.skeleton_group != p.skeleton_group:
        return "skeleton_confusion"
    if t.dot_count != p.dot_count:
        return "dot_count_confusion"
    if t.dot_position != p.dot_position:
        return "dot_position_confusion"
    return "same_structure_confusion"


@dataclass
class GlyphDiagnostic:
    """Structural error diagnostics from labels and predictions.

    The core class does not require a model object. Supply ``y_true`` together
    with either predicted class ids or class-probability scores.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    probs: Optional[np.ndarray]
    metadata: Dict[int, GlyphMetadata]

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred=None,
        probs=None,
        metadata=None,
    ) -> "GlyphDiagnostic":
        y_true = np.asarray(y_true, dtype=int).reshape(-1)
        probs_arr = None if probs is None else np.asarray(probs, dtype=float)

        if y_pred is None:
            if probs_arr is None:
                raise ValueError("Provide y_pred or probs.")
            if probs_arr.ndim != 2:
                raise ValueError("probs must be a 2D array of shape [n_samples, n_classes].")
            y_pred = probs_arr.argmax(axis=1)
        y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if probs_arr is not None and len(probs_arr) != len(y_true):
            raise ValueError("probs and y_true must have the same number of samples.")

        meta = normalize_metadata(metadata)
        known = set(meta)
        observed = set(np.unique(np.concatenate([y_true, y_pred])))
        missing = observed - known
        if missing:
            raise ValueError(f"Observed class ids missing from metadata: {sorted(missing)}")

        if probs_arr is not None:
            max_id = max(meta)
            if probs_arr.shape[1] <= max_id:
                raise ValueError(
                    "Probability matrix does not contain enough columns for the metadata class ids."
                )

        return cls(y_true=y_true, y_pred=y_pred, probs=probs_arr, metadata=meta)

    @property
    def class_ids(self):
        return sorted(self.metadata)

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.y_true == self.y_pred)) if len(self.y_true) else float("nan")

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, labels=self.class_ids)

    def confusion_pairs(self) -> pd.DataFrame:
        cm = self.confusion_matrix()
        rows = []
        for i, true_id in enumerate(self.class_ids):
            for j, pred_id in enumerate(self.class_ids):
                count = int(cm[i, j])
                if true_id == pred_id or count == 0:
                    continue
                rows.append(
                    {
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "true": self.metadata[true_id].name,
                        "pred": self.metadata[pred_id].name,
                        "count": count,
                        "failure_mode": failure_mode_for_pair(true_id, pred_id, self.metadata),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["true_id", "pred_id", "true", "pred", "count", "failure_mode"]
            )
        return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    def structural_fingerprint(self) -> pd.DataFrame:
        pairs = self.confusion_pairs()
        if pairs.empty:
            return pd.DataFrame(columns=["failure_mode", "count", "share"])
        out = pairs.groupby("failure_mode", as_index=False)["count"].sum()
        out["share"] = out["count"] / out["count"].sum()
        return out.sort_values("count", ascending=False).reset_index(drop=True)

    def per_class_accuracy(self) -> pd.DataFrame:
        rows = []
        for class_id in self.class_ids:
            mask = self.y_true == class_id
            n = int(mask.sum())
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": self.metadata[class_id].name,
                    "glyph": self.metadata[class_id].glyph,
                    "n": n,
                    "accuracy": float(np.mean(self.y_pred[mask] == class_id)) if n else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def family_error_burden(self, multi_letter_only: bool = True) -> pd.DataFrame:
        groups: Dict[str, list] = {}
        for class_id, m in self.metadata.items():
            groups.setdefault(m.skeleton_group, []).append(class_id)

        rows = []
        for family, ids in sorted(groups.items()):
            if multi_letter_only and len(ids) < 2:
                continue
            mask = np.isin(self.y_true, ids)
            n = int(mask.sum())
            if not n:
                continue
            y = self.y_true[mask]
            p = self.y_pred[mask]
            wrong = p != y
            within = wrong & np.isin(p, ids)
            rows.append(
                {
                    "skeleton_family": family,
                    "n": n,
                    "family_total_error_rate": float(np.mean(wrong)),
                    "within_family_error_rate": float(np.mean(within)),
                    "within_family_share_of_family_errors": float(within.sum() / max(wrong.sum(), 1)),
                    "family_total_errors": int(wrong.sum()),
                    "within_family_errors": int(within.sum()),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        pairs = self.confusion_pairs()
        fingerprint = self.structural_fingerprint()
        total_errors = int(np.sum(self.y_pred != self.y_true))
        within = 0
        if total_errors:
            for t, p in zip(self.y_true, self.y_pred):
                if t != p and self.metadata[int(t)].skeleton_group == self.metadata[int(p)].skeleton_group:
                    within += 1
        return {
            "n_samples": int(len(self.y_true)),
            "n_classes": int(len(self.class_ids)),
            "accuracy": self.accuracy,
            "total_errors": total_errors,
            "within_skeleton_errors": int(within),
            "within_skeleton_share_of_errors": float(within / max(total_errors, 1)),
            "cross_skeleton_errors": int(total_errors - within),
            "n_nonzero_confusion_pairs": int(len(pairs)),
            "failure_modes": {
                row.failure_mode: {"count": int(row.count), "share": float(row.share)}
                for row in fingerprint.itertuples(index=False)
            },
        }

    def write_report(self, output_dir) -> Path:
        """Write a compact diagnostic report to CSV/JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.confusion_pairs().to_csv(out / "confusion_pairs.csv", index=False)
        self.structural_fingerprint().to_csv(out / "structural_fingerprint.csv", index=False)
        self.per_class_accuracy().to_csv(out / "per_class_accuracy.csv", index=False)
        self.family_error_burden().to_csv(out / "family_error_burden.csv", index=False)
        (out / "summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return out
