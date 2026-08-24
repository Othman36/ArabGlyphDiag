"""Command-line entry point for ArabGlyphDiag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostic import GlyphDiagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arabglyphdiag")
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose = sub.add_parser("diagnose", help="Generate structural diagnostics from a CSV file.")
    diagnose.add_argument("input", help="Input CSV containing labels/predictions.")
    diagnose.add_argument("--true-col", default="y_true")
    diagnose.add_argument("--pred-col", default="y_pred")
    diagnose.add_argument(
        "--proba-prefix",
        default=None,
        help="Optional probability-column prefix, e.g. p_ for p_0,p_1,...",
    )
    diagnose.add_argument("--output-dir", default="arabglyphdiag_report")
    return parser


def _probability_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    def suffix_num(name):
        suffix = name[len(prefix):]
        try:
            return int(suffix)
        except ValueError:
            return 10**9
    return sorted(cols, key=suffix_num)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        df = pd.read_csv(args.input)
        if args.true_col not in df.columns:
            raise SystemExit(f"Missing true-label column: {args.true_col}")

        probs = None
        y_pred = None
        if args.proba_prefix:
            pcols = _probability_columns(df, args.proba_prefix)
            if not pcols:
                raise SystemExit(f"No probability columns found with prefix {args.proba_prefix!r}")
            probs = df[pcols].to_numpy(dtype=float)
        elif args.pred_col in df.columns:
            y_pred = df[args.pred_col].to_numpy(dtype=int)
        else:
            raise SystemExit(
                f"Provide column {args.pred_col!r} or use --proba-prefix for probability columns."
            )

        diagnostic = GlyphDiagnostic.from_predictions(
            y_true=df[args.true_col].to_numpy(dtype=int),
            y_pred=y_pred,
            probs=probs,
        )
        out = diagnostic.write_report(args.output_dir)
        print(json.dumps(diagnostic.summary(), indent=2))
        print(f"Report written to: {Path(out).resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
