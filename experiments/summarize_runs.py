#!/usr/bin/env python3
"""Summarize repeated runs with mean, SD, and t-based 95% confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


DEFAULT_GROUP_COLUMNS = (
    "method",
    "n_double_train",
    "lambda_single",
    "sampling",
)

EXCLUDED_NUMERIC_COLUMNS = {
    "top_k",
    "seed",
    "split_seed",
    "init_seed",
    "epochs",
    "n_train_single",
    "n_train_double",
    "n_test_conditions",
    "n_missing_conditions",
    "n_control_conditions",
    "n_single_conditions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_COLUMNS))
    parser.add_argument(
        "--uncertainty-source",
        default="unspecified",
        choices=("split", "initialization", "mixed", "unspecified"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    group_columns = [column for column in args.group_by if column in frame.columns]
    metric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_NUMERIC_COLUMNS and column not in group_columns
    ]
    if not metric_columns:
        raise ValueError("No numeric metric columns were found")

    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    rows: list[dict] = []
    for key, subset in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(group_columns, key))
        for metric in metric_columns:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            if len(values) > 1:
                half_width = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
                ci_low, ci_high = mean - half_width, mean + half_width
            else:
                ci_low = ci_high = float("nan")
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "n_runs": len(values),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "uncertainty_source": args.uncertainty_source,
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
