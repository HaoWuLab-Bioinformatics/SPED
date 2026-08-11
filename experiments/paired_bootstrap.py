#!/usr/bin/env python3
"""Paired condition-level bootstrap for two saved prediction archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pearson_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(
        np.sum(left * left, axis=1) * np.sum(right * right, axis=1) + 1e-8
    )
    return numerator / denominator


def per_condition_metrics(archive: dict, top_k: int) -> pd.DataFrame:
    y_true = archive["y_true"].astype(np.float32)
    y_pred = archive["y_pred"].astype(np.float32)
    reference = archive["reference"].astype(np.float32)
    delta_true = y_true - reference[None, :]
    delta_pred = y_pred - reference[None, :]

    top_k = min(top_k, y_true.shape[1])
    true_top = np.argpartition(np.abs(delta_true), -top_k, axis=1)[:, -top_k:]
    pred_top = np.argpartition(np.abs(delta_pred), -top_k, axis=1)[:, -top_k:]

    deg_pearson = np.empty(len(y_true), dtype=np.float64)
    deg_recall = np.empty(len(y_true), dtype=np.float64)
    for index in range(len(y_true)):
        mask = true_top[index]
        deg_pearson[index] = pearson_rows(
            delta_true[index : index + 1, mask],
            delta_pred[index : index + 1, mask],
        )[0]
        deg_recall[index] = len(
            set(true_top[index].tolist()) & set(pred_top[index].tolist())
        ) / top_k

    deg_score = np.abs(delta_true)
    deg_score /= deg_score.sum(axis=1, keepdims=True) + 1e-8
    wmse = np.sum((y_true - y_pred) ** 2 * deg_score, axis=1)

    return pd.DataFrame(
        {
            "condition": archive["conditions"].astype(str),
            "pearson_delta": pearson_rows(delta_true, delta_pred),
            "deg_pearson_delta": deg_pearson,
            "deg_recall": deg_recall,
            "wmse": wmse,
        }
    )


def load_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    left = per_condition_metrics(load_archive(args.left), args.top_k)
    right = per_condition_metrics(load_archive(args.right), args.top_k)
    merged = left.merge(right, on="condition", suffixes=("_left", "_right"))
    if merged.empty:
        raise RuntimeError("The prediction archives have no shared conditions")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(merged), size=(args.n_bootstrap, len(merged)))
    rows: list[dict] = []
    lower_is_better = {"wmse"}
    for metric in ("pearson_delta", "deg_pearson_delta", "deg_recall", "wmse"):
        raw_difference = (
            merged[f"{metric}_left"].to_numpy()
            - merged[f"{metric}_right"].to_numpy()
        )
        oriented_difference = (
            -raw_difference if metric in lower_is_better else raw_difference
        )
        bootstrap = oriented_difference[indices].mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "left": args.left_name,
                "right": args.right_name,
                "n_conditions": len(merged),
                "left_mean": merged[f"{metric}_left"].mean(),
                "right_mean": merged[f"{metric}_right"].mean(),
                "oriented_difference": oriented_difference.mean(),
                "ci_low": np.quantile(bootstrap, 0.025),
                "ci_high": np.quantile(bootstrap, 0.975),
                "bootstrap_probability_left_better": np.mean(bootstrap > 0),
            }
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    difference_columns = ["condition"]
    for metric in ("pearson_delta", "deg_pearson_delta", "deg_recall", "wmse"):
        column = f"{metric}_difference"
        merged[column] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
        difference_columns.append(column)
    merged[difference_columns].to_csv(
        args.output.with_name(f"{args.output.stem}_condition_differences.csv"),
        index=False,
    )
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(output.to_dict(orient="records"), handle, indent=2)
    print(output.to_string(index=False))
