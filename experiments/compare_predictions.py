#!/usr/bin/env python3
"""Compare matched SPED and empirical-additive predictions across split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paired_bootstrap import load_archive, per_condition_metrics


METRICS = ("pearson_delta", "deg_pearson_delta", "deg_recall", "wmse")
LOWER_IS_BETTER = {"wmse"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--additive-dir", type=Path)
    parser.add_argument("--sped-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--lambda-single", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def oriented(values: np.ndarray, metric: str) -> np.ndarray:
    return -values if metric in LOWER_IS_BETTER else values


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


if __name__ == "__main__":
    args = parse_args()
    root = args.root.resolve()
    additive_dir = args.additive_dir or root / "outputs" / "empirical_additive"
    sped_dir = args.sped_dir or root / "outputs" / "loss_ablation"
    output_dir = args.output_dir or root / "outputs" / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    seed_frames: dict[int, pd.DataFrame] = {}
    seed_rows: list[dict] = []
    difference_rows: list[pd.DataFrame] = []

    for seed in args.seeds:
        additive_path = additive_dir / (
            f"norman_full_seed{seed}_by_perturbation_predictions.npz"
        )
        sped_path = sped_dir / (
            f"split{seed}_init{seed}_lambda{args.lambda_single:g}_predictions.npz"
        )
        additive = per_condition_metrics(load_archive(additive_path), args.top_k)
        sped = per_condition_metrics(load_archive(sped_path), args.top_k)
        merged = sped.merge(additive, on="condition", suffixes=("_sped", "_additive"))
        if merged.empty:
            raise RuntimeError(f"No matched conditions for split seed {seed}")
        merged.insert(0, "split_seed", seed)
        seed_frames[seed] = merged

        indices = rng.integers(0, len(merged), size=(args.n_bootstrap, len(merged)))
        differences = pd.DataFrame({"split_seed": seed, "condition": merged["condition"]})
        for metric in METRICS:
            raw = (
                merged[f"{metric}_sped"].to_numpy()
                - merged[f"{metric}_additive"].to_numpy()
            )
            gain = oriented(raw, metric)
            boot = gain[indices].mean(axis=1)
            ci_low, ci_high = percentile_interval(boot)
            seed_rows.append(
                {
                    "split_seed": seed,
                    "metric": metric,
                    "n_conditions": len(merged),
                    "sped_mean": merged[f"{metric}_sped"].mean(),
                    "additive_mean": merged[f"{metric}_additive"].mean(),
                    "oriented_sped_gain": gain.mean(),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_probability_sped_better": float(np.mean(boot > 0)),
                }
            )
            differences[f"{metric}_sped_minus_additive"] = raw
        difference_rows.append(differences)

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(output_dir / "paired_bootstrap_by_split.csv", index=False)
    pd.concat(difference_rows, ignore_index=True).to_csv(
        output_dir / "condition_differences.csv", index=False
    )

    hierarchical_rows: list[dict] = []
    seed_values = np.asarray(args.seeds)
    for metric in METRICS:
        observed_seed_means: list[float] = []
        for seed in args.seeds:
            frame = seed_frames[seed]
            raw = (
                frame[f"{metric}_sped"].to_numpy()
                - frame[f"{metric}_additive"].to_numpy()
            )
            observed_seed_means.append(float(oriented(raw, metric).mean()))

        hierarchical = np.empty(args.n_bootstrap, dtype=np.float64)
        for iteration in range(args.n_bootstrap):
            sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
            sampled_seed_means: list[float] = []
            for seed in sampled_seeds:
                frame = seed_frames[int(seed)]
                raw = (
                    frame[f"{metric}_sped"].to_numpy()
                    - frame[f"{metric}_additive"].to_numpy()
                )
                gain = oriented(raw, metric)
                sampled = rng.integers(0, len(gain), size=len(gain))
                sampled_seed_means.append(float(gain[sampled].mean()))
            hierarchical[iteration] = np.mean(sampled_seed_means)

        ci_low, ci_high = percentile_interval(hierarchical)
        hierarchical_rows.append(
            {
                "metric": metric,
                "n_split_seeds": len(args.seeds),
                "conditions_per_split": int(len(seed_frames[args.seeds[0]])),
                "mean_oriented_sped_gain": float(np.mean(observed_seed_means)),
                "between_split_sd": (
                    float(np.std(observed_seed_means, ddof=1))
                    if len(observed_seed_means) > 1 else float("nan")
                ),
                "hierarchical_ci95_low": ci_low,
                "hierarchical_ci95_high": ci_high,
                "bootstrap_probability_sped_better": float(
                    np.mean(hierarchical > 0)
                ),
            }
        )

    hierarchical_summary = pd.DataFrame(hierarchical_rows)
    hierarchical_summary.to_csv(
        output_dir / "hierarchical_bootstrap_summary.csv", index=False
    )
    with (output_dir / "comparison_metadata.json").open("w") as handle:
        json.dump(vars(args) | {"root": str(root), "output_dir": str(output_dir)}, handle, indent=2, default=str)

    print("Per-split paired bootstrap")
    print(seed_summary.to_string(index=False))
    print("\nHierarchical bootstrap across split seeds and conditions")
    print(hierarchical_summary.to_string(index=False))
