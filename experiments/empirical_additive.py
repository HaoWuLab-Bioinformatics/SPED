#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    gene1: str | None
    gene2: str | None


def parse_condition(name: str) -> Condition:
    if name == "ctrl":
        return Condition(name, "control", None, None)
    if "+" in name:
        parts = name.split("+")
        genes = [part for part in parts if part != "ctrl"]
        if "ctrl" in parts and len(genes) == 1:
            return Condition(name, "single", genes[0], None)
        if len(genes) == 2:
            return Condition(name, "double", genes[0], genes[1])

    left = name.split("__", 1)[0]
    parts = [part for part in left.split("_") if not part.isdigit()]
    genes = [part for part in parts if not part.startswith("NegCtrl")]
    if not genes:
        return Condition(name, "control", None, None)
    if len(genes) == 1:
        return Condition(name, "single", genes[0], None)
    return Condition(name, "double", genes[0], genes[1])


def dense_mean(matrix) -> np.ndarray:
    mean = matrix.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1.astype(np.float32, copy=False)
    return np.asarray(mean, dtype=np.float32).reshape(-1)


def condition_means(adata, expr_key: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    labels = adata.obs["guide_identity"].astype(str).to_numpy()
    expression = adata.layers[expr_key] if expr_key in adata.layers else adata.X
    means: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name in pd.unique(labels):
        mask = labels == name
        counts[name] = int(mask.sum())
        means[name] = dense_mean(expression[mask])
    return means, counts


def weighted_profile(
    names: list[str], means: dict[str, np.ndarray], counts: dict[str, int]
) -> np.ndarray:
    if not names:
        raise ValueError("No training conditions were available for this profile")
    weights = np.asarray([counts[name] for name in names], dtype=np.float64)
    profiles = np.stack([means[name] for name in names]).astype(np.float64)
    return np.average(profiles, axis=0, weights=weights).astype(np.float32)


def load_split(path: Path) -> dict:
    with path.open() as handle:
        split = json.load(handle)
    if "train_names" not in split or "test_names" not in split:
        raise KeyError(f"Split lacks train_names/test_names: {path}")
    return split


def seed_and_budget(path: Path) -> tuple[int, int | None]:
    seed_match = re.search(r"seed(\d+)", path.name)
    budget_match = re.search(r"ndouble(\d+)", path.name)
    if seed_match is None:
        raise ValueError(f"Cannot parse seed from {path.name}")
    return int(seed_match.group(1)), (
        int(budget_match.group(1)) if budget_match else None
    )


def evaluate_split(
    split_path: Path,
    means: dict[str, np.ndarray],
    counts: dict[str, int],
    evaluate_all,
    top_k: int,
    output_dir: Path,
    save_predictions: bool,
) -> tuple[dict, pd.DataFrame]:
    split = load_split(split_path)
    split_train_names = [name for name in split["train_names"] if name in means]
    split_single_names = [name for name in split.get("single_names", []) if name in means]
    if not split_single_names:
        split_single_names = [
            name for name in means if parse_condition(name).kind == "single"
        ]
    train_names = list(dict.fromkeys(split_train_names + split_single_names))
    test_names = [name for name in split["test_names"] if name in means]

    train_meta = [parse_condition(name) for name in train_names]
    control_names = [item.name for item in train_meta if item.kind == "control"]
    reference_source = "split_train_controls"
    if not control_names:
        control_names = [
            name for name in means if parse_condition(name).kind == "control"
        ]
        reference_source = "all_controls_not_assigned_to_test"
    reference = weighted_profile(control_names, means, counts)

    singles_by_gene: dict[str, list[str]] = {}
    for item in train_meta:
        if item.kind == "single" and item.gene1 is not None:
            singles_by_gene.setdefault(item.gene1, []).append(item.name)
    single_profiles = {
        gene: weighted_profile(names, means, counts)
        for gene, names in singles_by_gene.items()
    }

    conditions: list[Condition] = []
    true_profiles: list[np.ndarray] = []
    predicted_profiles: list[np.ndarray] = []
    weights: list[int] = []
    missing: list[str] = []
    for name in test_names:
        item = parse_condition(name)
        if item.kind != "double" or item.gene1 is None or item.gene2 is None:
            continue
        absent = [gene for gene in (item.gene1, item.gene2) if gene not in single_profiles]
        if absent:
            missing.append(f"{name}:{'/'.join(absent)}")
            continue
        prediction = single_profiles[item.gene1] + single_profiles[item.gene2] - reference
        conditions.append(item)
        true_profiles.append(means[name])
        predicted_profiles.append(prediction)
        weights.append(counts[name])

    if not conditions:
        raise RuntimeError(f"No evaluable double conditions in {split_path}")

    y_true = np.stack(true_profiles).astype(np.float32)
    y_pred = np.stack(predicted_profiles).astype(np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    aggregate = evaluate_all(
        y_true, y_pred, reference, weights=weight_array, top_k=top_k
    )

    seed, budget = seed_and_budget(split_path)
    aggregate.update(
        method="Empirical additive",
        seed=seed,
        n_double_train=budget,
        split_file=split_path.name,
        n_test_conditions=len(conditions),
        n_missing_conditions=len(missing),
        reference_source=reference_source,
        n_control_conditions=len(control_names),
        n_single_conditions=len(single_profiles),
    )

    rows: list[dict] = []
    for index, item in enumerate(conditions):
        metrics = evaluate_all(
            y_true[index : index + 1],
            y_pred[index : index + 1],
            reference,
            weights=weight_array[index : index + 1],
            top_k=top_k,
        )
        rows.append(
            {
                "method": "Empirical additive",
                "seed": seed,
                "n_double_train": budget,
                "split_file": split_path.name,
                "condition": item.name,
                "gene1": item.gene1,
                "gene2": item.gene2,
                "n_cells": weights[index],
                **metrics,
            }
        )
    condition_frame = pd.DataFrame(rows)

    stem = split_path.stem
    condition_frame.to_csv(output_dir / f"{stem}_condition_metrics.csv", index=False)
    if save_predictions:
        np.savez_compressed(
            output_dir / f"{stem}_predictions.npz",
            conditions=np.asarray([item.name for item in conditions]),
            genes=np.asarray(adata_var_names),
            y_true=y_true,
            y_pred=y_pred,
            reference=reference,
            weights=weight_array,
            missing=np.asarray(missing),
        )
    return aggregate, condition_frame


def select_splits(split_dir: Path, mode: str) -> list[Path]:
    standard = sorted(split_dir.glob("norman_full_seed*_by_perturbation.json"))
    efficiency = sorted(split_dir.glob("norman_full_seed*_ndouble*.json"))
    if mode == "smoke":
        candidates = [path for path in standard if "seed0" in path.name]
        return candidates[:1]
    if mode == "standard":
        return standard
    if mode == "efficiency":
        return efficiency
    return standard + efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--adata", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expr-key", default="log_expr")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--mode", choices=("smoke", "standard", "efficiency", "all"), default="smoke"
    )
    parser.add_argument("--no-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = args.project_root.resolve()
    source_dir = project_root / "src"
    sys.path.insert(0, str(source_dir))
    from sped.metrics import evaluate_all

    adata_path = (
        args.adata
        or project_root / "data" / "Norman" / "norman_2019_full_adata.h5ad"
    )
    split_dir = args.split_dir or project_root / "protocols"
    output_dir = args.output_dir or project_root / "outputs" / "empirical_additive"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = select_splits(split_dir, args.mode)
    if not split_paths:
        raise FileNotFoundError(f"No splits found for mode={args.mode} in {split_dir}")

    print(f"Loading {adata_path}", flush=True)
    adata = sc.read_h5ad(adata_path)
    adata_var_names = adata.var_names.astype(str).to_numpy()
    print(f"Computing means for {adata.obs['guide_identity'].nunique()} conditions", flush=True)
    means, counts = condition_means(adata, args.expr_key)

    aggregate_rows: list[dict] = []
    condition_frames: list[pd.DataFrame] = []
    for split_path in split_paths:
        print(f"Evaluating {split_path.name}", flush=True)
        aggregate, condition_frame = evaluate_split(
            split_path,
            means,
            counts,
            evaluate_all,
            args.top_k,
            output_dir,
            save_predictions=not args.no_predictions,
        )
        aggregate_rows.append(aggregate)
        condition_frames.append(condition_frame)
        print(
            f"  PearsonDelta={aggregate['pearson_delta']:.4f} "
            f"DEG-PearsonDelta={aggregate['deg_pearson_delta']:.4f} "
            f"WMSE={aggregate['wmse']:.4f}",
            flush=True,
        )

    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(output_dir / f"aggregate_{args.mode}.csv", index=False)
    pd.concat(condition_frames, ignore_index=True).to_csv(
        output_dir / f"condition_metrics_{args.mode}.csv", index=False
    )
    print(f"Saved outputs to {output_dir}", flush=True)
