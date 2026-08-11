#!/usr/bin/env python3
"""Matched-architecture SPED loss ablation on saved Norman standard splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_full_conditions(adata) -> pd.DataFrame:
    rows: list[dict] = []
    labels = adata.obs["guide_identity"].astype(str)
    counts = labels.value_counts()
    for guide, n_cells in counts.items():
        if guide == "ctrl":
            kind, gene1, gene2 = "control", None, None
        else:
            parts = guide.split("+")
            genes = [part for part in parts if part != "ctrl"]
            if "ctrl" in parts and len(genes) == 1:
                kind, gene1, gene2 = "single", genes[0], None
            elif len(genes) == 2:
                kind, gene1, gene2 = "double", genes[0], genes[1]
            else:
                continue
        rows.append(
            {
                "guide_identity": guide,
                "condition_type": kind,
                "gene1": gene1,
                "gene2": gene2,
                "n_cells": int(n_cells),
            }
        )
    return pd.DataFrame(rows)


def train_fixed_epochs(
    model,
    loader,
    reference: np.ndarray,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    lambda_single: float,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
    rows: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        single_sum = 0.0
        double_sum = 0.0
        n_single = 0
        n_double = 0
        for gene1, gene2, type_id, _, target in loader:
            gene1 = gene1.to(device)
            gene2 = gene2.to(device)
            type_id = type_id.to(device)
            target = target.to(device)
            prediction = model(
                gene1,
                gene2,
                type_id,
                reference_tensor.unsqueeze(0).expand(len(gene1), -1),
            )
            per_cell = torch.mean((prediction - target) ** 2, dim=1)
            single_mask = type_id == 0
            double_mask = type_id == 1
            loss = prediction.sum() * 0.0
            has_objective = False
            if single_mask.any():
                single_loss = per_cell[single_mask].mean()
                loss = loss + lambda_single * single_loss
                single_sum += float(per_cell[single_mask].sum().item())
                n_single += int(single_mask.sum().item())
                has_objective = has_objective or lambda_single > 0
            if double_mask.any():
                double_loss = per_cell[double_mask].mean()
                loss = loss + double_loss
                double_sum += float(per_cell[double_mask].sum().item())
                n_double += int(double_mask.sum().item())
                has_objective = True
            if not has_objective:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        rows.append(
            {
                "epoch": epoch,
                "single_loss": single_sum / max(n_single, 1),
                "double_loss": double_sum / max(n_double, 1),
            }
        )
        print(
            f"epoch={epoch:03d} lambda_single={lambda_single:g} "
            f"single={rows[-1]['single_loss']:.6f} "
            f"double={rows[-1]['double_loss']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def condition_metric_rows(
    names: list[str], y_true: np.ndarray, y_pred: np.ndarray,
    reference: np.ndarray, weights: np.ndarray, evaluate_all,
    split_seed: int, init_seed: int, lambda_single: float,
    sampling: str, top_k: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, name in enumerate(names):
        metrics = evaluate_all(
            y_true[index : index + 1], y_pred[index : index + 1], reference,
            weights=weights[index : index + 1], top_k=top_k,
        )
        rows.append(
            {
                "condition": name,
                "seed": split_seed,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "lambda_single": lambda_single,
                "sampling": sampling,
                "n_cells": int(weights[index]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--adata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--init-seeds",
        type=int,
        nargs="+",
        help="Initialization seeds; by default each split uses its matching seed",
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--effect-hidden", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sampling", choices=("cell", "condition"), default="cell"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.project_root.resolve()
    source_dir = root / "src"
    sys.path.insert(0, str(source_dir))
    from sped.metrics import evaluate_all
    from sped.model import (
        AdditiveOnlyModel,
        PerturbationCellDataset,
        build_gene_vocab,
        compute_global_reference,
        eval_interaction_model,
    )

    if args.smoke:
        args.seeds = args.seeds[:1]
        if args.init_seeds:
            args.init_seeds = args.init_seeds[:1]
        args.lambdas = args.lambdas[:2]
        args.epochs = min(args.epochs, 2)

    adata_path = (
        args.adata
        or root / "data" / "Norman" / "norman_2019_full_adata.h5ad"
    )
    output_dir = args.output_dir or root / "outputs" / "loss_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(adata_path)
    conditions = parse_full_conditions(adata)
    vocabulary, gene_to_id = build_gene_vocab(conditions)
    reference = compute_global_reference(adata, conditions, expr_key="log_expr")

    aggregate_rows: list[dict] = []
    for split_seed in args.seeds:
        split_path = root / "protocols" / f"norman_full_seed{split_seed}_by_perturbation.json"
        with split_path.open() as handle:
            split = json.load(handle)
        train_names = list(
            dict.fromkeys(split.get("single_names", []) + split["train_names"])
        )
        test_names = split["test_names"]
        train_dataset = PerturbationCellDataset(
            adata, conditions, gene_to_id, train_names, expr_key="log_expr"
        )
        test_dataset = PerturbationCellDataset(
            adata, conditions, gene_to_id, test_names, expr_key="log_expr"
        )

        init_seeds = args.init_seeds or [split_seed]
        for init_seed in init_seeds:
          for lambda_single in args.lambdas:
            tag = (
                f"split{split_seed}_init{init_seed}_lambda{lambda_single:g}"
            )
            run_aggregate_path = output_dir / f"{tag}_aggregate.csv"
            if run_aggregate_path.exists() and not args.no_resume:
                cached = pd.read_csv(run_aggregate_path).iloc[0].to_dict()
                aggregate_rows.append(cached)
                print(f"resuming: skip completed {tag}", flush=True)
                continue
            set_seed(init_seed)
            generator = torch.Generator().manual_seed(init_seed)
            sampler = None
            shuffle = True
            if args.sampling == "condition":
                condition_counts = np.bincount(train_dataset.condition_idxs)
                sample_weights = 1.0 / condition_counts[train_dataset.condition_idxs]
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(train_dataset),
                    replacement=True,
                    generator=generator,
                )
                shuffle = False
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                sampler=sampler, generator=generator, num_workers=0,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            model = AdditiveOnlyModel(
                n_genes=len(vocabulary), out_dim=adata.n_vars,
                emb_dim=args.embedding_dim, effect_hidden=args.effect_hidden,
            ).to(args.device)
            history = train_fixed_epochs(
                model, train_loader, reference, args.device, args.epochs,
                args.learning_rate, args.weight_decay, lambda_single,
            )
            history.to_csv(output_dir / f"{tag}_history.csv", index=False)

            arrays = eval_interaction_model(
                model, test_loader, args.device, reference,
                condition_type_filter=1, return_arrays=True,
            )
            y_true = arrays.pop("y_true")
            y_pred = arrays.pop("y_pred")
            weights = arrays.pop("weights")
            metrics = evaluate_all(
                y_true, y_pred, reference, weights=weights, top_k=args.top_k
            )
            metrics.update(
                method=(
                    "SPED (Ldouble only)"
                    if lambda_single == 0
                    else "SPED (Lsingle + Ldouble)"
                ),
                seed=split_seed,
                split_seed=split_seed,
                init_seed=init_seed,
                lambda_single=lambda_single,
                epochs=args.epochs,
                sampling=args.sampling,
                n_train_single=len(split.get("single_names", [])),
                n_train_double=len(split["train_names"]),
                n_test_conditions=len(y_true),
            )
            aggregate_rows.append(metrics)
            pd.DataFrame([metrics]).to_csv(run_aggregate_path, index=False)
            condition_names = test_dataset.unique_guides[: len(y_true)]
            condition_metric_rows(
                condition_names, y_true, y_pred, reference, weights,
                evaluate_all, split_seed, init_seed, lambda_single,
                args.sampling, args.top_k,
            ).to_csv(output_dir / f"{tag}_condition_metrics.csv", index=False)
            np.savez_compressed(
                output_dir / f"{tag}_predictions.npz",
                conditions=np.asarray(condition_names),
                genes=adata.var_names.astype(str).to_numpy(),
                y_true=y_true,
                y_pred=y_pred,
                reference=reference,
                weights=weights,
            )
            torch.save(model.state_dict(), output_dir / f"{tag}_model.pt")

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate_loss_ablation.csv", index=False)
    print(aggregate.to_string(index=False))
