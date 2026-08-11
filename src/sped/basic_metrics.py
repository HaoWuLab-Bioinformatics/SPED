from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rankdata_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    rx = rankdata_1d(x)
    ry = rankdata_1d(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    num = np.sum(rx * ry)
    den = math.sqrt(np.sum(rx * rx) * np.sum(ry * ry) + eps)
    return float(num / (den + eps))


def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + eps)
    return num / den


def normal_icdf(p: float) -> float:
    pp = torch.tensor(p, dtype=torch.float64)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * pp - 1.0)
    return float(z.item())


def reliability_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    levels: list[float] | None = None,
) -> pd.DataFrame:
    if levels is None:
        levels = [i / 10 for i in range(1, 10)]
    rows = []
    for lev in levels:
        z = normal_icdf((1.0 + lev) / 2.0)
        lo = y_pred - z * sigma
        hi = y_pred + z * sigma
        emp = float(np.mean((y_true >= lo) & (y_true <= hi)))
        rows.append({"level": lev, "z": z, "empirical_coverage": emp, "abs_err": abs(emp - lev)})
    return pd.DataFrame(rows)


def plot_reliability(df: pd.DataFrame, save_path: str = "reliability.png") -> None:
    import matplotlib.pyplot as plt

    x = df["level"].values
    y = df["empirical_coverage"].values
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Reliability diagram (Gaussian)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def point_prediction_metrics(
    y_true: Any,
    y_pred: Any,
    weights: Any | None = None,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    metrics = {"mse": float(np.mean((y_true_np - y_pred_np) ** 2))}
    pc = pearson_corr(y_true_np, y_pred_np)
    metrics["pearson"] = float(np.mean(pc))
    if weights is not None:
        w = _to_numpy(weights).astype(np.float64)
        metrics["wpearson"] = float(np.sum(pc * w) / (np.sum(w) + 1e-8))
    return metrics


def uncertainty_metrics(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    make_plots: bool = False,
    plot_path: str = "reliability.png",
) -> dict[str, Any]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    sigma_np = _to_numpy(sigma)

    # 1σ / 2σ coverage 是最直观的 calibration 指标：
    # 如果模型真的近似高斯且 sigma 校准良好，经验覆盖率应接近 68.27% / 95.45%。
    cov1 = float(np.mean((y_true_np >= (y_pred_np - sigma_np)) & (y_true_np <= (y_pred_np + sigma_np))))
    cov2 = float(
        np.mean((y_true_np >= (y_pred_np - 2 * sigma_np)) & (y_true_np <= (y_pred_np + 2 * sigma_np)))
    )
    calib_err = abs(cov1 - 0.6827) + abs(cov2 - 0.9545)

    # 下面几组 rho 用来回答一个更细的问题：
    # “模型给出的 sigma 是否真的在误差大的地方更大？”
    abs_err = np.abs(y_true_np - y_pred_np)
    gene_rhos = [spearman_corr(abs_err[:, j], sigma_np[:, j]) for j in range(abs_err.shape[1])]
    cond_abs = abs_err.mean(axis=1)
    cond_sig = sigma_np.mean(axis=1)
    rel_df = reliability_table(y_true_np, y_pred_np, sigma_np)
    if make_plots:
        plot_reliability(rel_df, save_path=plot_path)

    return {
        "cov1": cov1,
        "cov2": cov2,
        "calib_err": float(calib_err),
        "rel_mae": float(rel_df["abs_err"].mean()),
        "gene_rho_mean": float(np.mean(gene_rhos)),
        "gene_rho_median": float(np.median(gene_rhos)),
        "cond_rho": float(spearman_corr(cond_abs, cond_sig)),
        "global_rho": float(spearman_corr(abs_err.reshape(-1), sigma_np.reshape(-1))),
        "reliability_df": rel_df,
    }


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    weights: Any | None = None,
    sigma: Any | None = None,
    make_plots: bool = False,
    plot_path: str = "reliability.png",
) -> dict[str, Any]:
    metrics: dict[str, Any] = point_prediction_metrics(y_true, y_pred, weights=weights)
    if sigma is not None:
        metrics.update(
            uncertainty_metrics(
                y_true,
                y_pred,
                sigma,
                make_plots=make_plots,
                plot_path=plot_path,
            )
        )
    return metrics


def save_perturbation_split(
    output_path: str | Path,
    dataset_name: str,
    perturbation_names: list[str],
    train_ids: Any,
    test_ids: Any,
    seed: int,
    filter_description: str,
    n_cells: int,
    n_genes: int,
    split_type: str = "by_perturbation",
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_ids_np = sorted(int(x) for x in _to_numpy(train_ids).tolist())
    test_ids_np = sorted(int(x) for x in _to_numpy(test_ids).tolist())
    payload = {
        "dataset_name": dataset_name,
        "split_type": split_type,
        "seed": int(seed),
        "filter_description": filter_description,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_perturbations": int(len(perturbation_names)),
        "train_ids": train_ids_np,
        "test_ids": test_ids_np,
        "train_names": [perturbation_names[i] for i in train_ids_np],
        "test_names": [perturbation_names[i] for i in test_ids_np],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def load_split(split_path: str | Path) -> dict[str, Any]:
    """读取之前保存的 split JSON。"""
    return json.loads(Path(split_path).read_text())



def compute_deg_mask(
    y_true: np.ndarray,
    reference: np.ndarray,
    top_k: int = 20,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    # 差异 = 真实值 - reference，取绝对值排序
    diff = np.abs(y_true - reference)          # [n_conditions, n_genes]
    n_conditions, n_genes = diff.shape
    top_k = min(top_k, n_genes)

    deg_mask = np.zeros((n_conditions, n_genes), dtype=bool)
    top_indices = np.argpartition(diff, -top_k, axis=1)[:, -top_k:]
    for i in range(n_conditions):
        deg_mask[i, top_indices[i]] = True

    return deg_mask


def pearson_on_degs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deg_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    n_conditions = y_true.shape[0]
    corrs = np.zeros(n_conditions, dtype=np.float64)

    for i in range(n_conditions):
        mask = deg_mask[i]
        a = y_true[i, mask].astype(np.float64)
        b = y_pred[i, mask].astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        num = np.dot(a, b)
        den = np.sqrt(np.dot(a, a) * np.dot(b, b) + eps)
        corrs[i] = num / (den + eps)

    return corrs


def evaluate_predictions_with_deg(
    y_true: Any,
    y_pred: Any,
    reference: Any,
    weights: Any | None = None,
    top_k: int = 20,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true).astype(np.float32)
    y_pred_np = _to_numpy(y_pred).astype(np.float32)
    ref_np    = _to_numpy(reference).astype(np.float32)

    # 基础指标
    metrics = point_prediction_metrics(y_true_np, y_pred_np, weights=weights)

    # DEG mask（用真实值定义 DEG，避免信息泄露到预测侧）
    deg_mask = compute_deg_mask(y_true_np, ref_np, top_k=top_k)

    # 每个 condition 的 DEG Pearson
    deg_corrs = pearson_on_degs(y_true_np, y_pred_np, deg_mask)
    metrics['deg_pearson_mean'] = float(np.mean(deg_corrs))

    if weights is not None:
        w = _to_numpy(weights).astype(np.float64)
        metrics['deg_wpearson'] = float(
            np.sum(deg_corrs * w) / (np.sum(w) + 1e-8)
        )
    else:
        metrics['deg_wpearson'] = metrics['deg_pearson_mean']

    metrics['deg_pearson_per_condition'] = deg_corrs.tolist()
    metrics['top_k'] = top_k

    return metrics