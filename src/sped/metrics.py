from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


# ── 基础工具 ──────────────────────────────────────────────────────────────────

def _to_np(x: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def pearson_corr(a: np.ndarray, b: np.ndarray,
                 eps: float = 1e-8) -> np.ndarray:
    """[n_conditions, n_genes] → [n_conditions] 逐条件 Pearson。"""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + eps)
    return num / den


def weighted_mean(values: np.ndarray,
                  weights: np.ndarray | None) -> float:
    if weights is None or weights.sum() < 1e-8:
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


# ── DEG mask ──────────────────────────────────────────────────────────────────

def compute_deg_mask(y_true: np.ndarray,
                     reference: np.ndarray,
                     top_k: int = 20) -> np.ndarray:
    """
    用真实值和 reference 的绝对差异找每个 condition 的 top-k DEG。
    返回 [n_conditions, n_genes] bool mask，每行恰好 top_k 个 True。
    """
    diff = np.abs(y_true - reference[None, :])
    n_conditions, n_genes = diff.shape
    top_k = min(top_k, n_genes)
    mask = np.zeros((n_conditions, n_genes), dtype=bool)
    top_idx = np.argpartition(diff, -top_k, axis=1)[:, -top_k:]
    for i in range(n_conditions):
        mask[i, top_idx[i]] = True
    return mask


def pearson_on_subset(a: np.ndarray, b: np.ndarray,
                      mask: np.ndarray,
                      eps: float = 1e-8) -> np.ndarray:
    """
    对每个 condition 只在 mask==True 的基因子集上算 Pearson。
    a, b: [n_conditions, n_genes]
    mask: [n_conditions, n_genes] bool
    返回: [n_conditions]
    """
    n = a.shape[0]
    corrs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ai = a[i, mask[i]].astype(np.float64)
        bi = b[i, mask[i]].astype(np.float64)
        ai -= ai.mean(); bi -= bi.mean()
        num = np.dot(ai, bi)
        den = np.sqrt(np.dot(ai, ai) * np.dot(bi, bi) + eps)
        corrs[i] = num / (den + eps)
    return corrs


# ── 各指标计算函数 ────────────────────────────────────────────────────────────

def _pearson_metrics(y_true, y_pred, weights):
    """全基因 Pearson / wPearson（原始表达值）。"""
    pc = pearson_corr(y_true, y_pred)
    return {
        'pearson':  float(np.mean(pc)),
        'wpearson': weighted_mean(pc, weights),
    }


def _mse_metrics(y_true, y_pred):
    """全基因 MSE。"""
    return {'mse': float(np.mean((y_true - y_pred) ** 2))}


def _r2_metrics(y_true, y_pred, weights, deg_mask=None):
    """
    R²（CPA 论文标准）：每个 condition 单独算，再平均。
    同时计算 DEG R²（在 top-k DEG 子集上）。
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum(
        (y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    out = {
        'r2':  float(np.mean(r2)),
        'wr2': weighted_mean(r2, weights),
    }
    if deg_mask is not None:
        n = y_true.shape[0]
        r2_deg = np.zeros(n)
        for i in range(n):
            yt = y_true[i, deg_mask[i]]
            yp = y_pred[i, deg_mask[i]]
            ss_r = np.sum((yt - yp) ** 2)
            ss_t = np.sum((yt - yt.mean()) ** 2)
            r2_deg[i] = 1 - ss_r / (ss_t + 1e-8)
        out['r2_deg']  = float(np.mean(r2_deg))
        out['wr2_deg'] = weighted_mean(r2_deg, weights)
    return out


def _delta_metrics(y_true, y_pred, reference, weights, deg_mask=None):
    """
    PearsonΔ：在扰动效应 delta = y - reference 上算 Pearson。
    消除 control bias，是目前最推荐的 perturbation 预测指标。

    global_mean 的 PearsonΔ 理论上接近 0，因为
    global_mean 预测所有条件相同，delta_pred ≈ 常数，方差为 0。
    """
    ref = reference[None, :]
    d_true = (y_true - ref).astype(np.float32)
    d_pred = (y_pred - ref).astype(np.float32)

    pc = pearson_corr(d_true, d_pred)
    out = {
        'pearson_delta':  float(np.mean(pc)),
        'wpearson_delta': weighted_mean(pc, weights),
    }

    # DEG PearsonΔ
    if deg_mask is not None:
        deg_pc = pearson_on_subset(d_true, d_pred, deg_mask)
        out['deg_pearson_delta']  = float(np.mean(deg_pc))
        out['deg_wpearson_delta'] = weighted_mean(deg_pc, weights)

    # delta MSE（只看扰动效应部分的误差）
    out['mse_delta'] = float(np.mean((d_true - d_pred) ** 2))

    return out


def _wmse_metrics(y_true, y_pred, reference, deg_mask):
    """
    DEG-score 加权 MSE（WMSE）。
    对每个基因赋予权重 = |y_true - reference|（DEG score），
    使得预测误差在差异大的基因上被放大。
    """
    ref = reference[None, :]
    deg_score = np.abs(y_true - ref)               # [n_cond, n_genes]
    deg_score = deg_score / (deg_score.sum(axis=1, keepdims=True) + 1e-8)
    sq_err = (y_true - y_pred) ** 2
    wmse_per_cond = (sq_err * deg_score).sum(axis=1)
    return {'wmse': float(np.mean(wmse_per_cond))}


def _direction_accuracy(y_true, y_pred, reference, weights, deg_mask=None):
    """
    方向准确率：预测的扰动方向（上调 / 下调）与真实方向的一致率。
    分别计算全基因和 DEG 子集。
    """
    ref = reference[None, :]
    dir_true = np.sign(y_true - ref)   # +1 / -1 / 0
    dir_pred = np.sign(y_pred - ref)

    # 只统计真实有变化的基因（排除 dir_true==0 的情况）
    valid = dir_true != 0
    match = (dir_true == dir_pred) & valid

    acc_per_cond = match.sum(axis=1) / (valid.sum(axis=1) + 1e-8)
    out = {
        'direction_acc':  float(np.mean(acc_per_cond)),
        'wdirection_acc': weighted_mean(acc_per_cond, weights),
    }

    if deg_mask is not None:
        acc_deg = np.zeros(len(y_true))
        for i in range(len(y_true)):
            m = deg_mask[i]
            v = valid[i, m]
            if v.sum() == 0:
                acc_deg[i] = float('nan')
            else:
                acc_deg[i] = match[i, m].sum() / v.sum()
        valid_mask = ~np.isnan(acc_deg)
        out['deg_direction_acc'] = float(np.nanmean(acc_deg))
        out['deg_wdirection_acc'] = (
            weighted_mean(acc_deg[valid_mask], weights[valid_mask])
            if weights is not None else out['deg_direction_acc']
        )

    return out


def _deg_recall(y_true, y_pred, reference, top_k=20):
    """
    DEG recall @ k：
    真实 top-k DEG 中，有多少出现在预测 top-k DEG 里。
    衡量模型是否能找对差异表达基因。
    """
    ref = reference[None, :]
    true_diff = np.abs(y_true - ref)
    pred_diff = np.abs(y_pred - ref)

    n = y_true.shape[0]
    top_k = min(top_k, y_true.shape[1])
    recalls = np.zeros(n)

    for i in range(n):
        true_top = set(np.argpartition(true_diff[i], -top_k)[-top_k:])
        pred_top = set(np.argpartition(pred_diff[i], -top_k)[-top_k:])
        recalls[i] = len(true_top & pred_top) / top_k

    return {
        'deg_recall': float(np.mean(recalls)),
        'deg_recall_std': float(np.std(recalls)),
    }


def _deg_pearson_metrics(y_true, y_pred, weights, deg_mask):
    """全基因 Pearson 基础上的 DEG 子集版本（原有指标，保持兼容）。"""
    deg_pc = pearson_on_subset(y_true, y_pred, deg_mask)
    return {
        'deg_pearson':  float(np.mean(deg_pc)),
        'deg_wpearson': weighted_mean(deg_pc, weights),
    }


def evaluate_all(
    y_true: Any,
    y_pred: Any,
    reference: Any,
    weights: Any | None = None,
    top_k: int = 20,
) -> dict[str, float]:
    y_true = _to_np(y_true).astype(np.float32)
    y_pred = _to_np(y_pred).astype(np.float32)
    ref    = _to_np(reference).astype(np.float32)
    w      = (_to_np(weights).astype(np.float64)
              if weights is not None else None)

    deg_mask = compute_deg_mask(y_true, ref, top_k=top_k)

    metrics: dict[str, float] = {}
    metrics['top_k'] = top_k

    metrics.update(_pearson_metrics(y_true, y_pred, w))
    metrics.update(_mse_metrics(y_true, y_pred))
    metrics.update(_deg_pearson_metrics(y_true, y_pred, w, deg_mask))
    metrics.update(_delta_metrics(y_true, y_pred, ref, w, deg_mask))
    metrics.update(_r2_metrics(y_true, y_pred, w, deg_mask))
    metrics.update(_wmse_metrics(y_true, y_pred, ref, deg_mask))
    metrics.update(_direction_accuracy(y_true, y_pred, ref, w, deg_mask))
    metrics.update(_deg_recall(y_true, y_pred, ref, top_k=top_k))

    return metrics



def print_metrics(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    groups = [
        ("基础指标",   ['pearson', 'wpearson', 'mse',
                        'deg_pearson', 'deg_wpearson']),
        ("PearsonΔ(推荐)", ['pearson_delta', 'wpearson_delta',
                               'deg_pearson_delta', 'mse_delta']),
        ("R²",         ['r2', 'wr2', 'r2_deg']),
        ("WMSE",       ['wmse']),
        ("方向准确率", ['direction_acc', 'deg_direction_acc']),
        ("DEG Recall", ['deg_recall']),
    ]

    for group_name, keys in groups:
        vals = {k: metrics[k] for k in keys if k in metrics}
        if not vals:
            continue
        print(f"\n  [{group_name}]")
        for k, v in vals.items():
            print(f"    {k:<25} {v:.4f}")



def summarize_results(records: list[dict],
                      group_by: str = 'method') -> 'pd.DataFrame':
    import pandas as pd
    df = pd.DataFrame(records)

    metric_cols = [c for c in df.columns
                   if c not in [group_by, 'seed', 'top_k', 'tag',
                                 'n_double_train', 'dataset_name',
                                 'split_type', 'heldout_family',
                                 'train_names', 'test_names',
                                 'single_names']]

    agg = {}
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col + '_mean'] = (col, 'mean')
            agg[col + '_std']  = (col, 'std')

    summary = df.groupby(group_by).agg(**agg).round(4)
    return summary