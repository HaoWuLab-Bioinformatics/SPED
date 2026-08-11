from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any
import scanpy as sc

from .basic_metrics import evaluate_predictions


# ─────────────────────────────────────────────────────────────────────────────
# 数据解析工具
# ─────────────────────────────────────────────────────────────────────────────

def parse_norman_conditions(adata: sc.AnnData) -> pd.DataFrame:
    rows = []
    for guide in adata.obs['guide_identity'].unique():
        n_cells = (adata.obs['guide_identity'] == guide).sum()
        left = guide.split('__')[0]
        parts = left.split('_')
        # 去掉末尾数字后缀
        parts = [p for p in parts if not p.isdigit()]
        is_neg = [p.startswith('NegCtrl') for p in parts]

        if all(is_neg):
            rows.append({
                'guide_identity': guide,
                'condition_type': 'control',
                'gene1': None,
                'gene2': None,
                'n_cells': n_cells,
            })
        elif any(is_neg):
            gene = [p for p in parts if not p.startswith('NegCtrl')][0]
            rows.append({
                'guide_identity': guide,
                'condition_type': 'single',
                'gene1': gene,
                'gene2': None,
                'n_cells': n_cells,
            })
        else:
            # double：取前两个非数字 part
            gene_parts = [p for p in parts if not p.startswith('NegCtrl')]
            if len(gene_parts) >= 2:
                rows.append({
                    'guide_identity': guide,
                    'condition_type': 'double',
                    'gene1': gene_parts[0],
                    'gene2': gene_parts[1],
                    'n_cells': n_cells,
                })

    return pd.DataFrame(rows)


def build_gene_vocab(condition_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """
    从 condition DataFrame 里提取所有扰动基因的词表。
    包含一个特殊 token 'ctrl' 用于 single 条件的第二个基因位置。
    """
    genes = set()
    genes.add('ctrl')
    for _, row in condition_df.iterrows():
        if row['gene1'] is not None:
            genes.add(row['gene1'])
        if row['gene2'] is not None:
            genes.add(row['gene2'])
    vocab = sorted(genes)
    gene_to_id = {g: i for i, g in enumerate(vocab)}
    return vocab, gene_to_id


def compute_global_reference(
    adata: sc.AnnData,
    condition_df: pd.DataFrame,
    expr_key: str = 'log_expr',
) -> np.ndarray:
    """
    用 NegCtrl 条件的所有细胞计算全局 reference profile。
    """
    ctrl_guides = condition_df[condition_df['condition_type'] == 'control']['guide_identity'].tolist()
    ctrl_mask = adata.obs['guide_identity'].isin(ctrl_guides)
    if expr_key in adata.layers:
        ctrl_expr = adata.layers[expr_key][ctrl_mask]
    else:
        ctrl_expr = adata.X[ctrl_mask]
    if hasattr(ctrl_expr, 'toarray'):
        ctrl_expr = ctrl_expr.toarray()
    return np.asarray(ctrl_expr.mean(axis=0), dtype=np.float32).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PerturbationCellDataset(Dataset):

    def __init__(
        self,
        adata: sc.AnnData,
        condition_df: pd.DataFrame,
        gene_to_id: dict[str, int],
        split_guides: list[str],       # 这个 split 里包含的 guide_identity
        expr_key: str = 'log_expr',
    ):
        self.gene_to_id = gene_to_id
        ctrl_id = gene_to_id['ctrl']

        # 只保留 split_guides 里的细胞
        mask = adata.obs['guide_identity'].isin(split_guides)
        sub_adata = adata[mask]

        if expr_key in sub_adata.layers:
            X = sub_adata.layers[expr_key]
        else:
            X = sub_adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.X = np.asarray(X, dtype=np.float32)

        # 为每个细胞构建 (gene1_id, gene2_id, type_id, condition_idx)
        guide_series = sub_adata.obs['guide_identity'].values
        cond_info = condition_df.set_index('guide_identity')

        # condition_idx 映射
        unique_guides = list(dict.fromkeys(split_guides))  # 保序去重
        guide_to_cond_idx = {g: i for i, g in enumerate(unique_guides)}

        self.gene1_ids = np.zeros(len(self.X), dtype=np.int64)
        self.gene2_ids = np.zeros(len(self.X), dtype=np.int64)
        self.type_ids = np.zeros(len(self.X), dtype=np.int64)
        self.condition_idxs = np.zeros(len(self.X), dtype=np.int64)

        for i, guide in enumerate(guide_series):
            row = cond_info.loc[guide]
            g1 = row['gene1'] if row['gene1'] is not None else 'ctrl'
            g2 = row['gene2'] if row['gene2'] is not None else 'ctrl'
            self.gene1_ids[i] = gene_to_id.get(g1, ctrl_id)
            self.gene2_ids[i] = gene_to_id.get(g2, ctrl_id)
            self.type_ids[i] = 0 if row['condition_type'] == 'single' else 1
            self.condition_idxs[i] = guide_to_cond_idx.get(guide, 0)

        self.unique_guides = unique_guides

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.gene1_ids[i], dtype=torch.long),
            torch.tensor(self.gene2_ids[i], dtype=torch.long),
            torch.tensor(self.type_ids[i], dtype=torch.long),
            torch.tensor(self.condition_idxs[i], dtype=torch.long),
            torch.tensor(self.X[i], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────────────────────────────────────

class InteractionPerturbationModel(nn.Module):

    def __init__(
        self,
        n_genes: int,
        out_dim: int,
        emb_dim: int = 64,
        effect_hidden: int = 512,
        interaction_hidden: int = 256,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # 共享 gene embedding
        self.gene_emb = nn.Embedding(n_genes, emb_dim)

        # Effect decoder：gene embedding → perturbation effect
        self.effect_decoder = nn.Sequential(
            nn.Linear(emb_dim, effect_hidden),
            nn.ReLU(),
            nn.Linear(effect_hidden, effect_hidden),
            nn.ReLU(),
            nn.Linear(effect_hidden, out_dim),
        )

        # Interaction module：捕捉非线性的基因间交互
        # 输入：concat(emb_A, emb_B, emb_A*emb_B, |emb_A-emb_B|)
        # 这个特征设计保证了 order-invariant（A+B = B+A）
        interaction_in = emb_dim * 4
        self.interaction_module = nn.Sequential(
            nn.Linear(interaction_in, interaction_hidden),
            nn.ReLU(),
            nn.Linear(interaction_hidden, interaction_hidden),
            nn.ReLU(),
            nn.Linear(interaction_hidden, out_dim),
        )

        # interaction 的初始输出接近 0
        # 让模型在训练初期先学好 effect，再逐步学 interaction
        nn.init.zeros_(self.interaction_module[-1].weight)
        nn.init.zeros_(self.interaction_module[-1].bias)

    def get_effect(self, gene_id: torch.Tensor) -> torch.Tensor:
        """计算单个基因的扰动效应向量。"""
        emb = self.gene_emb(gene_id)
        return self.effect_decoder(emb)

    def get_interaction(
        self,
        gene1_id: torch.Tensor,
        gene2_id: torch.Tensor,
    ) -> torch.Tensor:
        """计算两个基因之间的交互效应向量。"""
        e1 = self.gene_emb(gene1_id)
        e2 = self.gene_emb(gene2_id)
        # order-invariant 特征
        feat = torch.cat([
            e1 + e2,           # 对称
            torch.abs(e1 - e2),  # 对称
            e1 * e2,           # 对称
            (e1 + e2) / 2,     # 对称（均值）
        ], dim=-1)
        return self.interaction_module(feat)

    def forward(
        self,
        gene1_id: torch.Tensor,
        gene2_id: torch.Tensor,
        condition_type: torch.Tensor,  # 0=single, 1=double
        reference: torch.Tensor,       # [B, G] 全局 reference profile
    ) -> torch.Tensor:
        """
        condition_type=0（single）：pred = reference + effect(gene1)
        condition_type=1（double）：pred = reference + effect(gene1)
                                              + effect(gene2)
                                              + interaction(gene1, gene2)
        """
        effect_1 = self.get_effect(gene1_id)   # [B, G]

        # single 条件：gene2 是 ctrl token，effect 和 interaction 置零
        is_double = (condition_type == 1).float().unsqueeze(1)  # [B, 1]

        effect_2 = self.get_effect(gene2_id) * is_double
        interaction = self.get_interaction(gene1_id, gene2_id) * is_double

        pred = reference + effect_1 + effect_2 + interaction
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 评测工具（condition-level，与 eval_protocol 兼容）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_interaction_model(
    model: InteractionPerturbationModel,
    loader: DataLoader,
    device: str,
    reference: np.ndarray,
    condition_type_filter: Optional[int] = None,  # None=全部, 0=single, 1=double
    return_arrays: bool = False,  # True 时额外返回 y_true/y_pred/weights 矩阵
) -> dict[str, Any]:
    model.eval()
    reference_t = torch.tensor(reference, dtype=torch.float32, device=device)

    # 收集每个 condition 的预测和真实值
    cond_preds: dict[int, list] = {}
    cond_trues: dict[int, list] = {}
    cond_types: dict[int, int] = {}

    for batch in loader:
        g1, g2, type_id, cond_idx, y = batch
        g1 = g1.to(device)
        g2 = g2.to(device)
        type_id = type_id.to(device)
        y = y.to(device)

        ref = reference_t.unsqueeze(0).expand(len(g1), -1)
        pred = model(g1, g2, type_id, ref)

        for i in range(len(g1)):
            ci = int(cond_idx[i].item())
            ti = int(type_id[i].item())
            if ci not in cond_preds:
                cond_preds[ci] = []
                cond_trues[ci] = []
                cond_types[ci] = ti
            cond_preds[ci].append(pred[i].cpu().numpy())
            cond_trues[ci].append(y[i].cpu().numpy())

    # 聚合成 condition-level mean
    pred_means = []
    true_means = []
    weights = []

    for ci in sorted(cond_preds.keys()):
        if condition_type_filter is not None and cond_types[ci] != condition_type_filter:
            continue
        pred_means.append(np.stack(cond_preds[ci]).mean(axis=0))
        true_means.append(np.stack(cond_trues[ci]).mean(axis=0))
        weights.append(len(cond_preds[ci]))

    if len(pred_means) == 0:
        return {'mse': float('nan'), 'pearson': float('nan'), 'wpearson': float('nan')}

    y_true_mat = np.stack(true_means)   # [n_conditions, n_genes]
    y_pred_mat = np.stack(pred_means)   # [n_conditions, n_genes]
    w_arr      = np.array(weights, dtype=np.float32)

    metrics = evaluate_predictions(y_true_mat, y_pred_mat, weights=w_arr)

    if return_arrays:
        metrics['y_true']   = y_true_mat
        metrics['y_pred']   = y_pred_mat
        metrics['weights']  = w_arr

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def train_interaction_model(
    model: InteractionPerturbationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    reference: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_single: float = 1.0,   # single loss 的权重
    lambda_double: float = 1.0,   # double loss 的权重
) -> pd.DataFrame:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    reference_t = torch.tensor(reference, dtype=torch.float32, device=device)
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_single_loss = 0.0
        total_double_loss = 0.0
        n_single = 0
        n_double = 0

        for batch in train_loader:
            g1, g2, type_id, cond_idx, y = batch
            g1 = g1.to(device)
            g2 = g2.to(device)
            type_id = type_id.to(device)
            y = y.to(device)

            opt.zero_grad()

            ref = reference_t.unsqueeze(0).expand(len(g1), -1)
            pred = model(g1, g2, type_id, ref)

            # 分开计算 single 和 double 的 loss
            single_mask = (type_id == 0)
            double_mask = (type_id == 1)

            loss = torch.tensor(0.0, device=device)

            if single_mask.any():
                loss_s = ((pred[single_mask] - y[single_mask]) ** 2).mean()
                loss = loss + lambda_single * loss_s
                total_single_loss += float(loss_s.item()) * single_mask.sum().item()
                n_single += single_mask.sum().item()

            if double_mask.any():
                loss_d = ((pred[double_mask] - y[double_mask]) ** 2).mean()
                loss = loss + lambda_double * loss_d
                total_double_loss += float(loss_d.item()) * double_mask.sum().item()
                n_double += double_mask.sum().item()

            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # 评测：分别看 single 和 double 的 condition-level 指标
        metrics_all = eval_interaction_model(model, test_loader, device, reference)
        metrics_double = eval_interaction_model(model, test_loader, device, reference,
                                                condition_type_filter=1)

        avg_single_loss = total_single_loss / (n_single + 1e-8)
        avg_double_loss = total_double_loss / (n_double + 1e-8)

        row = {
            'epoch': ep,
            'train_loss_single': avg_single_loss,
            'train_loss_double': avg_double_loss,
            'test_MSE_all': metrics_all['mse'],
            'test_Pearson_all': metrics_all['pearson'],
            'test_wPearson_all': metrics_all['wpearson'],
            'test_MSE_double': metrics_double['mse'],
            'test_Pearson_double': metrics_double['pearson'],
            'test_wPearson_double': metrics_double['wpearson'],
        }
        history.append(row)

        print(
            f"Epoch {ep:02d} | "
            f"loss_s={avg_single_loss:.4f} loss_d={avg_double_loss:.4f} | "
            f"double → MSE={metrics_double['mse']:.4f} "
            f"Pearson={metrics_double['pearson']:.4f} "
            f"wPearson={metrics_double['wpearson']:.4f}"
        )

    return pd.DataFrame(history)




class AdditiveOnlyModel(nn.Module):

    def __init__(
        self,
        n_genes: int,
        out_dim: int,
        emb_dim: int = 64,
        effect_hidden: int = 512,
    ):
        super().__init__()
        self.gene_emb = nn.Embedding(n_genes, emb_dim)
        self.effect_decoder = nn.Sequential(
            nn.Linear(emb_dim, effect_hidden),
            nn.ReLU(),
            nn.Linear(effect_hidden, effect_hidden),
            nn.ReLU(),
            nn.Linear(effect_hidden, out_dim),
        )

    def get_effect(self, gene_id: torch.Tensor) -> torch.Tensor:
        return self.effect_decoder(self.gene_emb(gene_id))

    def forward(
        self,
        gene1_id: torch.Tensor,
        gene2_id: torch.Tensor,
        condition_type: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        effect_1 = self.get_effect(gene1_id)
        is_double = (condition_type == 1).float().unsqueeze(1)
        effect_2 = self.get_effect(gene2_id) * is_double
        return reference + effect_1 + effect_2


def summarize_best_epochs(history_df: pd.DataFrame) -> pd.DataFrame:
    """输出常用的最佳 epoch 汇总。"""
    rows = []
    for name, metric, mode in [
        ('best_by_double_Pearson',  'test_Pearson_double',  'max'),
        ('best_by_double_wPearson', 'test_wPearson_double', 'max'),
        ('best_by_double_MSE',      'test_MSE_double',      'min'),
    ]:
        idx = history_df[metric].idxmax() if mode == 'max' else history_df[metric].idxmin()
        row = history_df.loc[idx].to_dict()
        row['selection'] = name
        rows.append(row)
    return pd.DataFrame(rows)