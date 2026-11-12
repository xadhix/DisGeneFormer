import click
from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.DisGeneFormer import DisGeneFormer
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import ast
import networkx as nx
from torch_geometric.utils import to_networkx
from datetime import datetime
import seaborn as sns
import argparse
import yaml
from box import Box
import gzip
import random
import pickle
import os
import os.path as osp
import numpy as np
import re
import shutil
from pathlib import Path

from torch.cuda.amp import autocast, GradScaler

def next_experiment_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base

parser = argparse.ArgumentParser()
parser.add_argument("exp_dir", type=Path, help="Directory that contains a valid config.yml file for training.")
args = parser.parse_args()
experiment_dir = next_experiment_dir(args.exp_dir)
# Used to speed up training- set to False if not supported 
use_autocast = True
print("Using experiment directory →", experiment_dir)

cfg_path = experiment_dir / "config.yml" 
if not cfg_path.is_file():
    parser.error(f"{cfg_path} not found — every experiment directory needs a config.yml")
with cfg_path.open() as f:
    cfg = Box(yaml.safe_load(f)) 

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch 
import torch.nn.functional as F

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_seed = cfg.get('train', {}).get('seed', 42)
set_seed(train_seed)

# For reproducibility if supported 
# torch.use_deterministic_algorithms(True, warn_only=True)  # hard fail on a non-det kernel
# torch.backends.cudnn.deterministic = True                  # pick deterministic impl
# torch.backends.cudnn.benchmark = False                     # disable autotune

# torch.backends.cuda.matmul.allow_tf32 = False              # TF32 off (Ampere/Hopper)
# torch.backends.cudnn.allow_tf32 = False

# Must turn TF32 off for deterministic behavior
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(train_seed)

import time
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split


def main(cfg):

    @torch.no_grad()
    def encode_graphs(model, gene_data, disease_data):
        model.eval()
        # forward once with a dummy pair just to get the embeddings
        g_emb, d_emb = model.encode_nodes(gene_data, disease_data)   # ⇦ add this helper
        return g_emb, d_emb

    print('Load the gene and disease graphs.')
    start_time = datetime.now()
    # Get arguments as lists
    gene_features_to_use = cfg.gene_net.features_to_use
    items = gene_features_to_use.strip('[]').split(',')
    gene_features_to_use = [item.strip() for item in items]

    disease_feature_source = cfg.disease_net.features_to_use
    items = disease_feature_source.strip('[]').split(',')
    disease_feature_source = [item.strip() for item in items]

    gene_dataset = GeneNet(cfg=cfg, experiment_dir=experiment_dir)

    ablate_cfg = cfg.get('model', {}).get('ablate', {})
    ablate_dict = {k: bool(v) for k, v in ablate_cfg.items() if v}

    # --- task / head / loss configuration (defaults preserve the original run) ---
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('train', {})
    task = model_cfg.get('task', 'classification')          # 'classification' | 'regression'
    loss_fn_name = train_cfg.get('loss_fn', 'focal_loss')   # focal_loss|cross_entropy|mse|smooth_l1
    optimizer_name = train_cfg.get('optimizer', 'Adam')     # Adam | AdamW
    grad_clip = float(train_cfg.get('grad_clip_max_norm', 0) or 0)  # 0 disables
    transformer_num_heads = int(model_cfg.get('transformer_num_heads', 4))
    transformer_num_layers = int(model_cfg.get('transformer_num_layers', 2))
    transformer_dropout = float(model_cfg.get('transformer_dropout', 0.1))
    # --- regression target reframing (defaults preserve the original run) ---
    # pos_weight>1 up-weights positive (score>0) pairs in the loss; target_transform
    # 'binary' fits every positive as 1.0 (Brier-style classification) instead of its
    # DisGeNET score. Both target the near-constant collapse on the 0-heavy score.
    pos_weight = float(train_cfg.get('pos_weight', 1.0))
    target_transform = str(train_cfg.get('target_transform', 'none')).lower()
    # --- sampled-softmax ranking objective (loss_fn: 'sampled_softmax') ---
    # Ranks each disease's positives against a large candidate set (K sampled
    # negatives, resampled every epoch), aligning TRAINING with the eval protocol
    # (recall@K ranks ALL genes). ranking_num_negatives: int K, or 'full' (all
    # non-positive genes). ranking_disease_batch: diseases scored per epoch (bounds
    # head-activation memory); 'all' = every covered disease. ranking_temperature:
    # softmax temperature on the raw head score.
    _rnn = train_cfg.get('ranking_num_negatives', 512)
    ranking_num_negatives = _rnn if (isinstance(_rnn, str) and _rnn == 'full') else int(_rnn)
    _rdb = train_cfg.get('ranking_disease_batch', 'all')
    ranking_disease_batch = _rdb if (isinstance(_rdb, str) and _rdb == 'all') else int(_rdb)
    ranking_temperature = float(train_cfg.get('ranking_temperature', 1.0))
    # loss_fn: 'combo' => smooth_l1(binary) + ranking_lambda * sampled_softmax, to keep a
    # calibrated score AND strong ranking. ranking_lambda weights the ranking term.
    ranking_lambda = float(train_cfg.get('ranking_lambda', 1.0))
    # Full disease coverage for the ranking loss: process ALL training diseases every epoch in
    # chunks of `ranking_disease_batch` with gradient accumulation (each chunk re-encodes the
    # graph), so coverage is independent of K / batch (default off = sub-sampled behaviour).
    ranking_full_coverage = bool(train_cfg.get('ranking_full_coverage', False))
    if task == 'regression':
        print(f"[task] regression | loss={loss_fn_name} | "
              f"target_transform={target_transform} | pos_weight={pos_weight} | "
              f"ranking_num_negatives={ranking_num_negatives} | "
              f"ranking_disease_batch={ranking_disease_batch} | tau={ranking_temperature}")


    def weight_based_edge_pruning(data, removal_prob=0.01, perturb_setting='random'):
        """
        Remove edges based on either a random probability or a threshold percentile.

        Args:
            data (torch_geometric.data.Data): PyG Data object containing edge_index and edge_attr.
            removal_prob (float): 
                - If setting='random', the probability each edge is removed.
                E.g. 0.01 => each edge has a 1% chance of being removed.
                - If setting='threshold', the fraction of edges to remove from the bottom
                based on normalized edge weights.
                E.g. 0.01 => remove the bottom 1% of edges by weight.
            setting (str): Either 'random' or 'threshold'.

        Returns:
            data (torch_geometric.data.Data): Pruned PyG Data object.
            stats (dict): Dictionary containing before/after statistics.
        """
        data = data.clone()
        
        # Extract & normalize edge weights
        edge_weights = data.edge_attr.squeeze()
        if edge_weights.max() == edge_weights.min():
            # All weights are identical; fallback to uniform removal or keep them all
            # For clarity, let's default to no removal in this edge case:
            normalized_weights = torch.zeros_like(edge_weights)
        else:
            normalized_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

        # Decide which edges to keep based on setting
        if perturb_setting == 'random':
            # Each edge has the same removal probability
            # Make a per-edge tensor of removal probability
            removal_probs = torch.full_like(normalized_weights, removal_prob)
            
            # Draw random numbers in [0, 1]; remove edge if random < removal_prob
            random_values = torch.rand_like(removal_probs)
            keep_mask = random_values > removal_probs

        elif perturb_setting == 'threshold':
            # Determine the threshold value for the bottom X% (removal_prob fraction)
            # For example, if removal_prob=0.01, remove edges with normalized_weights in the bottom 1%
            threshold_value = torch.quantile(normalized_weights, removal_prob)

            # Keep edges whose normalized weight is above this threshold
            keep_mask = normalized_weights > threshold_value

        else:
            raise ValueError(f"Invalid setting '{perturb_setting}'. Must be 'random' or 'threshold'.")

        # Apply the mask
        data.edge_index = data.edge_index[:, keep_mask]
        data.edge_attr = data.edge_attr[keep_mask]
        
        stats = {}
        stats['gene_edges_before'] = len(edge_weights)
        stats['gene_edges_removed'] = (~keep_mask).sum().item()
        stats['gene_edges_after'] = keep_mask.sum().item()
        stats['avg_weight_kept'] = edge_weights[keep_mask].mean().item() if keep_mask.any() else 0.0
        stats['avg_weight_removed'] = edge_weights[~keep_mask].mean().item() if (~keep_mask).any() else 0.0

        print(f"Original edges: {stats['gene_edges_before']}")
        print(f"Remaining edges: {stats['gene_edges_after']}")
        print(f"Removed edges: {stats['gene_edges_removed']}")
        print(f"Avg weight of kept edges: {stats['avg_weight_kept']:.4f}")
        print(f"Avg weight of removed edges: {stats['avg_weight_removed']:.4f}")

        return data, stats


    def weight_threshold_pruning(data, keep_top_percentage=0.2):
        """
        Keep only the top X% of edges by weight
        """
        # Create a copy
        data = data.clone()
        
        edge_weights = data.edge_attr.squeeze()
        
        # Find weight threshold for top X%
        threshold = torch.quantile(edge_weights, 1 - keep_top_percentage)
        
        # Keep only edges above threshold
        keep_mask = edge_weights >= threshold
        
        # Apply mask
        data.edge_index = data.edge_index[:, keep_mask]
        data.edge_attr = data.edge_attr[keep_mask]
        
        print(f"Kept {keep_mask.sum().item()} edges ({keep_top_percentage*100}% highest weight)")
        print(f"Average weight before: {edge_weights.mean():.4f}")
        print(f"Average weight after: {edge_weights[keep_mask].mean():.4f}")
        
        return data

    # Usage example:
    def perturb_network(geneNet, perturb_setting='random', removal_prob=0.01):
        """
        Apply weight-based edge pruning to GeneNet
        """

        gene_dataset_pruned = geneNet
        data = geneNet.data.clone()
        
        # Apply pruning
        pruned_data, stats = weight_based_edge_pruning(
            data,
            perturb_setting=perturb_setting,
            removal_prob=removal_prob
        )

        gene_dataset_pruned.data = pruned_data
        
        # return pruned_data
        return gene_dataset_pruned


    if cfg.gene_net.perturb:
        perturb_setting = cfg.gene_net.perturb_setting
        removal_prob=cfg.gene_net.perturb_prob
        pruned = perturb_network(gene_dataset, perturb_setting=perturb_setting, removal_prob=removal_prob)
        print(f"Pruning with setting: ={perturb_setting} and removal_prob={removal_prob}")
        gene_dataset = pruned

    disease_dataset = DiseaseNet(cfg=cfg, experiment_dir=experiment_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gene_net_data = gene_dataset[0]
    disease_net_data = disease_dataset[0]
    gene_net_data = gene_net_data.to(device)
    disease_net_data = disease_net_data.to(device)

    # Change to get new combined data with score column
    print('Generate training data.')

    disease_id_index_feature_mapping = disease_dataset.load_disease_index_feature_mapping()
    gene_id_index_feature_mapping = gene_dataset.load_node_index_mapping()

    all_genes = list(gene_id_index_feature_mapping.keys())
    all_diseases = list(disease_id_index_feature_mapping.keys())

    positives = pd.read_csv(cfg.data.train_positives_path, sep='\t', names=['EntrezGene ID', 'disease_id', 'y'])
    positives['EntrezGene ID'] = positives['EntrezGene ID'].astype(int)

    print(f"Number of positive associations in {cfg.data.train_positives_path}: {len(positives)}")

    positives = positives[
        positives["disease_id"].isin(all_diseases) & positives["EntrezGene ID"].isin(all_genes)
        ]
    covered_diseases = sorted(set(positives['disease_id']))
    covered_genes = sorted(set(positives['EntrezGene ID']))

    print(f"all genes: {len(covered_genes)}")
    print(f"all diseases: {len(covered_diseases)}")
    print(f"all positive associations: {len(positives)}")

    # Precompute each disease's positive gene NODE indices (for the sampled-softmax
    # ranking objective). disease_id -> np.int64 array of gene node indices.
    pos_gene_idx_by_disease = {}
    for _d_id, _sub in positives.groupby('disease_id')['EntrezGene ID']:
        _gidx = sorted({gene_id_index_feature_mapping[int(_g)] for _g in _sub})
        pos_gene_idx_by_disease[_d_id] = np.array(_gidx, dtype=np.int64)

    # Randomly generate negatives using diseases from positive data and genes from all genes
    def build_negatives_fast(positives_df, all_genes, rng=np.random.default_rng(train_seed)):
        """Return a DataFrame with exactly |positives_df| negatives."""
        # A hash-set gives O(1) look-ups instead of scanning a DataFrame
        pos_set = sorted(set(map(tuple, positives_df[['disease_id', 'EntrezGene ID']].to_numpy())))
        covered_diseases = sorted(positives_df['disease_id'].unique())

        # Precompute, for every disease, the genes that are *not* positives
        pos_by_disease = (
            positives_df
            .groupby('disease_id')['EntrezGene ID']
            .apply(set)
            .to_dict()
        )

        negatives = []
        for disease in covered_diseases:
            k = len(pos_by_disease[disease])          # keep the class balance 1 : 1
            candidates = np.setdiff1d(all_genes, list(pos_by_disease[disease]), assume_unique=True)
            candidates = np.sort(candidates)
            if k > len(candidates):                   
                raise ValueError(f'Not enough negatives for disease {disease}')
            sampled = rng.choice(candidates, size=k, replace=False)
            negatives.extend([(disease, g) for g in sampled])

        return pd.DataFrame(negatives, columns=['disease_id', 'EntrezGene ID'])
    
    # print(f"Random path for negatives: {cfg.data.train_negatives_path}")
    if cfg.data.train_negatives_path == 'random':
        print("Generating random negatives.")
        negatives = build_negatives_fast(positives, all_genes)
    else:
        negatives = pd.read_csv(cfg.data.train_negatives_path, sep='\t', names=['EntrezGene ID', 'disease_id'], dtype={'EntrezGene ID': pd.Int64Dtype()})

    def get_training_data_from_indexes(indexes, monogenetic_disease_only=False, multigenetic_diseases_only=False):
        # train_tuples = set()
        train_tuples = []
        for idx in indexes:
            pos = positives[positives['disease_id'] == covered_diseases[idx]]
            neg = negatives[negatives['disease_id'] == covered_diseases[idx]]
            if monogenetic_disease_only and len(pos) != 1:
                continue
            if multigenetic_diseases_only and len(pos) == 1:
                continue
            for index, row in pos.iterrows():
                if task == 'regression':
                    # Fit the DisGeNET score (column 'y'); fall back to 1.0 if the
                    # positives file carries no score column yet.
                    score = float(row['y']) if ('y' in row and pd.notna(row['y'])) else 1.0
                    if target_transform == 'binary':
                        # Reframe: every positive is 1.0 (classification), not its score.
                        score = 1.0
                else:
                    score = 1
                train_tuples.append((row['disease_id'], row['EntrezGene ID'], score))
            for index, row in neg.iterrows():
                # Negatives are fit as 0 ("no association") in both tasks.
                train_tuples.append((row['disease_id'], row['EntrezGene ID'], 0))

        train_tuples = sorted(train_tuples)

        n = len(train_tuples)
        x_out = np.ones((n, 2))  # will contain (gene_idx, disease_idx) tuples.

        # Regression needs float targets; classification keeps integer class labels.
        y_out = (torch.zeros((n,), dtype=torch.float32) if task == 'regression'
                 else torch.ones((n,), dtype=torch.long))

        for i, (disease_id, gene_id, s) in enumerate(train_tuples):
            x_out[i, 0] = gene_id_index_feature_mapping[int(gene_id)]
            x_out[i, 1] = disease_id_index_feature_mapping[disease_id]
            y_out[i] = s
        return x_out, y_out


    def train(
            max_epochs=cfg.train.max_epochs,
            early_stopping_window=cfg.train.early_stopping_window,
            info_each_epoch=cfg.train.info_each_epoch,
            folds=cfg.train.n_folds,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            fc_hidden_dim=cfg.model.fc_hidden_dim,
            gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
            disease_net_hidden_dim=cfg.model.disease_net_hidden_dim
    ):
        
        def focal_loss(logits, targets):
            ce = F.cross_entropy(logits, targets, reduction='none',
                                 weight=torch.tensor([0.2, 0.8]).to(device))
            return (0.25 * (1.0 - torch.exp(-ce)) ** 2.0 * ce).mean()  # alpha * (1-p_t)^gamma * CE

        def regression_loss(out, targets):
            # out: [B, 1] raw -> sigmoid into (0,1) to match the [0,1] score range.
            pred = torch.sigmoid(out.squeeze(-1))
            target = targets.float()
            reduction = 'none' if pos_weight != 1.0 else 'mean'
            if loss_fn_name in ('smooth_l1', 'huber'):
                per = F.smooth_l1_loss(pred, target, reduction=reduction)
            else:
                per = F.mse_loss(pred, target, reduction=reduction)
            if pos_weight == 1.0:
                return per
            # Up-weight positive (score>0) pairs so the loss is not dominated by the
            # 0-heavy target (the near-constant-collapse failure mode).
            w = torch.where(target > 0,
                            torch.as_tensor(pos_weight, device=per.device, dtype=per.dtype),
                            torch.ones((), device=per.device, dtype=per.dtype))
            return (per * w).sum() / w.sum()

        if task == 'regression':
            criterion = regression_loss
        elif loss_fn_name == 'cross_entropy':
            criterion = lambda logits, targets: F.cross_entropy(logits, targets)
        else:
            criterion = focal_loss

        # --- per-disease ranking objective (opt-in via train.loss_fn: 'ranking') ---
        # RankNet/BPR: within each disease, push positive (causal) gene scores above that
        # disease's negatives -- directly optimises the per-disease ranking that recall@K
        # measures ("1 causal + n random, rank them"). Uses raw scores (monotonic -> eval
        # AUC/recall unchanged). Default losses are untouched.
        use_ranking = (task == 'regression' and loss_fn_name == 'ranking')

        def group_pos_neg(x_arr, y_tensor):
            """Per-disease (pos_idx, neg_idx) tensors for diseases that have both."""
            g = x_arr[:, 1].astype(int)
            y = y_tensor.detach().cpu().numpy()
            pos, neg = {}, {}
            for i in range(len(g)):
                (pos if y[i] > 0 else neg).setdefault(int(g[i]), []).append(i)
            groups = []
            for d, pl in pos.items():
                nl = neg.get(d)
                if nl:
                    groups.append((torch.tensor(pl, device=device, dtype=torch.long),
                                   torch.tensor(nl, device=device, dtype=torch.long)))
            return groups

        def ranking_loss(out, groups):
            s = out.squeeze(-1)
            terms = []
            for pi, ni in groups:
                diff = s[pi].unsqueeze(1) - s[ni].unsqueeze(0)   # [P, N] pos-minus-neg
                terms.append(F.softplus(-diff).mean())           # BPR/RankNet
            if not terms:
                return s.sum() * 0.0
            return torch.stack(terms).mean()

        def phased_loss(out, targets, rank_groups):
            # Ranking objective for TRAINING only; falls back to `criterion` if a batch has
            # no rankable disease (keeps validation/selection identical to the other runs).
            if use_ranking and rank_groups:
                return ranking_loss(out, rank_groups)
            return criterion(out, targets)

        # --- sampled-softmax ranking objective (opt-in via train.loss_fn: 'sampled_softmax') ---
        # For each disease, rank its positives against K sampled negatives (resampled every
        # epoch) via a listwise softmax: -mean_p log( exp(s_p/tau) / sum_c exp(s_c/tau) ).
        # Aligns training with eval's recall@K (which ranks ALL genes). Scores come from one
        # forward() over a constructed candidate batch; disease sub-batching bounds memory.
        use_sampled_softmax = (task == 'regression' and loss_fn_name == 'sampled_softmax')
        use_combo = (task == 'regression' and loss_fn_name == 'combo')
        use_chunked = use_sampled_softmax and ranking_full_coverage
        n_genes = gene_net_data.x.shape[0]

        def pos_by_disease_for(disease_indexes):
            """disease_node_idx -> np.int64 array of positive gene node indices, for a fold."""
            out = {}
            for di in disease_indexes:
                d_id = covered_diseases[di]
                gidx = pos_gene_idx_by_disease.get(d_id)
                if gidx is not None and len(gidx):
                    out[disease_id_index_feature_mapping[d_id]] = gidx
            return out

        def sampled_softmax_loss(pos_by_d, k, dis_batch, rng, tau):
            items = list(pos_by_d.items())
            if dis_batch != 'all' and len(items) > dis_batch:
                sel = rng.choice(len(items), size=dis_batch, replace=False)
                items = [items[i] for i in sel]
            gene_rows, dis_rows, blocks, cursor = [], [], [], 0
            for d_idx, pos_idx in items:
                n_pos = len(pos_idx)
                if k == 'full':
                    mask = np.ones(n_genes, dtype=bool); mask[pos_idx] = False
                    neg_idx = np.nonzero(mask)[0]
                else:
                    kk = min(int(k), n_genes - n_pos)
                    cand = rng.choice(n_genes, size=min(n_genes, kk + n_pos + 32), replace=False)
                    neg_idx = cand[~np.isin(cand, pos_idx)][:kk]
                all_idx = np.concatenate([pos_idx, neg_idx])
                gene_rows.append(all_idx)
                dis_rows.append(np.full(len(all_idx), d_idx, dtype=np.int64))
                blocks.append((cursor, n_pos, len(all_idx)))
                cursor += len(all_idx)
            batch_idx = torch.from_numpy(
                np.stack([np.concatenate(gene_rows), np.concatenate(dis_rows)], axis=1)
            ).to(device).long()
            out = model(gene_net_data, disease_net_data, batch_idx)   # [M, out_dim]
            scores = out.squeeze(-1) / tau
            terms = []
            for (start, n_pos, n_total) in blocks:
                s = scores[start:start + n_total]
                logZ = torch.logsumexp(s, dim=0)
                terms.append(logZ - s[:n_pos].mean())   # -mean_p log softmax_p
            return torch.stack(terms).mean()

        print(f"Start Training")
        best_global_val = 1e80
        best_global_state = None
        fold = 0

        kf = KFold(n_splits=folds, shuffle=True, random_state=train_seed)
        for train_index, test_index in kf.split(covered_diseases):
            fold += 1
            best_val_loss = 1e80
            print(f'Generate training data for fold {fold}.')
            # Train on ALL pairs of the training-fold diseases; validate on the
            # held-out fold's diseases so model selection is DISEASE-DISJOINT
            # (proper CV). Previously the validation signal was a random 1%-of-pairs
            # split (`id_val`), which could place the same disease on both sides
            # (leakage for selection) while the disease-disjoint fold was computed
            # and discarded. Now the held-out fold IS the validation set.
            train_x, train_y = get_training_data_from_indexes(train_index)
            train_y = train_y.to(device)
            val_x, val_y = get_training_data_from_indexes(test_index)
            val_y = val_y.to(device)

            # Precompute per-disease pos/neg groupings for the ranking TRAINING objective.
            train_groups_rank = group_pos_neg(train_x, train_y) if use_ranking else None

            # Per-disease positive gene indices for the sampled-softmax objective
            # (train = training-fold diseases, val = held-out fold diseases).
            if use_sampled_softmax or use_combo:
                ss_pos_train = pos_by_disease_for(train_index)
                ss_pos_val = pos_by_disease_for(test_index)

            model = DisGeneFormer(
                gene_feature_dim=gene_net_data.x.shape[1],
                disease_feature_dim=disease_net_data.x.shape[1],
                fc_hidden_dim=fc_hidden_dim,
                gene_net_hidden_dim=gene_net_hidden_dim,
                disease_net_hidden_dim=disease_net_hidden_dim,
                mode='DGP',
                ablate=ablate_dict,
                task=task,
                transformer_num_heads=transformer_num_heads,
                transformer_num_layers=transformer_num_layers,
                transformer_dropout=transformer_dropout
            )

            model = model.to(device)

            scaler = torch.cuda.amp.GradScaler()

            optim_cls = torch.optim.AdamW if optimizer_name == 'AdamW' else torch.optim.Adam
            optimizer = optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f'Stat training fold {fold}/{folds}:')

            losses = dict()
            losses['train'] = list()
            losses['val'] = list()

            losses['mono'] = {
                'AUC': 0,
                'TPR': None,
                'FPR': None
            }
            losses['multi'] = {
                'AUC': 0,
                'TPR': None,
                'FPR': None
            }

            def _train_loss(epoch):
                if use_sampled_softmax:
                    return sampled_softmax_loss(
                        ss_pos_train, ranking_num_negatives, ranking_disease_batch,
                        np.random.default_rng(train_seed + epoch), ranking_temperature)
                out = model(gene_net_data, disease_net_data, train_x)
                base = phased_loss(out, train_y, train_groups_rank)
                if use_combo:
                    ss = sampled_softmax_loss(
                        ss_pos_train, ranking_num_negatives, ranking_disease_batch,
                        np.random.default_rng(train_seed + epoch), ranking_temperature)
                    return base + ranking_lambda * ss
                return base

            def _val_loss():
                if use_sampled_softmax:
                    return sampled_softmax_loss(
                        ss_pos_val, ranking_num_negatives, ranking_disease_batch,
                        np.random.default_rng(train_seed), ranking_temperature)
                val_logits = model(gene_net_data, disease_net_data, val_x)
                base = criterion(val_logits, val_y)
                if use_combo:
                    ss = sampled_softmax_loss(
                        ss_pos_val, ranking_num_negatives, ranking_disease_batch,
                        np.random.default_rng(train_seed), ranking_temperature)
                    return base + ranking_lambda * ss
                return base

            def _train_step_chunked(epoch):
                """Full-coverage ranking step: process ALL training diseases every epoch in chunks
                of `ranking_disease_batch`, with gradient accumulation (each chunk re-encodes via
                forward()). Peak memory = one chunk; coverage is independent of K/batch."""
                optimizer.zero_grad()
                items = list(ss_pos_train.items())
                rng = np.random.default_rng(train_seed + epoch)
                order = rng.permutation(len(items))
                chunk = int(ranking_disease_batch) if ranking_disease_batch != 'all' else len(items)
                chunk = max(1, chunk)
                idx_chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
                nchunks = max(1, len(idx_chunks))
                total = 0.0
                for cidx in idx_chunks:
                    sub = dict(items[int(j)] for j in cidx)
                    with autocast(enabled=use_autocast):
                        lc = sampled_softmax_loss(sub, ranking_num_negatives, 'all',
                                                  rng, ranking_temperature) / nchunks
                    if use_autocast:
                        scaler.scale(lc).backward()
                    else:
                        lc.backward()
                    total += lc.item()
                if grad_clip > 0:
                    if use_autocast:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_autocast:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                return total

            for epoch in range(max_epochs):
                # Train model.
                model.train()

                if use_chunked:
                    train_loss_val = _train_step_chunked(epoch)
                elif use_autocast:
                    optimizer.zero_grad()
                    with autocast():
                        loss = _train_loss(epoch)
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)   # clip on real (unscaled) grads
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss_val = loss.item()
                else:
                    optimizer.zero_grad()
                    loss = _train_loss(epoch)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    train_loss_val = loss.item()

                losses['train'].append(train_loss_val)

                # Validation (same objective on the disease-disjoint held-out fold; for
                # sampled_softmax/combo the negatives use a fixed seed so val is comparable
                # across epochs for early stopping / selection).
                model.eval()
                with torch.no_grad(), autocast(enabled=use_autocast):
                    val_loss = _val_loss()
                    
                current_val_loss = val_loss.item()
                losses['val'].append(current_val_loss)

                if epoch % info_each_epoch == 0:
                    print(
                        'Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}'.format(
                            epoch, losses['train'][epoch], losses['val'][epoch]
                        )
                    )
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    # Use to save per-fold model else only saves the best among the folds
                    # torch.save(model.state_dict(), osp.join(experiment_dir, f'best_model_fold_{fold}.ptm'), _use_new_zipfile_serialization=False)
                    best_fold_state = model.state_dict().copy()

                if current_val_loss < best_global_val:
                    best_global_val = current_val_loss
                    best_global_state = model.state_dict().copy()

                # Early stopping (epoch-level): stop THIS fold once the latest val
                # loss exceeds the best of the trailing `early_stopping_window`
                # epochs. Breaks the EPOCH loop (not the fold loop), so every CV
                # fold still runs and trains until it stops improving. (Previously
                # this check sat outside the epoch loop, so it never truncated
                # epochs and instead aborted the remaining folds.)
                if epoch > early_stopping_window:
                    last_window_losses = losses['val'][epoch - early_stopping_window:epoch]
                    if losses['val'][-1] > max(last_window_losses):
                        print(f'Early stopping fold {fold} at epoch {epoch}.')
                        break

            # Save the best model for this fold
            torch.save(best_fold_state, osp.join(experiment_dir, f'best_model_fold_{fold}.ptm'), _use_new_zipfile_serialization=False)
            print(f"Fold {fold} best val loss: {best_val_loss:.4f}")

        print(f"Training completed for fold {fold} ({time.time()})")
        torch.save(best_global_state, osp.join(experiment_dir, f'best_model.ptm'), _use_new_zipfile_serialization=False)
        print(f"\nOverall best val loss {best_global_val:.4f} saved to {experiment_dir}/best_model.ptm")

        return model


    disease_idx_to_id_mapping = dict()
    for disease_id, disease_idx in disease_id_index_feature_mapping.items():
        disease_idx_to_id_mapping[disease_idx] = disease_id

    gene_idx_entrez_id_mapping = dict()
    for entrez_id, gene_idx in gene_id_index_feature_mapping.items():
        gene_idx_entrez_id_mapping[gene_idx] = entrez_id

    model = train(
        max_epochs=cfg.train.max_epochs,
        early_stopping_window=cfg.train.early_stopping_window,
        folds=cfg.train.n_folds,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim
    )

    end_time = datetime.now()
    print(f"Training complete: {end_time - start_time}")


if __name__ == '__main__':
    main(cfg)