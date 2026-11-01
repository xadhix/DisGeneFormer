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
                train_tuples.append((row['disease_id'], row['EntrezGene ID'], 1))
            for index, row in neg.iterrows():
                train_tuples.append((row['disease_id'], row['EntrezGene ID'], 0))

        train_tuples = sorted(train_tuples)

        n = len(train_tuples)
        x_out = np.ones((n, 2))  # will contain (gene_idx, disease_idx) tuples.
        
        # Use float32 if using BCEWithLogits else use long for custom focal loss (CE)
        y_out = torch.ones((n,), dtype=torch.long)

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
        
        criterion = lambda logits, targets: (
                0.25 *  # alpha
                (1.0 - torch.exp(-F.cross_entropy(logits, targets, reduction='none', weight=torch.tensor([0.2, 0.8]).to(device))))**2.0 *  # (1 - p_t)^gamma
                F.cross_entropy(logits, targets, reduction='none', weight=torch.tensor([0.2, 0.8]).to(device))
                ).mean()
        
        
        print(f"Start Training")
        best_global_val = 1e80
        best_global_state = None
        fold = 0

        kf = KFold(n_splits=folds, shuffle=True, random_state=train_seed)
        for train_index, test_index in kf.split(covered_diseases):
            fold += 1
            best_val_loss = 1e80
            print(f'Generate training data for fold {fold}.')
            all_train_x, all_train_y = get_training_data_from_indexes(train_index)

            # Split into train and validation set.
            id_tr, id_val = train_test_split(range(len(all_train_x)), test_size=0.01, random_state=train_seed)
            train_x = all_train_x[id_tr]
            train_y = all_train_y[id_tr].to(device)
            val_x = all_train_x[id_val]
            val_y = all_train_y[id_val].to(device)

            # Generate the test data for mono and multigenetic diseases.
            all_indexes = list(range(len(covered_diseases)))
            print(f'Generate test data for fold {fold}.')
            test_x = dict()
            test_y = dict()
            test_x['mono'], test_y['mono'] = get_training_data_from_indexes(test_index, monogenetic_disease_only=True)
            test_y['mono'] = test_y['mono'].to(device)
            test_x['multi'], test_y['multi'] = get_training_data_from_indexes(test_index, multigenetic_diseases_only=True)
            test_y['multi'] = test_y['multi'].to(device)

            # Save the test data to the relevant directory
            test_data = {
                'mono': {
                    'x': test_x['mono'].tolist(),  # Convert numpy array to list
                    'y': test_y['mono'].cpu().tolist()  # Move tensor to CPU and convert to list
                },
                'multi': {
                    'x': test_x['multi'].tolist(),  # Convert numpy array to list
                    'y': test_y['multi'].cpu().tolist()  # Move tensor to CPU and convert to list
                }
            }

            model = DisGeneFormer(
                gene_feature_dim=gene_net_data.x.shape[1],
                disease_feature_dim=disease_net_data.x.shape[1],
                fc_hidden_dim=fc_hidden_dim,
                gene_net_hidden_dim=gene_net_hidden_dim,
                disease_net_hidden_dim=disease_net_hidden_dim,
                mode='DGP',
                ablate=ablate_dict
            )

            model = model.to(device)

            scaler = torch.cuda.amp.GradScaler()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

            for epoch in range(max_epochs):
                # Train model.
                # if epoch % refresh_every == 0 and epoch > 0:
                #     gene_cache, disease_cache = encode_graphs(model, gene_net_data, disease_net_data)
                model.train()
                optimizer.zero_grad()

                # Use autocast to speed up 
                if use_autocast:
                    with autocast():
                        out = model(gene_net_data, disease_net_data, train_x)
                        loss = criterion(out, train_y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    out = model(gene_net_data, disease_net_data, train_x)
                    loss = criterion(out, train_y)
                    loss.backward()
                    optimizer.step()
                    
                losses['train'].append(loss.item())

                # Validation.
                model.eval()
                with torch.no_grad(), autocast(enabled=use_autocast):
                    val_logits = model(gene_net_data, disease_net_data, val_x)
                    val_loss = criterion(val_logits, val_y)
                    
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

            # Save the best model for this fold
            torch.save(best_fold_state, osp.join(experiment_dir, f'best_model_fold_{fold}.ptm'), _use_new_zipfile_serialization=False)
            print(f"Fold {fold} best val loss: {best_val_loss:.4f}")

            # Early stopping
            if epoch > early_stopping_window:
                last_window_losses = losses['val'][epoch - early_stopping_window:epoch]
                if losses['val'][-1] > max(last_window_losses):
                    print('Early Stopping!')
                    break

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