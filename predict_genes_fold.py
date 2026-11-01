"""
predict_genes_fold.py - Generate ranked gene predictions for diseases (OPTIMIZED)

This script loads trained DisGeneFormer models and generates ranked gene lists
for each disease (UMLS CUI) in the evaluation set. It runs inference on both the
best model and all 5 fold models.

Usage:
    python predict_genes_fold.py experiment_dir                  # Generate predictions and evaluate
    python predict_genes_fold.py experiment_dir --predict-only   # Only generate predictions
"""

import os
import os.path as osp
import ast
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from box import Box

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.DisGeneFormer import DisGeneFormer

# Suppress matplotlib warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ranked gene predictions for diseases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Directory containing config.yml and trained model"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only generate predictions without evaluation"
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_gene_id(x):
    """Normalize a gene identifier to a bare int (strip whitespace / version)."""
    return int(str(x).strip().split(".", 1)[0])


def get_disease_positive_genes(all_disgenet_associations, disease_id):
    """Ground-truth positive genes for a disease (UMLS CUI), from the DisGeNET
    gene-disease associations: disease CUI in column 4, gene id in column 0."""
    mask = all_disgenet_associations.iloc[:, 4].astype(str) == disease_id
    genes = all_disgenet_associations.loc[mask].iloc[:, 0].tolist()
    return sorted({_clean_gene_id(g) for g in genes})


def get_disease_test_set_pairs(all_genes, disease_id, positive_gene_ids):
    """Create test pairs (positives + negatives) for a disease (CUI)."""
    positive_set = set(positive_gene_ids)
    positives = pd.DataFrame({
        'gene_id': sorted(positive_set),
        'disease_id': disease_id,
        'y': 1.0
    })
    negatives = pd.DataFrame({
        'gene_id': sorted(set(all_genes) - positive_set),
        'disease_id': disease_id,
        'y': 0.0
    })
    return pd.concat([positives, negatives], ignore_index=True)


def format_test_set(test_set_pairs, gene_id_idx_mapping, disease_id_idx_mapping):
    """Format test pairs with index mappings."""
    test_set_pairs = test_set_pairs.copy()
    test_set_pairs['gene_idx'] = test_set_pairs['gene_id'].map(gene_id_idx_mapping).astype(float)
    test_set_pairs['disease_idx'] = test_set_pairs['disease_id'].map(disease_id_idx_mapping).astype(float)

    # Remove unmapped entries
    test_set_pairs = test_set_pairs.dropna(subset=["gene_idx", "disease_idx"]).copy()
    test_set_pairs["gene_idx"] = test_set_pairs["gene_idx"].astype(float)
    test_set_pairs["disease_idx"] = test_set_pairs["disease_idx"].astype(float)

    X_pairs = []
    y_labels = []

    for _, row in test_set_pairs.iterrows():
        X_pairs.append((float(row['gene_idx']), float(row['disease_idx'])))
        y_labels.append(float(row['y']))

    test_data_df = pd.DataFrame(
        [[X_pairs], [y_labels]],
        index=["X", "y"]
    )

    return test_data_df


def get_id_mappings(experiment_dir):
    """Load all ID mappings."""
    # Gene mappings
    gene_id_mapping_path = osp.join(experiment_dir, 'gene_net', 'processed', 'gene_id_index_mapping.tsv')
    df_gene = pd.read_csv(gene_id_mapping_path, sep='\t', header=None, skiprows=1)
    gene_id_to_idx = dict(zip(df_gene[0].astype(int), df_gene[1].astype(int)))
    gene_idx_to_id = dict(zip(df_gene[1].astype(int), df_gene[0].astype(int)))

    # Disease mappings (disease ids are UMLS CUIs)
    disease_id_mapping_path = osp.join(experiment_dir, 'disease_net', 'processed', 'disease_id_feature_index_mapping.tsv')
    df_disease = pd.read_csv(disease_id_mapping_path, sep='\t', header=None, skiprows=1)
    disease_id_to_idx = dict(zip(df_disease[0].astype(str), df_disease[1].astype(int)))
    disease_idx_to_id = dict(zip(df_disease[1].astype(int), df_disease[0].astype(str)))

    return gene_id_to_idx, gene_idx_to_id, disease_id_to_idx, disease_idx_to_id


def get_disease_name(disease_id):
    """Get disease name from UMLS CUI."""
    disease_id_name_mapping = {
        'C0006142': 'Breast Cancer',
        'C0009402': 'Colorectal Carcinoma',
        'C0023893': 'Liver Cirrhosis',
        'C0036341': 'Schizophrenia',
        'C0376358': 'Prostate Cancer',
        'C0001973': 'Chronic Alcoholic Intoxication',
        'C0011581': 'Depressive Disorder',
        'C0860207': 'Drug Induced Liver Disease',
        'C3714756': 'Intellectual Disability',
        'C0005586': 'Bipolar Disorder'
    }
    return disease_id_name_mapping.get(disease_id, 'Unknown_Disease')


def prepare_all_test_data(
    cfg,
    gene_id_to_idx,
    disease_id_to_idx,
    all_genes,
    test_sets,
    all_disgenet_associations,
    predict_only
):
    """Pre-create all test pairs (shared across all models)."""
    print("\n" + "="*70)
    print("Pre-creating test pairs (shared across all models)...")
    print("="*70)

    disease_test_data = {}

    for test_name in tqdm(test_sets, desc="Creating test pairs"):
        disease_id = str(test_name).strip()

        # Ground-truth positive genes for this disease (CUI)
        positive_gene_ids = get_disease_positive_genes(all_disgenet_associations, disease_id)
        if not positive_gene_ids:
            continue

        # Create test pairs (positives + negatives) for this disease
        test_set_pairs = get_disease_test_set_pairs(all_genes, disease_id, positive_gene_ids)
        test_set = format_test_set(test_set_pairs, gene_id_to_idx, disease_id_to_idx)

        if test_set.shape[1] == 1:
            X = test_set.at["X", test_set.columns[0]]
        else:
            X = ast.literal_eval(test_set.loc['X'])

        # Convert to numpy array once
        x_tensor = np.array(X, dtype=np.int64)

        positive_genes = None if predict_only else list(positive_gene_ids)

        disease_test_data[disease_id] = {
            'x_tensor': x_tensor,
            'positive_genes': positive_genes,
            'test_name': test_name
        }

    print(f"Created test pairs for {len(disease_test_data)} diseases")
    return disease_test_data


def run_inference_for_model(
    model_path,
    output_suffix,
    cfg,
    experiment_dir,
    gene_net_data,
    disease_net_data,
    gene_idx_to_id,
    disease_idx_to_id,
    disease_test_data,
    predict_only,
    device
):
    """Run inference for a single model using pre-created test pairs."""

    print(f"\n{'='*70}")
    print(f"Processing model: {os.path.basename(model_path)}")
    print('='*70)

    # Initialize model
    ablate_cfg = cfg.get('model', {}).get('ablate', {})
    ablate_dict = {k: bool(v) for k, v in ablate_cfg.items() if v}

    model = DisGeneFormer(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data.x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim,
        mode='DGP',
        ablate=ablate_dict
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Pre-compute embeddings for efficiency
    print("Computing embeddings...")
    with torch.no_grad():
        gene_emb_cache, dis_emb_cache = model.encode_nodes(gene_net_data, disease_net_data)
        gene_emb_cache = gene_emb_cache.cpu().numpy()
        dis_emb_cache = dis_emb_cache.cpu().numpy()
        mlp = model.final_fc.cpu()
        mlp.eval()

    def fast_predict(pair_idx_np, batch=2_000_000):
        """Vectorized prediction using cached embeddings."""
        out = np.empty((pair_idx_np.shape[0],), dtype=np.float32)

        for i in range(0, pair_idx_np.shape[0], batch):
            blk = pair_idx_np[i:i+batch]
            g = torch.from_numpy(gene_emb_cache[blk[:, 0]])
            d = torch.from_numpy(dis_emb_cache[blk[:, 1]])
            feats = torch.cat([g, d], dim=1)

            with torch.no_grad():
                logits = mlp(feats).float()
            out[i:i+batch] = logits[:, 1].numpy()

        return out

    # Process each disease using pre-created test pairs
    k_rows = []

    for disease_id, test_data in tqdm(disease_test_data.items(), desc=f"{output_suffix}"):
        x_tensor = test_data['x_tensor']
        positive_genes = test_data['positive_genes']
        test_name = test_data['test_name']

        # Run prediction
        prob_scores = fast_predict(x_tensor)

        # Build results DataFrame
        model_disease_pred_rows = []
        for (gene_idx, disease_idx), p in zip(x_tensor, prob_scores):
            model_disease_pred_rows.append({
                'gene_idx': int(gene_idx),
                'disease_idx': int(disease_idx),
                'model_prob': float(p),
            })

        # Rank genes
        model_disease_preds = pd.DataFrame(model_disease_pred_rows)
        model_disease_preds['gene_id'] = model_disease_preds['gene_idx'].map(gene_idx_to_id)
        model_disease_preds['disease_id'] = model_disease_preds['disease_idx'].map(disease_idx_to_id)

        ranked_genes = model_disease_preds.sort_values(by='model_prob', ascending=False)
        ranked_genes_unique = ranked_genes.groupby('gene_id').agg({'model_prob': 'mean'}).reset_index()
        ranked_genes_unique = ranked_genes_unique.sort_values(by='model_prob', ascending=False)

        # Save ranked genes
        ranked_genes_dir = osp.join(experiment_dir, f'ranked_genes_{output_suffix}')
        os.makedirs(ranked_genes_dir, exist_ok=True)
        ranked_genes_output_path = osp.join(
            ranked_genes_dir, f'{disease_id}_ranked_genes.tsv'
        )
        ranked_genes_unique.to_csv(ranked_genes_output_path, sep='\t', index=False)

        # Evaluate if not predict-only
        if not predict_only and positive_genes is not None:
            ranked_genes_list = ranked_genes_unique['gene_id'].astype(int).tolist()

            top_k_list = ast.literal_eval(cfg.evaluation.top_k)

            for top_k in top_k_list:
                top_k_genes = ranked_genes_list[:top_k]
                intersection = set(top_k_genes).intersection(set(positive_genes))

                tp = len(intersection)
                precision = tp / top_k if top_k > 0 else 0
                recall = tp / len(positive_genes) if positive_genes else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                k_rows.append({
                    "Disease": test_name,
                    "Disease_name": get_disease_name(test_name),
                    "K": top_k,
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1": round(f1, 4)
                })

    # Save results
    if not predict_only and k_rows:
        table = pd.DataFrame(k_rows)
        out_csv = osp.join(experiment_dir, f"precision_by_k_{output_suffix}.csv")
        table.to_csv(out_csv, index=False)
        print(f"Saved results to {out_csv}")
        return table

    return None


def compute_aggregate_results(experiment_dir, fold_results):
    """Compute mean and best results across folds."""

    if not fold_results:
        print("No fold results to aggregate")
        return

    print(f"\n{'='*70}")
    print("Computing aggregate statistics across folds...")
    print('='*70)

    # Combine all fold results
    all_folds = []
    for fold_num, df in fold_results.items():
        df = df.copy()
        df['Fold'] = fold_num
        all_folds.append(df)

    combined = pd.concat(all_folds, ignore_index=True)

    # Compute mean precision across folds for each (Disease, K) combination
    mean_results = combined.groupby(['Disease', 'K']).agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean'
    }).reset_index()

    mean_results['Precision'] = mean_results['Precision'].round(4)
    mean_results['Recall'] = mean_results['Recall'].round(4)
    mean_results['F1'] = mean_results['F1'].round(4)

    mean_csv = osp.join(experiment_dir, "precision_by_k_mean.csv")
    mean_results.to_csv(mean_csv, index=False)
    print(f"Saved mean results to {mean_csv}")

    # Compute best precision for each (Disease, K) combination
    best_results = []

    for (disease, k), group in combined.groupby(['Disease', 'K']):
        max_precision = group['Precision'].max()
        best_folds = group[group['Precision'] == max_precision]['Fold'].tolist()
        best_fold_str = ','.join(map(str, sorted(best_folds)))

        best_row = group[group['Precision'] == max_precision].iloc[0].copy()
        best_row['Fold'] = best_fold_str
        best_results.append(best_row)

    best_df = pd.DataFrame(best_results)[['Disease', 'K', 'Precision', 'Recall', 'F1', 'Fold']]
    best_df = best_df.sort_values(['Disease', 'K']).reset_index(drop=True)

    best_csv = osp.join(experiment_dir, "precision_by_k_best.csv")
    best_df.to_csv(best_csv, index=False)
    print(f"Saved best results to {best_csv}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print('='*70)
    print(f"Mean Precision across all folds: {mean_results['Precision'].mean():.4f}")
    print(f"Best Precision across all folds: {best_df['Precision'].max():.4f}")
    print(f"Mean F1 across all folds: {mean_results['F1'].mean():.4f}")
    print(f"Best F1 across all folds: {best_df['F1'].max():.4f}")


def main():
    args = parse_args()

    # Load config
    cfg_path = args.experiment_dir / "config.yml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")

    with cfg_path.open() as f:
        cfg = Box(yaml.safe_load(f))

    # Set seed
    seed = cfg.get('train', {}).get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load networks (once for all models)
    print("Loading GeneNet...")
    gene_dataset = GeneNet(cfg=cfg, experiment_dir=args.experiment_dir)
    gene_net_data = gene_dataset[0].to(device)

    print("Loading DiseaseNet...")
    disease_dataset = DiseaseNet(cfg=cfg, experiment_dir=args.experiment_dir)
    disease_net_data = disease_dataset[0].to(device)

    # Get mappings
    gene_id_to_idx, gene_idx_to_id, disease_id_to_idx, disease_idx_to_id = get_id_mappings(args.experiment_dir)
    all_genes = list(gene_id_to_idx.keys())

    # Load evaluation diseases (UMLS CUIs)
    eval_path = "data/eval_diseases.tsv"
    test_sets = (
        pd.read_csv(eval_path, sep="\t", header=None)
        .iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Load DisGeNET gene-disease associations (the ground truth)
    all_disgenet_associations = pd.read_csv(cfg.evaluation.all_disgenet_associations_path, sep='\t')

    # PRE-CREATE ALL TEST PAIRS (SHARED ACROSS ALL MODELS)
    disease_test_data = prepare_all_test_data(
        cfg=cfg,
        gene_id_to_idx=gene_id_to_idx,
        disease_id_to_idx=disease_id_to_idx,
        all_genes=all_genes,
        test_sets=test_sets,
        all_disgenet_associations=all_disgenet_associations,
        predict_only=args.predict_only
    )

    # Define model paths
    models_to_process = []

    # Best model
    best_model_path = osp.join(args.experiment_dir, 'best_model.ptm')
    if os.path.exists(best_model_path):
        models_to_process.append((best_model_path, 'best'))
    else:
        print(f"Warning: {best_model_path} not found, skipping best model")

    # Fold models
    fold_results = {}
    for fold in range(1, 6):
        fold_model_path = osp.join(args.experiment_dir, f'best_model_fold_{fold}.ptm')
        if os.path.exists(fold_model_path):
            models_to_process.append((fold_model_path, f'fold_{fold}'))
        else:
            print(f"Warning: {fold_model_path} not found, skipping fold {fold}")

    if not models_to_process:
        raise FileNotFoundError(f"No model files found in {args.experiment_dir}")

    # Run inference for each model
    for model_path, output_suffix in models_to_process:
        result_df = run_inference_for_model(
            model_path=model_path,
            output_suffix=output_suffix,
            cfg=cfg,
            experiment_dir=args.experiment_dir,
            gene_net_data=gene_net_data,
            disease_net_data=disease_net_data,
            gene_idx_to_id=gene_idx_to_id,
            disease_idx_to_id=disease_idx_to_id,
            disease_test_data=disease_test_data,
            predict_only=args.predict_only,
            device=device
        )

        # Store fold results for aggregation (exclude best model)
        if not args.predict_only and result_df is not None and output_suffix.startswith('fold_'):
            fold_num = int(output_suffix.split('_')[1])
            fold_results[fold_num] = result_df

    # Compute aggregate statistics across folds
    if not args.predict_only and fold_results:
        compute_aggregate_results(args.experiment_dir, fold_results)

    print("\n" + "="*70)
    print("All processing complete!")
    print("="*70)


if __name__ == '__main__':
    main()
