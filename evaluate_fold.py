import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Disease ID (UMLS CUI) to name mapping
disease_id_name_mapping = {
    'C0006142': 'Malignant_Neoplasm_Of_breast',
    'C0009402': 'Colorectal_Carcinoma',
    'C0023893': 'Liver_Cirrhosis_Experimental',
    'C0036341': 'Schizophrenia',
    'C0376358': 'Malignant_Neoplasm_Of_Prostate',
    'C0001973': 'Alcoholic_Intoxication_Chronic',
    'C0011581': 'Depressive_Disorder',
    'C0860207': 'Drug_Induced_Liver_Disease',
    'C3714756': 'Intellectual_Disability',
    'C0005586': 'Bipolar_Disorder'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ranked gene lists for all models")
    parser.add_argument("experiment_dir", type=Path,
                       help="Directory containing 'ranked_genes_*/' subfolders")
    parser.add_argument("--max-k", type=int, default=250,
                       help="Maximum K value for evaluation (default: 250)")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                       help="Root directory for data files (default: data)")
    parser.add_argument("--include-tp-ranks", action="store_true", default=True,
                       help="Generate tp_ranks.csv output file")
    return parser.parse_args()


def clean_id(x) -> str:
    """Make every gene identifier comparable (strip whitespace, cut version)."""
    s = str(x).strip()
    return s.split(".", 1)[0] if "." in s else s


def disease_positive_genes(disgenet_all: pd.DataFrame, disease_id: str) -> list:
    """Ground-truth positive genes for a disease (UMLS CUI), from the DisGeNET
    gene-disease associations: disease CUI in column 4, gene id in column 0."""
    mask = disgenet_all.iloc[:, 4].astype(str) == disease_id
    return disgenet_all.loc[mask].iloc[:, 0].map(clean_id).unique().tolist()


def evaluate_single_ranked_dir(ranked_dir, output_suffix, args, data_root,
                               eval_diseases, disgenet_all):
    """Evaluate a single ranked_genes directory and return results."""

    print(f"\n{'='*70}")
    print(f"Evaluating: {ranked_dir.name}")
    print('='*70)

    if not ranked_dir.is_dir():
        print(f"  ✘︎ Directory not found: {ranked_dir}")
        return None, None

    # Output containers
    metrics_rows = []
    tp_rank_rows = []

    k_start, k_stop, k_step = 5, args.max_k, 1

    for tsv in tqdm(sorted(ranked_dir.glob("*.tsv")), desc=f"  {output_suffix}"):
        m = re.match(r"([^_]+).*ranked_genes\.tsv$", tsv.name)
        if not m:
            continue
        dis_id = m.group(1)
        if dis_id not in eval_diseases:
            continue
        dis_name = disease_id_name_mapping.get(dis_id, "Unknown_Disease")

        ranked_genes = (pd.read_csv(tsv, sep="\t", usecols=[0], dtype=str)
                         .iloc[:, 0].map(clean_id).tolist())

        disease_pos = disease_positive_genes(disgenet_all, dis_id)
        if not disease_pos:
            continue

        max_k = min(len(ranked_genes), args.max_k)
        for k in range(k_start, max_k + 1, k_step):
            top_k = ranked_genes[:k]

            disease_tp = sum(g in disease_pos for g in top_k)
            prec = disease_tp / k
            rec = disease_tp / len(disease_pos) if disease_pos else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0

            metrics_rows.append({
                "experiment_path": str(args.experiment_dir),
                "disease_id": dis_id,
                "disease_name": dis_name,
                "K": k,
                "disease_tp": disease_tp,
                "disease_prec": prec,
                "disease_recall": rec,
                "disease_f1": f1,
            })

        # TP ranks (ALL ranks, not limited to MAX_K)
        if args.include_tp_ranks:
            tp_ranks = [i + 1 for i, g in enumerate(ranked_genes) if g in disease_pos]
            tp_rank_rows.append({
                "disease_id": dis_id,
                "disease_name": dis_name,
                "tp_ranks": str(tp_ranks)
            })

    # Convert to DataFrames
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else None
    tp_ranks_df = pd.DataFrame(tp_rank_rows) if tp_rank_rows else None

    # Save results
    if metrics_df is not None:
        out_csv = args.experiment_dir / f"top_k_eval_metrics_{output_suffix}.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"  ✔︎  Saved metrics to {out_csv.name}")

    if tp_ranks_df is not None and args.include_tp_ranks:
        out_csv = args.experiment_dir / f"true_gene_ranks_{output_suffix}.csv"
        tp_ranks_df.to_csv(out_csv, index=False)
        print(f"  ✔︎  Saved TP ranks to {out_csv.name}")

    return metrics_df, tp_ranks_df


def compute_aggregate_results(experiment_dir, fold_results):
    """Compute mean and best results across folds."""

    if not fold_results:
        print("\nNo fold results to aggregate")
        return

    print(f"\n{'='*70}")
    print("Computing aggregate statistics across folds...")
    print('='*70)

    # Combine all fold results
    all_folds = []
    for fold_num, df in fold_results.items():
        if df is None:
            continue
        df = df.copy()
        df['Fold'] = fold_num
        all_folds.append(df)

    if not all_folds:
        print("No valid fold results to aggregate")
        return

    combined = pd.concat(all_folds, ignore_index=True)

    # Compute mean metrics across folds for each (disease_id, K) combination
    group_cols = ['disease_id', 'disease_name', 'K']
    agg_cols = {
        'disease_tp': 'mean',
        'disease_prec': 'mean',
        'disease_recall': 'mean',
        'disease_f1': 'mean'
    }

    mean_results = combined.groupby(group_cols).agg(agg_cols).reset_index()

    # Round all metric columns
    for col in mean_results.columns:
        if col not in group_cols:
            mean_results[col] = mean_results[col].round(4)

    mean_csv = experiment_dir / "top_k_eval_metrics_mean.csv"
    mean_results.to_csv(mean_csv, index=False)
    print(f"✔︎  Saved mean results to {mean_csv.name}")

    # Compute best precision for each (disease_id, K) combination
    best_results = []

    for (disease_id, disease_name, k), group in combined.groupby(group_cols):
        max_precision = group['disease_prec'].max()
        best_folds = group[group['disease_prec'] == max_precision]['Fold'].tolist()
        best_fold_str = ','.join(map(str, sorted(best_folds)))

        best_row = group[group['disease_prec'] == max_precision].iloc[0].to_dict()
        best_row['Fold'] = best_fold_str
        best_results.append(best_row)

    best_df = pd.DataFrame(best_results)

    # Reorder columns to put Fold at the end
    cols = [c for c in best_df.columns if c != 'Fold'] + ['Fold']
    best_df = best_df[cols]
    best_df = best_df.sort_values(['disease_id', 'K']).reset_index(drop=True)

    best_csv = experiment_dir / "top_k_eval_metrics_best.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"✔︎  Saved best results to {best_csv.name}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print('='*70)
    print(f"Mean Precision across all folds: {mean_results['disease_prec'].mean():.4f}")
    print(f"Best Precision across all folds: {best_df['disease_prec'].max():.4f}")
    print(f"Mean F1 across all folds: {mean_results['disease_f1'].mean():.4f}")
    print(f"Best F1 across all folds: {best_df['disease_f1'].max():.4f}")


def main():
    args = parse_args()

    # Load evaluation diseases (UMLS CUIs)
    data_root = args.data_root
    eval_diseases_path = data_root / "eval_diseases.tsv"
    eval_diseases = pd.read_csv(eval_diseases_path, sep="\t", header=None)[0].astype(str).tolist()

    # Load DisGeNET gene-disease associations (the ground truth)
    all_disgenet_path = data_root / "test/raw/all_disgenet_associations.tsv"
    disgenet_all = pd.read_csv(all_disgenet_path, sep="\t")

    # Find all ranked_genes directories
    ranked_dirs = []

    # Best model
    best_dir = args.experiment_dir / "ranked_genes_best"
    if best_dir.is_dir():
        ranked_dirs.append((best_dir, 'best'))
    else:
        # Fallback to old naming convention
        old_best_dir = args.experiment_dir / "ranked_genes"
        if old_best_dir.is_dir():
            ranked_dirs.append((old_best_dir, 'best'))

    # Fold models
    fold_results = {}
    for fold in range(1, 6):
        fold_dir = args.experiment_dir / f"ranked_genes_fold_{fold}"
        if fold_dir.is_dir():
            ranked_dirs.append((fold_dir, f'fold_{fold}'))

    if not ranked_dirs:
        raise SystemExit(f"No 'ranked_genes_*' directories found in {args.experiment_dir}")

    print(f"Found {len(ranked_dirs)} ranked_genes directories to evaluate")

    # Evaluate each directory
    for ranked_dir, output_suffix in ranked_dirs:
        metrics_df, tp_ranks_df = evaluate_single_ranked_dir(
            ranked_dir=ranked_dir,
            output_suffix=output_suffix,
            args=args,
            data_root=data_root,
            eval_diseases=eval_diseases,
            disgenet_all=disgenet_all
        )

        # Store fold results for aggregation (exclude best model)
        if metrics_df is not None and output_suffix.startswith('fold_'):
            fold_num = int(output_suffix.split('_')[1])
            fold_results[fold_num] = metrics_df

    # Compute aggregate statistics across folds
    if fold_results:
        compute_aggregate_results(args.experiment_dir, fold_results)

    print("\n" + "="*70)
    print("All evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    args = parse_args()
    main()
