import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

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
    parser = argparse.ArgumentParser(description="Evaluate ranked gene lists")
    parser.add_argument("experiment_dir", type=Path,
                       help="Directory containing 'ranked_genes/' subfolder")
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


def evaluate_ranked_dir(args):
    ranked_dir = args.experiment_dir / "ranked_genes"
    if not ranked_dir.is_dir():
        raise SystemExit(f"Expected 'ranked_genes/' subfolder in {args.experiment_dir}")

    # Setup paths
    data_root = args.data_root
    all_disgenet_associations_path = data_root / "test/raw/all_disgenet_associations.tsv"
    eval_diseases_path = data_root / "eval_diseases.tsv"

    # Load evaluation diseases (UMLS CUIs)
    eval_diseases = pd.read_csv(eval_diseases_path, sep="\t", header=None)[0].astype(str).tolist()

    # Load DisGeNET gene-disease associations (the ground truth)
    disgenet_all = pd.read_csv(all_disgenet_associations_path, sep="\t")

    # Output containers
    metrics_rows = []
    tp_rank_rows = []

    k_start, k_stop, k_step = 5, args.max_k, 1

    for tsv in sorted(ranked_dir.glob("*.tsv")):
        m = re.match(r"([^_]+).*ranked_genes\.tsv$", tsv.name)
        if not m:
            print(f"  ✘︎ skip weird file name {tsv.name}")
            continue
        dis_id = m.group(1)
        if dis_id not in eval_diseases:
            print(f"  ✘︎ {dis_id} not in eval set; skipping")
            continue
        dis_name = disease_id_name_mapping.get(dis_id, "Unknown_Disease")

        # Read the gene_id column by name (not by position)
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
            if 'gene_id' not in df.columns:
                print(f"  ✘︎ No 'gene_id' column found in {tsv.name}, skipping")
                continue
            # Drop NaN values and clean IDs
            ranked_genes = df['gene_id'].dropna().map(clean_id).tolist()
        except Exception as e:
            print(f"  ✘︎ Error reading {tsv.name}: {e}, skipping")
            continue

        if not ranked_genes:
            print(f"  ✘︎ No valid gene IDs found in {tsv.name}, skipping")
            continue

        disease_pos = disease_positive_genes(disgenet_all, dis_id)
        if not disease_pos:
            print(f"  ✘︎ No DisGeNET associations for {dis_id}; skipping")
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

    # Save results
    pd.DataFrame(metrics_rows).to_csv(args.experiment_dir / "top_k_eval_metrics.csv", index=False)
    print(f"✔︎  top_k_eval_metrics.csv written to {args.experiment_dir}")

    if args.include_tp_ranks:
        pd.DataFrame(tp_rank_rows).to_csv(args.experiment_dir / "true_gene_ranks.csv", index=False)
        print(f"✔︎  true_gene_ranks.csv written to {args.experiment_dir}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_ranked_dir(args)
