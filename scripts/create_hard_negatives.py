#!/usr/bin/env python3
"""
Generate hard negative gene-disease pairs for DisGeneFormer training.

This script:
1. Builds a raw hard negatives dataset from phenotype/pathway overlap between
   genes and diseases (cached to avoid recomputation).
2. Samples N pairs using a configurable strategy:
   - 'top'    : original approach — take the N hardest (highest overlap score)
   - 'random' : uniformly sample N from all candidates
   - 'bottom': take the N easiest (lowest overlap score, still sharing ≥1 term)
   - 'mixed'  : sample N from a broader range (e.g. top 5N) weighted by rank

Usage:
    python generate_hard_negatives.py \
        --n 21028 \
        --strategy random \
        --positive_labels data/training/train_positive.tsv \
        --seed 42
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  File paths
# ---------------------------------------------------------------------------
DISEASE_PATHWAY = "data/disease_net/raw/disease_pathway.tsv"
DISEASE_HPO     = "data/disease_net/raw/disease_hpo.tsv"
GENE_PATHWAY    = "data/gene_net/raw/gene_pathway_associations.tsv"
GENE_HPO        = "data/gene_net/raw/gene_hpo_disease.tsv"

RAW_OUTPUT       = "data/training/raw/all_hard_negatives.tsv"
TRAIN_OUTPUT     = "data/training/train_hard_negatives.tsv"
TEST_ASSOC       = "data/test/raw/all_disgenet_associations.tsv"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def load_tsv(path: str, col_names: list = None) -> pd.DataFrame:
    """Load a headerless TSV with explicit column names."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    # Try tab first, fall back to comma
    df = pd.read_csv(p, sep="\t", header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(p, sep=",", header=None)
    if col_names:
        if len(col_names) != df.shape[1]:
            print(f"  [WARN] {path}: expected {len(col_names)} cols but found {df.shape[1]}. "
                  f"Assigning names to first {min(len(col_names), df.shape[1])} cols.")
        for i, name in enumerate(col_names[:df.shape[1]]):
            df.rename(columns={i: name}, inplace=True)
    print(f"  Loaded {path}: {df.shape[0]:,} rows, columns = {list(df.columns)}")
    return df


def build_association_map(df: pd.DataFrame, key_col: str, val_col: str) -> dict:
    """Build a dict mapping key -> set of values."""
    assoc = defaultdict(set)
    for key, val in zip(df[key_col], df[val_col]):
        assoc[key].add(val)
    return dict(assoc)


# ---------------------------------------------------------------------------
#  Stage 1: Build raw hard negatives (all gene-disease pairs sharing ≥1 term)
# ---------------------------------------------------------------------------
def build_raw_hard_negatives(positive_pairs: set, test_pairs: set) -> pd.DataFrame:
    """
    Enumerate all (gene, disease) pairs sharing ≥1 HPO term or pathway,
    excluding known positive associations AND test associations.
    
    All input files are headerless TSVs:
      disease_pathway.tsv:            col0=disease_id, col1=pathway_id
      disease_hpo.tsv:                col0=disease_id, col1=hpo_id
      gene_pathway_associations.tsv:  col0=gene_id,    col1=pathway_id
      gene_hpo_disease.tsv:           col0=gene_id,    col1=hpo_id
    """
    print("\n=== Stage 1: Building raw hard negatives dataset ===\n")

    # --- Load the four association files (headerless) ---
    print("Loading association files...")
    df_disease_pathway = load_tsv(DISEASE_PATHWAY, ["disease_id", "pathway_id"])
    df_disease_hpo     = load_tsv(DISEASE_HPO,     ["disease_id", "hpo_id"])
    df_gene_pathway    = load_tsv(GENE_PATHWAY,    ["gene_id", "pathway_id"])
    df_gene_hpo        = load_tsv(GENE_HPO,        ["gene_id", "hpo_id"])

    # --- Build association maps ---
    print("\nBuilding association maps...")
    disease_to_pathways = build_association_map(df_disease_pathway, "disease_id", "pathway_id")
    disease_to_hpo      = build_association_map(df_disease_hpo, "disease_id", "hpo_id")
    gene_to_pathways    = build_association_map(df_gene_pathway, "gene_id", "pathway_id")
    gene_to_hpo         = build_association_map(df_gene_hpo, "gene_id", "hpo_id")

    print(f"  Diseases with pathways: {len(disease_to_pathways):,}")
    print(f"  Diseases with HPO:      {len(disease_to_hpo):,}")
    print(f"  Genes with pathways:    {len(gene_to_pathways):,}")
    print(f"  Genes with HPO:         {len(gene_to_hpo):,}")

    # --- Build inverted indices for efficient enumeration ---
    # Instead of checking all gene x disease pairs, we iterate through
    # shared terms to find overlapping pairs.
    print("\nBuilding inverted indices...")

    # HPO term -> set of genes that have it
    hpo_to_genes = defaultdict(set)
    for gene, hpos in gene_to_hpo.items():
        for h in hpos:
            hpo_to_genes[h].add(gene)

    # HPO term -> set of diseases that have it
    hpo_to_diseases = defaultdict(set)
    for disease, hpos in disease_to_hpo.items():
        for h in hpos:
            hpo_to_diseases[h].add(disease)

    # Pathway -> set of genes
    pathway_to_genes = defaultdict(set)
    for gene, pathways in gene_to_pathways.items():
        for p in pathways:
            pathway_to_genes[p].add(gene)

    # Pathway -> set of diseases
    pathway_to_diseases = defaultdict(set)
    for disease, pathways in disease_to_pathways.items():
        for p in pathways:
            pathway_to_diseases[p].add(disease)

    print(f"  Unique HPO terms:  {len(hpo_to_genes):,} (genes), {len(hpo_to_diseases):,} (diseases)")
    print(f"  Unique pathways:   {len(pathway_to_genes):,} (genes), {len(pathway_to_diseases):,} (diseases)")

    # --- Enumerate all (gene, disease) pairs with shared HPO terms ---
    print("\nEnumerating shared HPO pairs...")
    pair_shared_hpo = defaultdict(set)  # (gene, disease) -> set of shared HPO IDs
    shared_hpo_terms = set(hpo_to_genes.keys()) & set(hpo_to_diseases.keys())
    print(f"  HPO terms shared between genes and diseases: {len(shared_hpo_terms):,}")

    for i, hpo_term in enumerate(shared_hpo_terms):
        genes = hpo_to_genes[hpo_term]
        diseases = hpo_to_diseases[hpo_term]
        for g in genes:
            for d in diseases:
                pair_shared_hpo[(g, d)].add(hpo_term)
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1:,}/{len(shared_hpo_terms):,} HPO terms...")

    print(f"  Pairs with shared HPO: {len(pair_shared_hpo):,}")

    # --- Enumerate all (gene, disease) pairs with shared pathways ---
    print("\nEnumerating shared pathway pairs...")
    pair_shared_pathway = defaultdict(set)
    shared_pathways = set(pathway_to_genes.keys()) & set(pathway_to_diseases.keys())
    print(f"  Pathways shared between genes and diseases: {len(shared_pathways):,}")

    for i, pathway in enumerate(shared_pathways):
        genes = pathway_to_genes[pathway]
        diseases = pathway_to_diseases[pathway]
        for g in genes:
            for d in diseases:
                pair_shared_pathway[(g, d)].add(pathway)
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1:,}/{len(shared_pathways):,} pathways...")

    print(f"  Pairs with shared pathways: {len(pair_shared_pathway):,}")

    # --- Merge into final dataset ---
    print("\nMerging HPO and pathway overlaps...")
    all_pairs = set(pair_shared_hpo.keys()) | set(pair_shared_pathway.keys())
    print(f"  Total unique (gene, disease) pairs with any overlap: {len(all_pairs):,}")

    # Remove known positives AND test associations to prevent leakage
    n_before = len(all_pairs)
    all_pairs -= positive_pairs
    n_pos_removed = n_before - len(all_pairs)
    print(f"  Removed {n_pos_removed:,} known positive (training) pairs")

    n_before = len(all_pairs)
    all_pairs -= test_pairs
    n_test_removed = n_before - len(all_pairs)
    print(f"  Removed {n_test_removed:,} test association pairs")
    print(f"  Remaining candidate hard negatives: {len(all_pairs):,}")

    # Build rows
    print("\nAssembling output rows...")
    rows = []
    for i, (g, d) in enumerate(all_pairs):
        shared_hpo = pair_shared_hpo.get((g, d), set())
        shared_pw  = pair_shared_pathway.get((g, d), set())
        hpo_count = len(shared_hpo)
        pathway_count = len(shared_pw)
        score = hpo_count + pathway_count

        rows.append({
            "gene_id": g,
            "disease_id": d,
            "shared_hpo_ids": "|".join(str(x) for x in sorted(shared_hpo)) if shared_hpo else "",
            "shared_pathway_ids": "|".join(str(x) for x in sorted(shared_pw)) if shared_pw else "",
            "hpo_count": hpo_count,
            "pathway_count": pathway_count,
            "score": score,
        })

        if (i + 1) % 500_000 == 0:
            print(f"    Assembled {i+1:,}/{len(all_pairs):,} rows...")

    df_raw = pd.DataFrame(rows)
    df_raw.sort_values("score", ascending=False, inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    print(f"\n  Final raw hard negatives: {len(df_raw):,} pairs")
    print(f"  Score range: {df_raw['score'].min()} – {df_raw['score'].max()}")
    print(f"  Score median: {df_raw['score'].median():.1f}")
    print(f"  Score mean:   {df_raw['score'].mean():.1f}")

    return df_raw


# ---------------------------------------------------------------------------
#  Stage 2: Sample N negatives with a given strategy
# ---------------------------------------------------------------------------
def sample_negatives(df_raw: pd.DataFrame, n: int, strategy: str, seed: int,
                     mixed_pool_factor: int = 5) -> pd.DataFrame:
    """
    Sample N negative pairs from the raw hard negatives.

    Strategies:
        top    : Take the N rows with the highest score (original DGF approach).
        random : Uniformly sample N from all candidates.
        bottom : Take the N rows with the lowest score (easiest hard negatives).
        mixed  : Sample from top (mixed_pool_factor * N) rows, weighted by
                 inverse rank so harder negatives are still more likely but
                 the distribution is spread out.
    """
    print(f"\n=== Stage 2: Sampling {n:,} negatives (strategy='{strategy}', seed={seed}) ===\n")

    rng = np.random.default_rng(seed)
    total = len(df_raw)

    if n > total:
        print(f"  [WARN] Requested {n:,} but only {total:,} candidates available. Using all.")
        return df_raw.copy()

    if strategy == "top":
        # Original approach: hardest negatives
        sampled = df_raw.head(n).copy()
        print(f"  Selected top {n:,} (score range: {sampled['score'].iloc[-1]} – {sampled['score'].iloc[0]})")

    elif strategy == "random":
        # Uniform random sample across all candidates
        idx = rng.choice(total, size=n, replace=False)
        sampled = df_raw.iloc[sorted(idx)].copy()
        print(f"  Randomly sampled {n:,} from {total:,} candidates")
        print(f"  Sampled score range: {sampled['score'].min()} – {sampled['score'].max()}")
        print(f"  Sampled score mean:  {sampled['score'].mean():.1f}")

    elif strategy == "bottom":
        # Easiest hard negatives (lowest scores, but still share ≥1 term)
        sampled = df_raw.tail(n).copy()
        print(f"  Selected bottom {n:,} (score range: {sampled['score'].iloc[0]} – {sampled['score'].iloc[-1]})")

    elif strategy == "mixed":
        # Sample from a broader pool, weighted toward harder negatives
        pool_size = min(mixed_pool_factor * n, total)
        pool = df_raw.head(pool_size)

        # Weight by inverse rank (rank 1 = highest score gets highest weight)
        ranks = np.arange(1, pool_size + 1, dtype=np.float64)
        weights = 1.0 / ranks
        weights /= weights.sum()

        idx = rng.choice(pool_size, size=n, replace=False, p=weights)
        sampled = pool.iloc[sorted(idx)].copy()
        print(f"  Sampled {n:,} from top {pool_size:,} pool with inverse-rank weighting")
        print(f"  Sampled score range: {sampled['score'].min()} – {sampled['score'].max()}")
        print(f"  Sampled score mean:  {sampled['score'].mean():.1f}")

    else:
        sys.exit(f"[ERROR] Unknown strategy: '{strategy}'. Choose from: top, random, bottom, mixed")

    sampled.reset_index(drop=True, inplace=True)
    return sampled


# ---------------------------------------------------------------------------
#  Load positive labels to exclude
# ---------------------------------------------------------------------------
def load_pair_file(path: str, label: str) -> set:
    """
    Load a headerless TSV of (gene_id, disease_id) pairs.
    Returns a set of (gene, disease) tuples.
    """
    if not os.path.exists(path):
        print(f"  [WARN] {label} file not found: {path}")
        print(f"         No pairs will be excluded from this source.")
        return set()

    df = load_tsv(path, ["gene_id", "disease_id"])
    pairs = set(zip(df["gene_id"], df["disease_id"]))
    print(f"  Loaded {len(pairs):,} {label} pairs from {path}")
    return pairs


def load_disgenet_pairs(path: str, label: str) -> set:
    """
    Load (gene_id, disease CUI) pairs from a DisGeNET gene-disease associations
    file: it has a header row, the gene id in column 0 and the disease CUI in
    column 4. Returns a set of (gene, disease) tuples.
    """
    if not os.path.exists(path):
        print(f"  [WARN] {label} file not found: {path}")
        print(f"         No pairs will be excluded from this source.")
        return set()

    df = pd.read_csv(path, sep="\t")
    pairs = set(zip(df.iloc[:, 0], df.iloc[:, 4]))
    print(f"  Loaded {len(pairs):,} {label} pairs from {path}")
    return pairs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate hard negative training data for DisGeneFormer"
    )
    parser.add_argument("--n", type=int, default=21028,
                        help="Number of negative pairs to sample (default: 21028, matching positive count)")
    parser.add_argument("--strategy", type=str, default="random",
                        choices=["top", "random", "bottom", "mixed"],
                        help="Sampling strategy (default: random)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--positive_labels", type=str, default="data/training/train_positive.tsv",
                        help="Path to positive training labels (to exclude from negatives)")
    parser.add_argument("--test_associations", type=str, default=TEST_ASSOC,
                        help="Path to test/eval associations (to exclude — prevent leakage)")
    parser.add_argument("--raw_output", type=str, default=RAW_OUTPUT,
                        help="Path to cache the full raw hard negatives dataset")
    parser.add_argument("--train_output", type=str, default=TRAIN_OUTPUT,
                        help="Path to write the sampled training negatives")
    parser.add_argument("--mixed_pool_factor", type=int, default=5,
                        help="For 'mixed' strategy: sample from top (factor * N) candidates")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild of raw hard negatives even if cache exists")

    args = parser.parse_args()

    print("=" * 70)
    print("  DisGeneFormer — Hard Negatives Generator")
    print("=" * 70)
    print(f"\n  Strategy:        {args.strategy}")
    print(f"  N:               {args.n:,}")
    print(f"  Seed:            {args.seed}")
    print(f"  Positive labels: {args.positive_labels}")
    print(f"  Test assoc:      {args.test_associations}")
    print(f"  Raw cache:       {args.raw_output}")
    print(f"  Train output:    {args.train_output}")

    # Load positive pairs for exclusion
    print("\nLoading positive labels...")
    positive_pairs = load_pair_file(args.positive_labels, "positive training")

    # Load test associations for exclusion (prevent leakage)
    print("\nLoading test associations...")
    test_pairs = load_disgenet_pairs(args.test_associations, "test/eval")

    # Stage 1: Build or load raw hard negatives
    raw_path = Path(args.raw_output)

    if raw_path.exists() and not args.force_rebuild:
        print(f"\n[CACHED] Loading existing raw hard negatives from {args.raw_output}")
        t0 = time.time()
        df_raw = pd.read_csv(raw_path, sep="\t")
        print(f"  Loaded {len(df_raw):,} rows in {time.time() - t0:.1f}s")
        print(f"  Score range: {df_raw['score'].min()} – {df_raw['score'].max()}")
    else:
        t0 = time.time()
        df_raw = build_raw_hard_negatives(positive_pairs, test_pairs)

        # Ensure output directory exists
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting raw hard negatives to {args.raw_output}...")
        df_raw.to_csv(raw_path, sep="\t", index=False)
        print(f"  Done in {time.time() - t0:.1f}s")

    # Stage 2: Sample
    df_sampled = sample_negatives(
        df_raw, args.n, args.strategy, args.seed, args.mixed_pool_factor
    )

    # Write training negatives
    out_path = Path(args.train_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(df_sampled):,} training negatives to {args.train_output}...")
    df_sampled.to_csv(out_path, sep="\t", index=False)

    # Summary statistics
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total raw candidates:    {len(df_raw):,}")
    print(f"  Sampled for training:    {len(df_sampled):,}")
    print(f"  Strategy:                {args.strategy}")
    print(f"  Unique genes in sample:  {df_sampled['gene_id'].nunique():,}")
    print(f"  Unique diseases in sample: {df_sampled['disease_id'].nunique():,}")
    print(f"  Score distribution of sample:")
    print(f"    min:    {df_sampled['score'].min()}")
    print(f"    25th:   {df_sampled['score'].quantile(0.25):.0f}")
    print(f"    median: {df_sampled['score'].median():.0f}")
    print(f"    75th:   {df_sampled['score'].quantile(0.75):.0f}")
    print(f"    max:    {df_sampled['score'].max()}")
    print(f"    mean:   {df_sampled['score'].mean():.1f}")
    print(f"\n  Output: {args.train_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()