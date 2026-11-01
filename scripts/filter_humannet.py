#!/usr/bin/env python3
"""
filter_humannet.py - Filter HumanNet edges by gene set

Filters a HumanNet file to keep only edges where at least one gene is in a
specified gene set derived from DisGeNET gene-disease associations.

Usage:
    # Filter for all genes in the DisGeNET associations
    python filter_humannet.py humannet.tsv all_disgenet_associations.tsv output.tsv

    # Filter for genes associated with specific diseases (UMLS CUIs)
    python filter_humannet.py humannet.tsv all_disgenet_associations.tsv output.tsv \
        --diseases C0006142 C0036341
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter HumanNet edges by gene set from DisGeNET associations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "humannet_file",
        type=Path,
        help="Input HumanNet file (gene1, gene2, LLS)"
    )
    parser.add_argument(
        "associations_file",
        type=Path,
        help="DisGeNET gene-disease associations (gene id in column 0, disease CUI in column 4)"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output filtered HumanNet file"
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        help="Disease CUIs to filter for (keep only genes associated with these diseases)"
    )
    parser.add_argument(
        "--both-genes",
        action="store_true",
        help="Require BOTH genes in edge to be in gene set (default: at least one)"
    )
    return parser.parse_args()


def load_gene_set_all(associations_file: Path) -> set:
    """Load all genes from the DisGeNET associations."""
    print(f"Loading all genes from {associations_file}...")

    df = pd.read_csv(associations_file, sep="\t")
    genes = set(df.iloc[:, 0].astype(str).str.strip())
    print(f"  ✓ Loaded {len(genes)} unique genes")

    return genes


def load_gene_set_diseases(associations_file: Path, disease_ids: list) -> set:
    """Load genes associated with specific diseases (UMLS CUIs)."""

    print(f"\nLoading DisGeNET associations from {associations_file}...")
    df = pd.read_csv(associations_file, sep="\t")

    disease_col = df.iloc[:, 4].astype(str).str.strip()
    gene_col = df.iloc[:, 0].astype(str).str.strip()

    target = set(disease_ids)
    filtered = gene_col[disease_col.isin(target)]

    genes = set(filtered)
    print(f"  ✓ Found {len(genes)} genes associated with {len(target)} selected diseases")

    return genes


def filter_humannet(humannet_file: Path, gene_set: set,
                   output_file: Path, both_genes: bool = False):
    """Filter HumanNet file to keep edges with genes in gene_set."""

    print(f"\nFiltering HumanNet file: {humannet_file}")
    print(f"  Gene set size: {len(gene_set)}")
    print(f"  Filter mode: {'BOTH genes' if both_genes else 'AT LEAST ONE gene'}")

    # Count total lines for progress bar
    with open(humannet_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    print(f"  Total edges in input: {total_lines:,}")

    kept = 0
    with open(humannet_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in tqdm(fin, total=total_lines, desc="  Filtering"):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            gene1 = parts[0].strip()
            gene2 = parts[1].strip()

            # Check filter condition
            if both_genes:
                # Both genes must be in set
                if gene1 in gene_set and gene2 in gene_set:
                    fout.write(line)
                    kept += 1
            else:
                # At least one gene must be in set
                if gene1 in gene_set or gene2 in gene_set:
                    fout.write(line)
                    kept += 1

    print(f"  ✓ Kept {kept:,} edges ({100*kept/total_lines:.2f}%)")
    print(f"  ✓ Output written to: {output_file}")


def main():
    args = parse_args()

    # Load gene set
    if args.diseases:
        gene_set = load_gene_set_diseases(
            args.associations_file,
            args.diseases
        )
    else:
        gene_set = load_gene_set_all(args.associations_file)

    # Filter HumanNet
    filter_humannet(
        args.humannet_file,
        gene_set,
        args.output_file,
        both_genes=args.both_genes
    )

    print("\n" + "="*70)
    print("✓ Filtering complete!")
    print("="*70)


if __name__ == "__main__":
    main()
