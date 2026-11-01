"""
plot_identity_scatter.py - Compare performance between two training approaches

Creates identity scatter plot comparing precision at a given K value between
two methods (e.g., random negatives vs hard negatives).

Usage:
    python plot_identity_scatter.py results_root method1 method2
    python plot_identity_scatter.py results_root method1 method2 --k-value 20
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Disease ID to name mapping
DISEASE_NAMES = {
    'C0006142': 'Breast cancer',
    'C0009402': 'Colorectal carcinoma',
    'C0023893': 'Liver cirrhosis',
    'C0036341': 'Schizophrenia',
    'C0376358': 'Prostate cancer',
    'C0001973': 'Chronic alcoholic intoxication',
    'C0011581': 'Depressive disorder',
    'C0860207': 'Drug-induced liver disease',
    'C3714756': 'Intellectual disability',
    'C0005586': 'Bipolar disorder'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create identity scatter plot comparing two methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_root",
        type=Path,
        help="Root directory containing method subdirectories"
    )
    parser.add_argument(
        "method_x",
        type=str,
        help="Method name for X-axis (directory name)"
    )
    parser.add_argument(
        "method_y",
        type=str,
        help="Method name for Y-axis (directory name)"
    )
    parser.add_argument(
        "--metric-file",
        type=str,
        default="top_k_eval_metrics_mean.csv",
        help="Name of metrics CSV file in each method directory"
    )
    parser.add_argument(
        "--k-value",
        type=int,
        default=50,
        help="K value to plot (default: 50)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="disease_prec",
        choices=["disease_prec", "disease_recall", "disease_f1", "disease_tp"],
        help="Metric to compare"
    )
    parser.add_argument(
        "--exclude-diseases",
        nargs="+",
        help="Disease IDs to exclude from plot"
    )
    parser.add_argument(
        "--method-names",
        type=Path,
        help="JSON file mapping directory names to display names"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/results"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom output filename (without extension)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8, 8],
        help="Figure size (width height)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure"
    )
    return parser.parse_args()


def load_method_names(json_path: Optional[Path]) -> Dict[str, str]:
    """Load custom method name mappings from JSON file."""
    if json_path is None or not json_path.exists():
        return {}
    
    import json
    with open(json_path) as f:
        mapping = json.load(f)
        print(f"✓ Loaded method name mappings from {json_path}")
        return mapping


def find_metric_file(method_dir: Path, primary_file: str) -> Optional[Path]:
    """Find metrics file with fallback to top_k_eval_metrics.csv."""
    primary_path = method_dir / primary_file
    if primary_path.exists():
        return primary_path
    
    fallback_file = "top_k_eval_metrics.csv"
    if primary_file != fallback_file:
        fallback_path = method_dir / fallback_file
        if fallback_path.exists():
            print(f"  → Using fallback {fallback_file} for {method_dir.name}")
            return fallback_path
    
    return None


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load method name mappings
    method_names = load_method_names(args.method_names)
    
    # Get display names
    method_x_display = method_names.get(args.method_x, args.method_x)
    method_y_display = method_names.get(args.method_y, args.method_y)
    
    print(f"\\nLoading data...")
    print(f"  X-axis: {method_x_display} ({args.method_x})")
    print(f"  Y-axis: {method_y_display} ({args.method_y})")
    
    # Load data for both methods
    all_data = []
    for method_name in [args.method_x, args.method_y]:
        method_dir = args.results_root / method_name
        
        if not method_dir.is_dir():
            raise ValueError(f"Method directory not found: {method_dir}")
        
        metric_path = find_metric_file(method_dir, args.metric_file)
        if metric_path is None:
            raise ValueError(f"No metrics file found for {method_name}")
        
        # Load metrics with error handling
        try:
            df = pd.read_csv(metric_path)
            
            if df.empty:
                raise ValueError(f"Empty file {metric_path}")
                
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty or invalid file {metric_path}")
        except Exception as e:
            raise ValueError(f"Error loading {metric_path}: {e}")
        
        df['method'] = method_name
        all_data.append(df)
        print(f"  ✓ Loaded: {method_name}")
    
    # Combine data
    results_df = pd.concat(all_data, ignore_index=True)
    
    # Filter for target K value and both methods
    mask = (results_df["K"] == args.k_value) & results_df["method"].isin([args.method_x, args.method_y])
    tbl = results_df.loc[mask]
    
    if tbl.empty:
        raise ValueError(f"No data found for K={args.k_value} in both methods")
    
    # Pivot so each disease has two metric values
    pivot = (
        tbl.pivot_table(
            index="disease_id",
            columns="method",
            values=args.metric,
            aggfunc="mean",
        )
        .loc[:, [args.method_x, args.method_y]]
        .dropna()
    )
    
    if pivot.empty:
        raise ValueError("No overlapping diseases found between methods")
    
    # Exclude specified diseases
    if args.exclude_diseases:
        pivot = pivot.drop(index=args.exclude_diseases, errors="ignore")
        print(f"  Excluded {len(args.exclude_diseases)} disease(s)")
    
    # Build point labels
    pivot["label"] = pivot.index.map(
        lambda cui: f"{DISEASE_NAMES.get(cui, 'Unknown')} ({cui})"
    )
    
    print(f"  Plotting {len(pivot)} diseases")
    
    # Create plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    
    # Plot points
    ax.scatter(pivot[args.method_x], pivot[args.method_y], s=100, alpha=0.7, zorder=3)
    
    # Identity line
    lo = pivot[[args.method_x, args.method_y]].values.min()
    hi = pivot[[args.method_x, args.method_y]].values.max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="grey", lw=1.5, zorder=1)
    
    # Add labels with smart stacking to avoid overlap
    # Group by similar y-values (within 0.02 threshold)
    fontsize = 12
    label_height_pts = fontsize * 2.0  # Approximate height in points with padding
    
    # Convert points to data coordinates for spacing
    # Get the data range
    y_range = hi - lo
    fig_height_inches = args.figsize[1]
    fig_height_pts = fig_height_inches * 72  # 72 points per inch
    
    # Calculate vertical spacing in data units
    # This ensures labels never overlap
    label_spacing_data = (y_range / fig_height_pts) * label_height_pts
    
    # Group points with similar y-values
    tolerance = 0.02  # Group points within 2% of y-range
    pivot_sorted = pivot.sort_values(args.method_y)
    
    for _, row in pivot_sorted.iterrows():
        x_val = row[args.method_x]
        y_val = row[args.method_y]
        label = row["label"]
        
        # Check for nearby labels already placed
        # Find how many labels are already at similar y-position
        nearby = pivot_sorted[
            (pivot_sorted[args.method_y] >= y_val - tolerance) & 
            (pivot_sorted[args.method_y] <= y_val + tolerance) &
            (pivot_sorted.index <= row.name)  # Only count already-processed points
        ]
        
        # Stack vertically based on how many nearby points exist
        stack_index = len(nearby) - 1
        y_offset = stack_index * label_spacing_data

        x_range = hi - lo
        x_offset = x_range * 0.02
        
        # Place label slightly to the right of point
        ax.text(
            x_val + x_offset, 
            y_val + y_offset,
            label,
            fontsize=fontsize,
            ha='left',
            va='center'
        )
    
    # Labels and formatting
    metric_label = args.metric.replace("disease_", "").replace("_", " ").title()
    ax.set_xlabel("Random Negatives", fontsize=14)
    ax.set_ylabel("Hard Negatives", fontsize=14)
    # ax.set_title(f"Top {args.k_value} Precision: Negative Comparison", fontsize=16)
    ax.set_title(f"Negative Data Comparison", fontsize=16)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3, zorder=0)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    
    # Save
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"identity_scatter_k{args.k_value}_{args.method_x}_vs_{args.method_y}"
    
    output_path = args.output_dir / f"{output_name}.pdf"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\\n✓ Saved plot to {output_path}")
    
    # Print summary statistics
    print(f"\\nSummary Statistics:")
    print(f"  Mean {metric_label} ({method_x_display}): {pivot[args.method_x].mean():.4f}")
    print(f"  Mean {metric_label} ({method_y_display}): {pivot[args.method_y].mean():.4f}")
    print(f"  Difference: {pivot[args.method_y].mean() - pivot[args.method_x].mean():.4f}")
    
    # Count wins
    wins_y = (pivot[args.method_y] > pivot[args.method_x]).sum()
    wins_x = (pivot[args.method_x] > pivot[args.method_y]).sum()
    ties = (pivot[args.method_x] == pivot[args.method_y]).sum()
    print(f"  {method_y_display} wins: {wins_y}/{len(pivot)}")
    print(f"  {method_x_display} wins: {wins_x}/{len(pivot)}")
    print(f"  Ties: {ties}/{len(pivot)}")


if __name__ == "__main__":
    main()