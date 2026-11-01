"""
plot_tp_curves.py - Plot True Positive curves for disease gene prediction methods

Creates TP vs K curves comparing multiple methods across diseases.

Usage:
    python plot_tp_curves.py results_root
    python plot_tp_curves.py results_root --diseases C0006142
    python plot_tp_curves.py results_root --exclude-methods method1 method2
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Disease ID to name mapping
DISEASE_NAMES = {
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

# Color palette (same as notebook)
COLOR_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#6272FF", "#A65628", "#F781BF", "#999999", "#66C2A5",
    "#FC8D62", "#8DA0CB", "#FF3DB5",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TP curves comparing methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "results_root",
        type=Path,
        help="Root directory containing method subdirectories"
    )
    parser.add_argument(
        "--metric-file",
        type=str,
        default="top_k_eval_metrics_mean.csv",
        help="Name of metrics CSV file in each method directory"
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        help="Specific disease IDs to plot (default: all)"
    )
    parser.add_argument(
        "--method-names",
        type=Path,
        help="JSON file mapping directory names to display names"
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="+",
        help="Method directory names to exclude from plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=250,
        help="Maximum K value to plot"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures"
    )
    parser.add_argument(
        "--create-example-json",
        action="store_true",
        help="Create example method_names.json file and exit"
    )
    return parser.parse_args()


def create_example_json(output_path: Path = Path("method_names_example.json")):
    """Create an example JSON file for method name mapping."""
    import json
    
    example = {
        "disgeneformer_baseline": "DisGeneFormer",
        "disgeneformer_gda_edges": "DisGeneFormer+GDA",
        "guild_netcombo": "GUILD/NetCombo",
        "guild_netscore": "GUILD/NetScore",
        "guild_netshort": "GUILD/NetShort",
        "mcl_clustering": "MCL",
        "diamond": "DIAMOnD",
        "random_walk": "Random Walk",
        "pagerank": "PageRank"
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"✓ Created example JSON file: {output_path}")
    print("\nExample contents:")
    print(json.dumps(example, indent=2))
    print("\nEdit this file to customize method display names.")
    print("Usage: python plot_tp_curves.py results/ --method-names method_names_example.json")


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


def collect_method_data(results_root: Path, metric_file: str, 
                        method_names: Dict[str, str], max_k: int,
                        exclude_methods: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect TP data from all method directories."""
    
    exclude_methods = exclude_methods or []
    all_data = []
    
    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue
        
        # Skip excluded methods
        if method_dir.name in exclude_methods:
            print(f"Skipping excluded method: {method_dir.name}")
            continue
        
        metric_path = find_metric_file(method_dir, metric_file)
        if metric_path is None:
            print(f"Warning: No metrics file found for {method_dir.name}, skipping")
            continue
        
        # Load metrics with error handling
        try:
            df = pd.read_csv(metric_path)
            
            # Check if DataFrame is empty
            if df.empty:
                print(f"Warning: Empty file {metric_path}, skipping {method_dir.name}")
                continue
                
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty or invalid file {metric_path}, skipping {method_dir.name}")
            continue
        except Exception as e:
            print(f"Warning: Error loading {metric_path}: {e}, skipping {method_dir.name}")
            continue
        
        # Filter to max_k
        df = df[df['K'] <= max_k].copy()
        
        # Add method name
        method_name = method_names.get(method_dir.name, method_dir.name)
        df['method'] = method_name

        df['disease_tp'] = df['disease_tp'].round().astype(int)

        all_data.append(df[['disease_id', 'K', 'disease_tp', 'method']])
    
    if not all_data:
        raise ValueError(f"No valid method data found in {results_root}")
    
    combined = pd.concat(all_data, ignore_index=True).sort_values(['disease_id', 'method', 'K'])
    print(f"Loaded data for {combined['method'].nunique()} methods")
    print(f"Methods: {sorted(combined['method'].unique())}")
    
    return combined


def main():
    args = parse_args()
    
    # Create example JSON and exit if requested
    if args.create_example_json:
        create_example_json()
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load method name mappings
    method_names = load_method_names(args.method_names)
    
    # Collect data from all methods
    print(f"\nCollecting data from {args.results_root}...")
    cum_tp_df = collect_method_data(
        args.results_root, 
        args.metric_file, 
        method_names, 
        args.max_k,
        exclude_methods=args.exclude_methods
    )
    
    # Determine which diseases to plot
    if args.diseases:
        disease_ids = args.diseases
    else:
        disease_ids = sorted(cum_tp_df['disease_id'].unique())
    
    # Create color mapping
    method_list = sorted(cum_tp_df['method'].unique())
    color_of = {m: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, m in enumerate(method_list)}
    
    print(f"\nPlotting {len(disease_ids)} diseases...")
    
    # Plot each disease
    for disease_id in disease_ids:
        grp = cum_tp_df[cum_tp_df['disease_id'] == disease_id]
        
        if grp.empty:
            print(f"Warning: No data for disease {disease_id}")
            continue
        
        disease_name = DISEASE_NAMES.get(disease_id, disease_id)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot each method
        for method, g_m in grp.groupby('method'):
            ax.plot(
                g_m['K'], g_m['disease_tp'],
                lw=1.6, color=color_of[method], label=method
            )
        
        ax.set_title(f"{disease_name} ({disease_id})")
        ax.set_xlabel("K (top-ranked genes list)")
        ax.set_ylabel("True Positives")
        ax.grid(alpha=0.3)

        # Ensure small values of K are still integer on the axis 
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.legend(
            title="Method", fontsize="small",
            loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0
        )
        
        plt.tight_layout()
        
        # Save
        disease_name_clean = disease_name.replace(' ', '_')
        output_path = args.output_dir / f"tp_curve_{disease_id}_{disease_name_clean}.pdf"
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {output_path.name}")
    
    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()