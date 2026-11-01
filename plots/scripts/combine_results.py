"""
create_results_table.py - Create combined results table from multiple methods

Extracts metrics from all method subdirectories and combines into a single table.

Usage:
    python create_results_table.py results_root                          # Default: disease_prec at K=5,20,50
    python create_results_table.py results_root --k-values 10 25 50 100  # Custom K values
    python create_results_table.py results_root --metric disease_recall  # Different metric
    python create_results_table.py results_root --method-names names.json # Custom method names
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Disease ID to name mapping
DISEASE_NAMES = {
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
    parser = argparse.ArgumentParser(
        description="Create combined results table from multiple methods",
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
        "--metric",
        type=str,
        default="disease_prec",
        choices=["disease_prec", "disease_recall", "disease_f1", "disease_tp"],
        help="Metric to extract"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 20, 50],
        help="K values to include in table"
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        help="Specific disease IDs to include (default: all)"
    )
    parser.add_argument(
        "--method-names",
        type=Path,
        help="JSON file mapping directory names to display names"
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="+",
        help="Method directory names to exclude"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_results_table.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose table (methods as rows, diseases/K as columns)"
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Number of decimal places to round to"
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


def collect_method_data(results_root: Path, metric_file: str, 
                        method_names: Dict[str, str],
                        exclude_methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Collect data from all method directories."""
    
    exclude_methods = exclude_methods or []
    method_data = {}
    
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
        
        # Load metrics
        try:
            df = pd.read_csv(metric_path)
            method_name = method_names.get(method_dir.name, method_dir.name)
            method_data[method_name] = df
            print(f"✓ Loaded: {method_name}")
        except Exception as e:
            print(f"Warning: Error loading {metric_path}: {e}, skipping")
            continue
    
    if not method_data:
        raise ValueError(f"No valid method data found in {results_root}")
    
    print(f"\nLoaded data for {len(method_data)} methods")
    return method_data


def create_combined_table(method_data: Dict[str, pd.DataFrame],
                         metric: str,
                         k_values: List[int],
                         diseases: Optional[List[str]] = None,
                         round_decimals: int = 4) -> pd.DataFrame:
    """Create combined results table."""
    
    # Determine diseases to include
    if diseases is None:
        # Get all diseases from first method
        first_method = list(method_data.values())[0]
        diseases = sorted(first_method['disease_id'].unique())
    
    # Create rows: (disease_id, disease_name, K) tuples
    rows = []
    for disease_id in diseases:
        disease_name = DISEASE_NAMES.get(disease_id, disease_id)
        for k in sorted(k_values):
            rows.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'K': k
            })
    
    # Create base DataFrame
    result_df = pd.DataFrame(rows)
    
    # Add column for each method
    for method_name, df in method_data.items():
        method_values = []
        
        for _, row in result_df.iterrows():
            disease_id = row['disease_id']
            k = row['K']
            
            # Find matching row in method data
            mask = (df['disease_id'] == disease_id) & (df['K'] == k)
            matching_rows = df[mask]
            
            if len(matching_rows) > 0:
                value = matching_rows.iloc[0][metric]
                method_values.append(round(value, round_decimals))
            else:
                method_values.append(None)
        
        result_df[method_name] = method_values
    
    return result_df


def main():
    args = parse_args()
    
    # Load method name mappings
    method_names = load_method_names(args.method_names)
    
    # Collect data from all methods
    print(f"Collecting data from {args.results_root}...\n")
    method_data = collect_method_data(
        args.results_root,
        args.metric_file,
        method_names,
        exclude_methods=args.exclude_methods
    )
    
    # Create combined table
    print(f"\nCreating combined table...")
    print(f"  Metric: {args.metric}")
    print(f"  K values: {args.k_values}")
    
    result_df = create_combined_table(
        method_data,
        args.metric,
        args.k_values,
        diseases=args.diseases,
        round_decimals=args.round_decimals
    )
    
    # Transpose if requested
    if args.transpose:
        # Set disease_id, disease_name, K as index before transposing
        result_df = result_df.set_index(['disease_id', 'disease_name', 'K']).T
        result_df.index.name = 'method'
        print("  Table transposed (methods as rows)")
    
    # Save to file
    output_path = args.results_root / args.output_file
    result_df.to_csv(output_path, index=args.transpose)
    
    print(f"\n✓ Saved combined table to {output_path}")
    print(f"  Shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
    
    # Print preview
    print("\nPreview:")
    print(result_df.head(10).to_string())
    
    # Print summary statistics
    if not args.transpose:
        print("\n" + "="*70)
        print("Summary Statistics (across all diseases and K values):")
        print("="*70)
        method_cols = [col for col in result_df.columns 
                      if col not in ['disease_id', 'disease_name', 'K']]
        for method in method_cols:
            values = result_df[method].dropna()
            if len(values) > 0:
                print(f"{method:30s} Mean: {values.mean():.4f}  "
                      f"Std: {values.std():.4f}  Min: {values.min():.4f}  "
                      f"Max: {values.max():.4f}")


if __name__ == "__main__":
    main()