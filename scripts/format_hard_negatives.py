#!/usr/bin/env python3
"""Strip a hard negatives TSV down to its first two columns (gene_id, disease_id), no header."""

import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, sep="\t")
df.iloc[:, :2].to_csv(path, sep="\t", index=False, header=False)
print(f"Formatted {path}: {len(df)} rows, 2 columns, no header")