#!/bin/bash
#SBATCH --job-name=filter_humannet
#SBATCH --partition=bidlc2_cpu-cascadelake
#SBATCH --account=bi-dlc2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/filter_humannet_%j.out
#SBATCH --error=logs/filter_humannet_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DisGeneFormer_env

# Configuration
HUMANNET_FILE="data/gene_net/raw/HumanNet-XC-V3.tsv"
DISGENET_ASSOC_FILE="data/test/raw/all_disgenet_associations.tsv"
OUTPUT_DIR="data/gene_net/raw"

# List of diseases to process
DISEASES=(
    "C0006142"
    "C0009402"
    "C0023893"
    "C0036341"
    "C0376358"
    "C0001973"
    "C0011581"
    "C0860207"
    "C0005586"
    "C3714756"
)

echo "======================================================================="
echo "Filtering HumanNet for individual diseases"
echo "======================================================================="
echo "Input HumanNet: $HUMANNET_FILE"
echo "Total diseases: ${#DISEASES[@]}"
echo ""

# Process each disease
for disease_id in "${DISEASES[@]}"; do
    output_file="${OUTPUT_DIR}/HumanNet-XC-V3_filtered_${disease_id}.tsv"
    
    echo "-----------------------------------------------------------------------"
    echo "Processing disease: $disease_id"
    echo "Output: $output_file"
    echo "-----------------------------------------------------------------------"
    
    python scripts/filter_humannet.py \
        "$HUMANNET_FILE" \
        "$DISGENET_ASSOC_FILE" \
        "$output_file" \
        --diseases "$disease_id"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully filtered for $disease_id"
    else
        echo "✗ Failed to filter for $disease_id"
    fi
    echo ""
done

echo "======================================================================="
echo "All diseases processed!"
echo "======================================================================="
echo "Output files in: $OUTPUT_DIR"
ls -lh "${OUTPUT_DIR}"/HumanNet-XC-V3_filtered_*.tsv