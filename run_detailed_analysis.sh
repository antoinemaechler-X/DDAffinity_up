#!/bin/bash

# Shell script to run the detailed sequence analysis
# This script provides easy access to different configurations

echo "Detailed Sequence Analysis Script"
echo "================================="

# Check if Python script exists
if [ ! -f "test_sequence_grouped_detailed.py" ]; then
    echo "Error: test_sequence_grouped_detailed.py not found!"
    exit 1
fi

# Function to run analysis with given configuration
run_analysis() {
    local config=$1
    local description=$2
    echo ""
    echo "Running analysis with $description configuration..."
    echo "Configuration: $config"
    echo "----------------------------------------"
    
    python test_sequence_grouped_detailed.py --config $config
    
    if [ $? -eq 0 ]; then
        echo "✅ $description analysis completed successfully!"
    else
        echo "❌ $description analysis failed!"
    fi
}

# Main menu
echo ""
echo "Choose an option:"
echo "1) Run with medium configuration (70% sequence identity)"
echo "2) Run with strict configuration (90% sequence identity)"
echo "3) Run with lenient configuration (50% sequence identity)"
echo "4) Run with custom parameters"
echo "5) Run all configurations"
echo "6) Show help"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        run_analysis "medium" "Medium (70% sequence identity)"
        ;;
    2)
        run_analysis "strict" "Strict (90% sequence identity)"
        ;;
    3)
        run_analysis "lenient" "Lenient (50% sequence identity)"
        ;;
    4)
        echo ""
        echo "Custom configuration:"
        read -p "Enter CSV path (or press Enter for default): " csv_path
        read -p "Enter PDB WT directory (or press Enter for default): " pdb_wt_dir
        read -p "Enter PDB MT directory (or press Enter for default): " pdb_mt_dir
        read -p "Enter sequence identity threshold (0.0-1.0): " threshold
        read -p "Enter output CSV path (or press Enter for default): " output
        
        cmd="python test_sequence_grouped_detailed.py --config custom"
        if [ ! -z "$csv_path" ]; then
            cmd="$cmd --csv-path $csv_path"
        fi
        if [ ! -z "$pdb_wt_dir" ]; then
            cmd="$cmd --pdb-wt-dir $pdb_wt_dir"
        fi
        if [ ! -z "$pdb_mt_dir" ]; then
            cmd="$cmd --pdb-mt-dir $pdb_mt_dir"
        fi
        if [ ! -z "$threshold" ]; then
            cmd="$cmd --threshold $threshold"
        fi
        if [ ! -z "$output" ]; then
            cmd="$cmd --output $output"
        fi
        
        echo ""
        echo "Running custom analysis..."
        echo "Command: $cmd"
        echo "----------------------------------------"
        eval $cmd
        ;;
    5)
        echo ""
        echo "Running all configurations..."
        run_analysis "lenient" "Lenient (50% sequence identity)"
        run_analysis "medium" "Medium (70% sequence identity)"
        run_analysis "strict" "Strict (90% sequence identity)"
        echo ""
        echo "All analyses completed!"
        ;;
    6)
        echo ""
        echo "Help:"
        echo "This script runs detailed sequence analysis on protein-protein interaction data."
        echo ""
        echo "Configurations:"
        echo "  - Lenient (50%): Groups complexes with 50% or higher sequence identity"
        echo "  - Medium (70%): Groups complexes with 70% or higher sequence identity"
        echo "  - Strict (90%): Groups complexes with 90% or higher sequence identity"
        echo ""
        echo "Output files:"
        echo "  - detailed_sequence_analysis_*.csv: Main analysis results"
        echo "  - group_similarity_analysis.csv: Detailed similarity statistics"
        echo ""
        echo "The analysis will:"
        echo "  1. Load protein-protein interaction data"
        echo "  2. Extract protein sequences from PDB files"
        echo "  3. Compute pairwise sequence similarities"
        echo "  4. Group similar complexes together"
        echo "  5. Generate detailed CSV with complex names, mutations, sequences, and group assignments"
        echo ""
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1-6."
        exit 1
        ;;
esac

echo ""
echo "Analysis complete! Check the output CSV files for results." 