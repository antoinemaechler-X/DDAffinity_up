import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from flexibility_score import flexibility_score
from tqdm import tqdm
import argparse
import os
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load mapping from SKEMPI2.csv
skempi_df = pd.read_csv("data/SKEMPI2/SKEMPI2.csv")
pdb_to_number = dict(zip(skempi_df['#Pdb_origin'], skempi_df['#Pdb'].str.split('_').str[0]))

def get_pdb_path(complex_id: str) -> str:
    """Convert complex ID to PDB file path."""
    # Extract PDB ID from complex ID (e.g., "1SBB" -> "1SBB")
    pdb_id = complex_id.split('_')[0] if '_' in complex_id else complex_id
    
    # Get the number from the mapping
    if pdb_id not in pdb_to_number:
        raise ValueError(f"No mapping found for PDB ID {pdb_id}")
    number = pdb_to_number[pdb_id]
    
    return f"data/SKEMPI2/SKEMPI2_cache/wildtype/{number}_{pdb_id}.pdb"

def categorize_flexibility(score: float) -> int:
    """Categorize flexibility score into 12 bins with custom splits."""
    if score > 0.65:
        return 11
    elif score > 0.55:
        return 10
    elif score > 0.45:
        return 9
    elif score > 0.35:
        return 8
    elif score > 0.25:
        return 7
    elif score > 0.20:
        return 6
    elif score > 0.15:
        return 5
    elif score > 0.125:
        return 4
    elif score > 0.10:
        return 3
    elif score > 0.075:
        return 2
    elif score >= 0.05:
        return 1
    else:
        return 0

def get_category_label(cat: int) -> str:
    """Get label for a category number."""
    if cat == 11:
        return ">0.65"
    elif cat == 10:
        return "0.55-0.65"
    elif cat == 9:
        return "0.45-0.55"
    elif cat == 8:
        return "0.35-0.45"
    elif cat == 7:
        return "0.25-0.35"
    elif cat == 6:
        return "0.20-0.25"
    elif cat == 5:
        return "0.15-0.20"
    elif cat == 4:
        return "0.125-0.15"
    elif cat == 3:
        return "0.10-0.125"
    elif cat == 2:
        return "0.075-0.10"
    elif cat == 1:
        return "0.05-0.075"
    else:
        return "0.00-0.05"

def calculate_metrics(true_values: list[float], pred_values: list[float]) -> tuple[float, float]:
    """Calculate Pearson and Spearman correlations."""
    if len(true_values) < 2:
        return np.nan, np.nan
    
    pearson = abs(stats.pearsonr(true_values, pred_values)[0])  # Take absolute value
    spearman = abs(stats.spearmanr(true_values, pred_values)[0])  # Take absolute value
    return pearson, spearman

def calculate_cumulative_correlations(df: pd.DataFrame, step: float = 0.01) -> tuple[list[float], list[float], list[float]]:
    """Calculate cumulative correlations for increasing flexibility thresholds."""
    thresholds = np.arange(0.05, 1.0, step)
    pearsons = []
    spearmans = []
    counts = []
    
    for threshold in thresholds:
        subset = df[df['flexibility'] <= threshold]
        if len(subset) >= 2:  # Need at least 2 points for correlation
            pearson, spearman = calculate_metrics(
                subset['ddG'].tolist(),
                subset['ddG_pred'].tolist()
            )
            pearsons.append(pearson)
            spearmans.append(spearman)
            counts.append(len(subset))
        else:
            pearsons.append(np.nan)
            spearmans.append(np.nan)
            counts.append(0)
    
    return thresholds, pearsons, spearmans

def analyze_flexibility_results(results_file: str, n_rows: int = -1, verbose: bool = False):
    # Read results
    df = pd.read_csv(results_file)
    
    # Filter single mutations
    df = df[df['num_muts'] == 1].copy()
    
    # Limit number of rows if specified
    if n_rows > 0:
        df = df.head(n_rows)
    
    # Calculate flexibility scores
    flex_scores = []
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating flexibility scores"):
        try:
            pdb_path = get_pdb_path(row['complex'])
            if not os.path.exists(pdb_path):
                if verbose:
                    print(f"PDB file not found: {pdb_path}")
                flex_scores.append(np.nan)
                errors += 1
                continue
            flex_score = flexibility_score(pdb_path, row['mutstr'])
            flex_scores.append(flex_score)
        except Exception as e:
            if verbose:
                print(f"Error processing {row['complex']} {row['mutstr']}: {e}")
            flex_scores.append(np.nan)
            errors += 1
    
    df['flexibility'] = flex_scores
    
    # Print error summary
    print(f"\nProcessed {len(df)} mutations")
    print(f"Failed to calculate flexibility scores for {errors} mutations ({errors/len(df)*100:.1f}%)")
    
    # Remove rows with NaN flexibility scores and scores > 1.0
    df = df.dropna(subset=['flexibility'])
    df = df[df['flexibility'] <= 1.0]
    
    if len(df) == 0:
        raise ValueError("No valid flexibility scores were calculated.")
    
    # Categorize mutations
    df['category'] = df['flexibility'].apply(categorize_flexibility)
    
    # Print flexibility score distribution
    print("\nFlexibility Score Distribution:")
    print(f"Min: {df['flexibility'].min():.3f}")
    print(f"Max: {df['flexibility'].max():.3f}")
    print(f"Mean: {df['flexibility'].mean():.3f}")
    print(f"Median: {df['flexibility'].median():.3f}")
    
    # Count mutations in each category
    print("\nMutations per category:")
    for cat in range(12):  # Updated to 12 categories
        count = len(df[df['category'] == cat])
        if count > 0:
            print(f"{get_category_label(cat)}: {count} mutations")
    
    # Create three plots
    plt.figure(figsize=(20, 6))
    
    # Plot 1: Scatter plot of error vs flexibility
    plt.subplot(1, 3, 1)
    # Calculate absolute error
    df['error'] = abs(df['ddG'] - df['ddG_pred'])
    plt.scatter(df['flexibility'], df['error'], alpha=0.6, color='#FF6B6B')
    plt.xlabel('Flexibility Score')
    plt.ylabel('Absolute Error (|ΔΔG - ΔΔG_pred|)')
    plt.title('Prediction Error by Flexibility')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add a trend line
    z = np.polyfit(df['flexibility'], df['error'], 1)
    p = np.poly1d(z)
    plt.plot(df['flexibility'].sort_values(), p(df['flexibility'].sort_values()), 
             color='#4A90E2', linestyle='--', label=f'Trend (slope: {z[0]:.2f})')
    plt.legend()
    
    # Plot 2: Bar plot of correlations by category
    plt.subplot(1, 3, 2)
    
    # Calculate metrics for each category
    categories = []
    pearsons = []
    spearmans = []
    counts = []
    
    for cat in range(12):  # Updated to 12 categories
        cat_data = df[df['category'] == cat]
        if len(cat_data) > 0:
            pearson, spearman = calculate_metrics(
                cat_data['ddG'].tolist(),
                cat_data['ddG_pred'].tolist()
            )
            # Only add categories with valid metrics
            if not np.isnan(pearson):
                categories.append(cat)
                pearsons.append(pearson)
                spearmans.append(spearman)
                counts.append(len(cat_data))
    
    if not categories:
        raise ValueError("No categories with enough mutations for statistical analysis.")
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, pearsons, width, label='Pearson', color='#FF6B6B')
    plt.bar(x + width/2, spearmans, width, label='Spearman', color='#4A90E2')
    
    plt.xlabel('Flexibility Score Category')
    plt.ylabel('Correlation Coefficient')
    plt.title('Model Performance by Flexibility Score Category')
    
    # Set x-axis labels
    plt.xticks(x, [f"{get_category_label(cat)}\n(n={c})" for cat, c in zip(categories, counts)], rotation=45)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 1)
    
    plt.legend()
    
    # Plot 3: Cumulative correlations
    plt.subplot(1, 3, 3)
    thresholds, cum_pearsons, cum_spearmans = calculate_cumulative_correlations(df)
    
    plt.plot(thresholds, cum_pearsons, label='Pearson', color='#FF6B6B')
    plt.plot(thresholds, cum_spearmans, label='Spearman', color='#4A90E2')
    
    plt.xlabel('Maximum Flexibility Score')
    plt.ylabel('Cumulative Correlation Coefficient')
    plt.title('Cumulative Model Performance by Flexibility Threshold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('flexibility_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total mutations analyzed: {len(df)}")
    print(f"Average absolute error: {df['error'].mean():.3f}")
    print(f"Error trend slope: {z[0]:.3f}")
    print("\nMetrics by category:")
    for cat, pearson, spearman, count in zip(categories, pearsons, spearmans, counts):
        print(f"\nCategory {get_category_label(cat)}:")
        print(f"Number of mutations: {count}")
        print(f"Pearson: {pearson:.3f}")
        print(f"Spearman: {spearman:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model performance by flexibility score')
    parser.add_argument('-n', type=int, default=-1,
                      help='Number of rows to process (-1 for all rows)')
    parser.add_argument('--results', type=str, 
                      default="logs_skempi/[10-fold-16]_05_27_16_17_35/checkpoints/results_75.csv",
                      help='Path to results CSV file')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Print detailed error messages')
    args = parser.parse_args()
    
    analyze_flexibility_results(args.results, args.n, args.verbose) 