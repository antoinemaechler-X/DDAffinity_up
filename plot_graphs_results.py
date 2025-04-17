import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# Load the CSV data
file_path = "M1340_results"
df = pd.read_csv(f"{file_path}.csv")

# Graph 1: ddG vs ddG_pred with y = x line
plt.figure()
plt.scatter(df['ddG'], df['ddG_pred'], label='Predictions')
plt.plot(df['ddG'], df['ddG'], color='red', label='y = x')
plt.xlabel('Experimental ddG')
plt.ylabel('Predicted ddG')
plt.title('ddG vs Predicted ddG')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'ddG_vs_pred_{file_path}.png')

# Graph 2: double bar plot of num_muts vs avg Pearson and Spearman
grouped = df.groupby('num_muts')
results = []

for num, group in grouped:
    pearson_corr, _ = pearsonr(group['ddG'], group['ddG_pred'])
    spearman_corr, _ = spearmanr(group['ddG'], group['ddG_pred'])
    results.append({'num_muts': num, 'pearson': pearson_corr, 'spearman': spearman_corr})

results_df = pd.DataFrame(results)

# Plot double bar graph
x = results_df['num_muts']
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, results_df['pearson'], width, label='Pearson')
ax.bar(x + width/2, results_df['spearman'], width, label='Spearman')
ax.set_xlabel('Number of Mutations')
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Average Pearson and Spearman Correlation by Mutation Count')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'correlation_by_mut_count_{file_path}.png')

# ===== Extra Graphs =====

# Add error column
df['error'] = df['ddG_pred'] - df['ddG']

# Graph 3: Histogram of prediction error
plt.figure()
plt.hist(df['error'], bins=10, edgecolor='black')
plt.xlabel('Prediction Error (ddG_pred - ddG)')
plt.ylabel('Count')
plt.title('Distribution of Prediction Error')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'error_histogram_{file_path}.png')

# Graph 4: Boxplot of prediction error by num_muts
plt.figure()
sns.boxplot(x='num_muts', y='error', data=df)
plt.xlabel('Number of Mutations')
plt.ylabel('Prediction Error')
plt.title('Prediction Error by Mutation Count')
plt.tight_layout()
plt.savefig(f'error_boxplot_by_mut_count_{file_path}.png')

# Graph 5: Scatter of error vs ddG
plt.figure()
plt.scatter(df['ddG'], df['error'])
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Experimental ddG')
plt.ylabel('Prediction Error (ddG_pred - ddG)')
plt.title('Prediction Error vs Experimental ddG')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'error_vs_ddG_{file_path}.png')
