import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(csv_path, by_method=False):
    # Read the results
    df = pd.read_csv(csv_path)
    print(f"\nInitial results dataframe shape: {df.shape}")
    
    # If by_method is True, load SKEMPI2 data and merge with results
    if by_method:
        # Read SKEMPI2 data
        skempi_df = pd.read_csv("data/SKEMPI2/full_SKEMPI2.csv", sep=';')
        print(f"\nSKEMPI2 dataframe shape: {skempi_df.shape}")
        
        # Extract complex and mutation info from SKEMPI2
        skempi_df['complex'] = skempi_df['#Pdb'].str.split('_').str[0]
        
        # Convert mutation strings to match results format
        def convert_mutation(mut):
            if pd.isna(mut):
                return mut
            # Handle multiple mutations
            if ',' in mut:
                return ','.join(convert_mutation(m) for m in mut.split(','))
            # Extract components
            chain = mut[0]
            orig_res = mut[1]
            pos = mut[2:-1]
            new_res = mut[-1]
            # Convert to results format (e.g., LB38G)
            return f"{chain}{orig_res}{pos}{new_res}"
        
        skempi_df['mutstr'] = skempi_df['Mutation(s)_cleaned'].apply(convert_mutation)
        
        # Merge with results
        df = pd.merge(df, skempi_df[['complex', 'mutstr', 'Method']], 
                     on=['complex', 'mutstr'], how='left')
        print(f"\nShape after merge: {df.shape}")
        
        # For overall correlations, use only one entry per mutation
        df_unique = df.drop_duplicates(['complex', 'mutstr'])
        print(f"Number of unique mutations: {len(df_unique)}")
        
        # Calculate overall correlations using unique mutations
        pearson_corr = df_unique['ddG'].corr(df_unique['ddG_pred'], method='pearson')
        spearman_corr = df_unique['ddG'].corr(df_unique['ddG_pred'], method='spearman')
        rmse = np.sqrt(np.mean((df_unique['ddG'] - df_unique['ddG_pred'])**2))
        mae = np.mean(np.abs(df_unique['ddG'] - df_unique['ddG_pred']))
        
        # Print overall metrics
        print(f"\nAnalysis of {csv_path}")
        print("-" * 50)
        print(f"Number of unique mutations: {len(df_unique)}")
        print(f"Overall Pearson correlation: {pearson_corr:.4f}")
        print(f"Overall Spearman correlation: {spearman_corr:.4f}")
        print(f"Overall RMSE: {rmse:.4f}")
        print(f"Overall MAE: {mae:.4f}")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot each method in a different color
        methods = df['Method'].dropna().unique()
        if len(methods) > 0:
            colors = sns.color_palette("husl", len(methods))
            method_colors = dict(zip(methods, colors))
            
            # Store method statistics for bar plot
            method_stats = []
            
            for method in methods:
                # For each method, use all entries (including duplicates) for method-specific correlations
                subset = df[df['Method'] == method]
                method_pearson = subset['ddG'].corr(subset['ddG_pred'], method='pearson')
                method_spearman = subset['ddG'].corr(subset['ddG_pred'], method='spearman')
                
                # Store stats for bar plot
                method_stats.append({
                    'Method': method,
                    'Count': len(subset),
                    'Percentage': len(subset) / len(df) * 100,
                    'Pearson': method_pearson
                })
                
                plt.scatter(subset['ddG'], subset['ddG_pred'], 
                           alpha=0.5, 
                           label=f'{method} (n={len(subset)}, r={method_pearson:.3f})',
                           color=method_colors[method])
                
                # Print method-specific correlations
                print(f"\n{method} (n={len(subset)}):")
                print(f"  Pearson: {method_pearson:.4f}")
                print(f"  Spearman: {method_spearman:.4f}")
            
            # Print correlations for methods with n≥200
            print("\nMethods with n≥200:")
            print("-" * 50)
            for method in methods:
                subset = df[df['Method'] == method]
                if len(subset) >= 200:
                    method_pearson = subset['ddG'].corr(subset['ddG_pred'], method='pearson')
                    method_spearman = subset['ddG'].corr(subset['ddG_pred'], method='spearman')
                    print(f"\n{method} (n={len(subset)}):")
                    print(f"  Pearson: {method_pearson:.4f}")
                    print(f"  Spearman: {method_spearman:.4f}")
            
            # Create bar plot
            plt.figure(figsize=(12, 6))
            method_df = pd.DataFrame(method_stats)
            method_df = method_df.sort_values('Count', ascending=False)
            
            # Create bar plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot percentage bars
            bars = ax1.bar(method_df['Method'], method_df['Percentage'], 
                          color='skyblue', alpha=0.6, label='Percentage')
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Percentage of Data (%)', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            
            # Add percentage values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Create second y-axis for Pearson correlation
            ax2 = ax1.twinx()
            ax2.plot(method_df['Method'], method_df['Pearson'], 'ro-', label='Pearson')
            ax2.set_ylabel('Pearson Correlation', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add Pearson values on top of points
            for i, pearson in enumerate(method_df['Pearson']):
                ax2.text(i, pearson, f'{pearson:.3f}', ha='center', va='bottom')
            
            plt.title('Method Distribution and Performance')
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save bar plot
            bar_plot_path = csv_path.replace('.csv', '_method_stats.png')
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            print(f"\nBar plot saved to: {bar_plot_path}")
            
            # Return to scatter plot
            plt.figure(figsize=(12, 8))
            
            # Replot the scatter points
            for method in methods:
                subset = df[df['Method'] == method]
                method_pearson = subset['ddG'].corr(subset['ddG_pred'], method='pearson')
                plt.scatter(subset['ddG'], subset['ddG_pred'], 
                           alpha=0.5, 
                           label=f'{method} (n={len(subset)}, r={method_pearson:.3f})',
                           color=method_colors[method])
        else:
            print("\nWarning: No methods found after merge. Falling back to single-color plot.")
            sns.scatterplot(data=df, x='ddG', y='ddG_pred', alpha=0.5)
        
        # Add perfect prediction line and correlation line only once
        min_val = min(df['ddG'].min(), df['ddG_pred'].min())
        max_val = max(df['ddG'].max(), df['ddG_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        z = np.polyfit(df_unique['ddG'], df_unique['ddG_pred'], 1)
        p = np.poly1d(z)
        plt.plot(df_unique['ddG'], p(df_unique['ddG']), "b--", label=f'Overall fit (r={pearson_corr:.3f})')

        plt.title(f'Predicted vs True ΔΔG\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
        plt.xlabel('True ΔΔG (kcal/mol)')
        plt.ylabel('Predicted ΔΔG (kcal/mol)')
        # Deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save scatter plot
        plot_path = csv_path.replace('.csv', '_correlation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nScatter plot saved to: {plot_path}")

        # --- Compute global Pearson/Spearman for methods with n >= 200 ---
        # Find methods with at least 200 entries
        method_counts = df['Method'].value_counts()
        methods_200 = method_counts[method_counts >= 200].index.tolist()
        df_200 = df[df['Method'].isin(methods_200)]
        df_200_unique = df_200.drop_duplicates(['complex', 'mutstr'])
        pearson_200 = df_200_unique['ddG'].corr(df_200_unique['ddG_pred'], method='pearson')
        spearman_200 = df_200_unique['ddG'].corr(df_200_unique['ddG_pred'], method='spearman')
        print("\nGlobal correlations for methods with n >= 200:")
        print(f"Pearson: {pearson_200:.4f}")
        print(f"Spearman: {spearman_200:.4f}")
    else:
        # Original single-color plot
        sns.scatterplot(data=df, x='ddG', y='ddG_pred', alpha=0.5)
        # Add perfect prediction line and fit line
        min_val = min(df['ddG'].min(), df['ddG_pred'].min())
        max_val = max(df['ddG'].max(), df['ddG_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        z = np.polyfit(df['ddG'], df['ddG_pred'], 1)
        p = np.poly1d(z)
        plt.plot(df['ddG'], p(df['ddG']), "b--", label=f'Fit (r={pearson_corr:.3f})')
        plt.title(f'Predicted vs True ΔΔG\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
        plt.xlabel('True ΔΔG (kcal/mol)')
        plt.ylabel('Predicted ΔΔG (kcal/mol)')
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_path = csv_path.replace('.csv', '_correlation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nScatter plot saved to: {plot_path}")

if __name__ == "__main__":
    # Analyze the results file
    csv_path = "logs_skempi/[10-fold-16]_05_27_16_17_35/checkpoints/results_75.csv"
    analyze_results(csv_path, by_method=True)  # Set to True to analyze by method 