#!/usr/bin/env python3
"""
Script to plot cluster analysis results showing the relationship between
sequence identity threshold, number of clusters, and model performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_cluster_analysis_plot():
    """Create a dual-axis plot showing clusters vs performance metrics"""
    
    # Data
    thresholds = [40, 60, 80, 95, 99, 100]  # Sequence identity thresholds (%)
    num_clusters = [86, 109, 124, 155, 191, 404]  # Number of complex clusters (last value to be calculated)
    pcc_values = [0.456, 0.508, 0.433, 0.548, 0.539, 0.655]  # Pearson correlation coefficients
    spearman_values = [0.386, 0.407, 0.375, 0.418, 0.434, 0.513]  # Spearman correlation coefficients
    
    # Debug: Print all data
    print("DEBUG: All data points:")
    for i, (t, c, p, s) in enumerate(zip(thresholds, num_clusters, pcc_values, spearman_values)):
        print(f"  {i}: Threshold={t}%, Clusters={c}, PCC={p:.3f}, Spearman={s:.3f}")
    
    # Filter out None values for plotting
    valid_indices = [i for i, x in enumerate(num_clusters) if x is not None]
    thresholds_plot = [thresholds[i] for i in valid_indices]
    clusters_plot = [num_clusters[i] for i in valid_indices]
    pcc_plot = [pcc_values[i] for i in valid_indices]
    spearman_plot = [spearman_values[i] for i in valid_indices]
    
    # Debug: Print filtered data
    print(f"\nDEBUG: Valid indices: {valid_indices}")
    print("DEBUG: Filtered data:")
    for i, (t, c, p, s) in enumerate(zip(thresholds_plot, clusters_plot, pcc_plot, spearman_plot)):
        print(f"  {i}: Threshold={t}%, Clusters={c}, PCC={p:.3f}, Spearman={s:.3f}")
    
    # Use regularly spaced x positions for better visualization
    x_positions = list(range(1, len(thresholds_plot) + 1))
    print(f"DEBUG: X positions: {x_positions}")
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 9))  # Increased figure size
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot bars for number of clusters (left y-axis)
    bars = ax1.bar(x_positions, clusters_plot, alpha=0.7, color='skyblue', 
                   edgecolor='navy', linewidth=1.5, width=0.5)  # Reduced bar width
    
    # Add value labels on bars with offset to avoid overlap
    for i, (bar, value) in enumerate(zip(bars, clusters_plot)):
        height = bar.get_height()
        # Use normal height for all cluster count labels
        y_offset = 10
        ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot lines for correlation metrics (right y-axis)
    line1 = ax2.plot(x_positions, pcc_plot, 'o-', color='red', linewidth=3, 
                     markersize=10, label='Pearson Correlation', markerfacecolor='white', markeredgewidth=2)
    line2 = ax2.plot(x_positions, spearman_plot, 's-', color='green', linewidth=3, 
                     markersize=10, label='Spearman Correlation', markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels on correlation lines with better positioning
    for i, (pcc, spearman) in enumerate(zip(pcc_plot, spearman_plot)):
        # Position PCC labels below the line to avoid overlap with cluster counts
        ax2.annotate(f'{pcc:.3f}', (x_positions[i], pcc), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, fontweight='bold')
        # Position Spearman labels further below the line
        ax2.annotate(f'{spearman:.3f}', (x_positions[i], spearman), 
                    textcoords="offset points", xytext=(0,-30), ha='center', fontsize=9, fontweight='bold')
    
    # Customize the plot
    ax1.set_xlabel('Sequence Identity Threshold (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Complex Clusters', fontsize=14, fontweight='bold', color='navy')
    ax2.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold', color='darkgreen')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(thresholds_plot, fontsize=12)
    
    # Set title
    plt.title('Impact of Sequence Identity Threshold on Clustering and Model Performance', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Customize axes
    ax1.tick_params(axis='y', labelcolor='navy', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='darkgreen', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    # Set grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits with more space to separate overlapping values
    ax1.set_ylim(0, max(clusters_plot) * 1.25)  # Increased upper limit for clusters
    ax2.set_ylim(0, max(max(pcc_plot), max(spearman_plot)) * 1.25)  # Increased upper limit for correlations
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cluster_analysis_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_analysis_plot.pdf', bbox_inches='tight')
    
    print("Plots saved as 'cluster_analysis_plot.png' and 'cluster_analysis_plot.pdf'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Threshold (%)':<15} {'Clusters':<10} {'PCC':<10} {'Spearman':<10}")
    print("-" * 60)
    for i in valid_indices:
        print(f"{thresholds[i]:<15} {num_clusters[i]:<10} {pcc_values[i]:<10.3f} {spearman_values[i]:<10.3f}")
    
    # Find best performing threshold
    best_pcc_idx = np.argmax(pcc_plot)
    best_spearman_idx = np.argmax(spearman_plot)
    
    print("\n" + "="*60)
    print("BEST PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Best Pearson Correlation: {pcc_plot[best_pcc_idx]:.3f} at {thresholds_plot[best_pcc_idx]}% threshold")
    print(f"Best Spearman Correlation: {spearman_plot[best_spearman_idx]:.3f} at {thresholds_plot[best_spearman_idx]}% threshold")
    print(f"Number of clusters at best PCC: {clusters_plot[best_pcc_idx]}")
    print(f"Number of clusters at best Spearman: {clusters_plot[best_spearman_idx]}")


def create_additional_analysis():
    """Create additional analysis plots if needed"""
    
    # Data
    thresholds = [40, 60, 80, 95, 99, 100]
    num_clusters = [86, 109, 124, 155, 191, 404]  # Updated cluster counts
    pcc_values = [0.442, 0.508, 0.433, 0.548, 0.539, 0.655]
    spearman_values = [0.358, 0.407, 0.375, 0.418, 0.434, 0.513]
    
    # Filter valid data
    valid_data = [(t, c, p, s) for t, c, p, s in zip(thresholds, num_clusters, pcc_values, spearman_values) if c is not None]
    thresholds_valid, clusters_valid, pcc_valid, spearman_valid = zip(*valid_data)
    
    # Use regularly spaced x positions
    x_positions = list(range(1, len(thresholds_valid) + 1))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Clusters vs Threshold
    ax1.bar(x_positions, clusters_valid, color='lightblue', edgecolor='navy')
    ax1.set_xlabel('Sequence Identity Threshold (%)')
    ax1.set_ylabel('Number of Complex Clusters')
    ax1.set_title('Number of Clusters vs Threshold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(thresholds_valid)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PCC vs Threshold
    ax2.plot(x_positions, pcc_valid, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Identity Threshold (%)')
    ax2.set_ylabel('Pearson Correlation Coefficient')
    ax2.set_title('Pearson Correlation vs Threshold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(thresholds_valid)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spearman vs Threshold
    ax3.plot(x_positions, spearman_valid, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Sequence Identity Threshold (%)')
    ax3.set_ylabel('Spearman Correlation Coefficient')
    ax3.set_title('Spearman Correlation vs Threshold')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(thresholds_valid)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PCC vs Number of Clusters
    ax4.scatter(clusters_valid, pcc_valid, c='red', s=100, alpha=0.7, label='Pearson')
    ax4.scatter(clusters_valid, spearman_valid, c='green', s=100, alpha=0.7, label='Spearman')
    ax4.set_xlabel('Number of Complex Clusters')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_title('Correlation vs Number of Clusters')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Create the main plot
    create_cluster_analysis_plot()
    
    # Uncomment the line below to create additional analysis plots
    # create_additional_analysis() 