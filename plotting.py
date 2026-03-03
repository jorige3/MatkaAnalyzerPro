"""
Plotting Helpers for Matka Analyzer Pro
---------------------------------------
Provides functions for generating visualizations and reports for deep-dive analysis.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict, Any

def generate_report(results: Dict[str, Any], report_dir: str = "reports/", plot_dir: str = "reports/plots/"):
    """
    Generates a text summary and a rolling frequency plot.
    """
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    jodi = results['jodi']
    
    # Save Summary Text
    summary_path = os.path.join(report_dir, f"{jodi}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Jodi {jodi} Analysis Report\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total Occurrences  : {results['count']}\n")
        f.write(f"Expected (Uniform) : {results['expected']:.2f}\n")
        f.write(f"Z-Score            : {results['z_score']:.4f}\n")
        f.write(f"Frequency Score    : {results['frequency_score']:.2f}/100\n")
        f.write(f"Last Occurrence    : {results['last_occurrence'].date() if results['last_occurrence'] else 'N/A'}\n")
        f.write(f"Days Since Last    : {results['days_since'] if results['days_since'] is not None else 'N/A'}\n")
        f.write(f"Rolling (30d) Freq : {results['current_rolling_freq']}\n")
        f.write(f"Dataset Size       : {results['total_records']}\n")

    # Generate Plot
    df_plot = results['rolling_data']
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot['Date'], df_plot['rolling_freq'], color='#1f77b4', label='30-Day Rolling Freq')
    
    # Add occurrence markers
    occurs = df_plot[df_plot['is_target'] == 1]
    plt.scatter(occurs['Date'], [0.05] * len(occurs), color='red', marker='|', alpha=0.6, label='Hits')
    
    plt.title(f"Jodi {jodi} Bias Analysis (Z-Score: {results['z_score']:.2f})")
    plt.xlabel("Timeline")
    plt.ylabel("Occurrences in Window")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(plot_dir, f"{jodi}_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    
    return summary_path, plot_path
