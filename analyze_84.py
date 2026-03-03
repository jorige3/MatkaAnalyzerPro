"""
Matka Analyzer Pro - One-off Analysis Script
--------------------------------------------
Performs a deep-dive analysis for Jodi '84'.
"""

from analyzer import load_data, analyze_jodi
from plotting import generate_report

def run_specific_analysis(jodi="84"):
    print(f"--- Running Deep-Dive Analysis: {jodi} ---")
    
    df = load_data()
    results = analyze_jodi(df, jodi)
    summary_path, plot_path = generate_report(results)
    
    print(f"Analysis Complete for {jodi}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")
    print(f"Z-Score: {results['z_score']:.4f}")

if __name__ == "__main__":
    run_specific_analysis()
