"""
Oracle Stress Test (Sequence Randomization)
-------------------------------------------
Generates randomized histories by shuffling the Jodi sequence and runs 
the full MI-Adaptive backtest on each. This determines if the 16.11% 
hit rate is statistically distinguishable from overfitting to noise.
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from engines.mi_adaptive_ensemble import MIAdaptiveEnsemble, run_backtest_window
from research.tracker import log_experiment
from tqdm import tqdm
from typing import Dict, List, Any

def run_oracle_test(df: pd.DataFrame, real_hit_rate: float, iterations: int = 100):
    """
    Runs the stress test by shuffling the Jodi column multiple times.
    """
    random_hit_rates = []
    
    print(f"Running Oracle Stress Test ({iterations} iterations)...")
    
    # We create a base copy
    df_test = df.copy()
    
    # Using a 150-day window for the backtest consistency
    window_days = 150
    
    for _ in tqdm(range(iterations), desc="Simulating Random Histories"):
        # Shuffle Jodis to destroy any real time-series signal
        shuffled_jodis = np.random.permutation(df_test['Jodi'].values)
        df_shuffled = df_test.copy()
        df_shuffled['Jodi'] = shuffled_jodis
        
        # Run the actual backtest logic on this fake history
        res = run_backtest_window(df_shuffled, window_days=window_days, top_k=10)
        if res:
            hits, total = res
            random_hit_rates.append(hits / total)

    # Statistical Analysis
    random_hit_rates = np.array(random_hit_rates)
    mean_hr = np.mean(random_hit_rates)
    std_hr = np.std(random_hit_rates)
    max_hr = np.max(random_hit_rates)
    
    p50 = np.percentile(random_hit_rates, 50)
    p90 = np.percentile(random_hit_rates, 90)
    p95 = np.percentile(random_hit_rates, 95)
    p99 = np.percentile(random_hit_rates, 99)
    
    percentile_rank = percentileofscore(random_hit_rates, real_hit_rate)

    print("\n## ORACLE STRESS TEST RESULTS")
    print(f"Random Runs: {iterations}")
    print(f"Mean Hit Rate: {mean_hr:.2%}")
    print(f"Std Dev: {std_hr:.4f}")
    print(f"Max Random Hit Rate: {max_hr:.2%}")
    print("\nPercentiles:")
    print(f"50th: {p50:.2%}")
    print(f"90th: {p90:.2%}")
    print(f"95th: {p95:.2%}")
    print(f"99th: {p99:.2%}")
    
    print("\n## REAL MODEL RESULT")
    print(f"Hit Rate: {real_hit_rate:.2%}")
    print(f"Percentile Rank: {percentile_rank:.2f}")

    # Interpretation
    conclusion = ""
    if percentile_rank >= 99:
        conclusion = "STRONG SIGNAL: Model significantly outperforms random permutations."
    elif percentile_rank >= 90:
        conclusion = "WEAK SIGNAL: Model shows edge but requires more validation."
    else:
        conclusion = "OVERFITTING: Model performance is indistinguishable from random noise optimization."

    print(f"\nSTATISTICAL CONCLUSION: {conclusion}")

    log_experiment("oracle_stress_test", {
        "iterations": iterations,
        "mean_hr": mean_hr,
        "max_hr": max_hr,
        "p99": p99,
        "real_hit_rate": real_hit_rate,
        "percentile_rank": percentile_rank,
        "conclusion": conclusion
    })

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    # Real hit rate achieved in previous experiment
    REAL_HR = 0.1611
    
    # Use 20 iterations for research speed; 100+ recommended for final audit
    run_oracle_test(df, real_hit_rate=REAL_HR, iterations=20)
