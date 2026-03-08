"""
Synthetic Random Tester
-----------------------
A "Zero-Signal" sanity check. Feeds pure random data to engines. 
If they detect a "signal", we have a false positive in our methodology.
"""
import pandas as pd
import numpy as np
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from research.tracker import log_experiment

def generate_random_data(days: int = 1200) -> pd.DataFrame:
    dates = pd.date_range(end='2026-03-08', periods=days)
    jodis = np.random.randint(0, 100, size=days)
    return pd.DataFrame({
        'Date': dates,
        'Jodi': [str(j).zfill(2) for j in jodis]
    })

def run_synthetic_test():
    print("--- Running Synthetic Random Baseline Test ---")
    df = generate_random_data()
    
    # 1. Frequency Engine
    fe = FrequencyEngine()
    freq = fe.run(df)
    # Check max frequency in random data
    max_freq = max(freq.values()) if freq else 0
    
    # 2. Transition Matrix
    te = TransitionEngine()
    te_results = te.quick_backtest(df, top_k=10)
    
    # 3. Bayesian
    be = BayesianEngine()
    be_results = be.quick_backtest(df, top_k=10)

    metrics = {
        "max_freq_in_random": max_freq,
        "markov_hit_rate": te_results['accuracy'],
        "markov_edge": te_results['edge'],
        "bayesian_hit_rate": be_results['accuracy'],
        "bayesian_edge": be_results['edge'],
        "note": "Baseline for all edges in pure random data should be near 0."
    }
    
    log_experiment("synthetic_random_test", metrics)
    print("\nMetrics on PURE RANDOM data:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_synthetic_test()
