"""
Spectral Analyzer Engine
------------------------
Hypothesis: The Jodi sequence contains hidden periodic cycles.
This module uses Lomb-Scargle Periodograms to identify dominant frequencies
and tests if they can predict future draws.
"""
import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from scipy.stats import norm
from typing import Dict, Any

class SpectralAnalyzer:
    """
    Analyzes periodicities in the Jodi time series using spectral density.
    """
    def __init__(self, num_classes: int = 100):
        self.num_classes = num_classes

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().values
        t = np.arange(len(jodis))
        
        # We look for periods between 2 and 60 days
        periods = np.linspace(2, 60, 500)
        freqs = 2 * np.pi / periods
        
        # Center the data
        y = jodis - np.mean(jodis)
        
        # Compute Lomb-Scargle Periodogram
        pgram = lombscargle(t, y, freqs, precenter=True)
        
        # Find dominant periods
        top_indices = np.argsort(pgram)[-5:][::-1]
        top_periods = periods[top_indices]
        top_powers = pgram[top_indices]
        
        return {
            "top_periods": [round(p, 2) for p in top_periods],
            "top_powers": [round(p, 2) for p in top_powers],
            "max_power": round(np.max(pgram), 2)
        }

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Walk-forward backtest.
        Hypothesis: If period P is dominant, X_t is likely to be near X_{t-P}.
        We predict the Jodis that appeared exactly P days ago for the top K periods.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Minimum history to find a 60-day cycle
        min_history = 150
        
        for i in range(min_history, len(jodis) - 1):
            window = jodis[:i+1]
            t_win = np.arange(len(window))
            y_win = np.array(window) - np.mean(window)
            
            periods = np.linspace(2, 60, 200)
            freqs = 2 * np.pi / periods
            pgram = lombscargle(t_win, y_win, freqs, precenter=True)
            
            # Get top K periods
            best_periods = periods[np.argsort(pgram)[-top_k:]]
            
            # Predictions: Jodis at t - Period
            current_idx = i
            predictions = []
            for p in best_periods:
                lookback_idx = int(current_idx - round(p))
                if 0 <= lookback_idx < len(window):
                    predictions.append(window[lookback_idx])
            
            actual_next = jodis[i+1]
            if actual_next in predictions:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        expected_prob = top_k / 100.0
        
        std_error = np.sqrt((expected_prob * (1 - expected_prob)) / total)
        z_score = (hit_rate - expected_prob) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        return {
            "hit_rate": round(hit_rate, 4),
            "expected_prob": expected_prob,
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_samples": total,
            "hits": hits
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    analyzer = SpectralAnalyzer()
    print("--- Spectral Analysis (Cycle Detection) ---")
    stats = analyzer.run_analysis(df)
    print(f"Top 5 Dominant Periods (Days): {stats['top_periods']}")
    print(f"Peak Power: {stats['max_power']}")
    
    print("\n--- Spectral Cycle Backtest (Top-5 Periods) ---")
    bt = analyzer.quick_backtest(df, top_k=5)
    print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['expected_prob']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
