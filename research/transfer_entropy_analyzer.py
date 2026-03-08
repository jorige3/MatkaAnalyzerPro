"""
Transfer Entropy & Multi-Lag Information Analyzer
-------------------------------------------------
Hypothesis: Predictability is not limited to Lag-1 (yesterday). 
This module calculates Mutual Information across Lags 1-7 to find 
the 'Optimal Informational Lag' for prediction.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import norm
from research.tracker import log_experiment
from typing import Dict, List, Any

class TransferEntropyAnalyzer:
    """
    Analyzes information flow from historical lags to the current result.
    """
    def __init__(self, max_lag: int = 7):
        self.max_lag = max_lag

    def calculate_lagged_mi(self, series: pd.Series) -> Dict[int, float]:
        """
        Calculates Mutual Information between X_t and X_{t-lag}.
        """
        mi_values = {}
        df = pd.DataFrame({'target': series})
        
        for lag in range(1, self.max_lag + 1):
            df[f'lag_{lag}'] = series.shift(lag)
            
            # Drop NaNs created by shifting
            temp_df = df[['target', f'lag_{lag}']].dropna()
            
            # Mutual Information Score
            mi = mutual_info_score(temp_df['target'], temp_df[f'lag_{lag}'])
            mi_values[lag] = round(mi, 6)
            
        return mi_values

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Backtest: Use the 'Optimal Lag' (highest MI) instead of just Lag-1
        to build a Transition Matrix and predict.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        
        # 1. Find Optimal Lag from first half of data
        half_idx = len(jodis) // 2
        mi_scores = self.calculate_lagged_mi(pd.Series(jodis[:half_idx]))
        optimal_lag = max(mi_scores, key=mi_scores.get)
        
        print(f"Detected Optimal Informational Lag: {optimal_lag} (MI: {mi_scores[optimal_lag]})")

        # 2. Backtest Transition Matrix using Optimal Lag
        hits = 0
        total = 0
        num_classes = 100
        
        # Expanding window backtest on second half
        for i in range(half_idx, len(jodis) - 1):
            # Training window (all data up to i)
            hist = jodis[:i+1]
            
            # Build Matrix: X_{t-optimal_lag} -> X_t
            matrix = np.zeros((num_classes, num_classes))
            for t in range(optimal_lag, len(hist)):
                s_from, s_to = hist[t - optimal_lag], hist[t]
                matrix[s_from, s_to] += 1
            
            # Predict for i+1
            # Current source is at i+1 - optimal_lag
            source_idx = (i + 1) - optimal_lag
            if source_idx >= 0:
                s_from = jodis[source_idx]
                row = matrix[s_from]
                if row.sum() > 0:
                    probs = row / row.sum()
                    top_preds = np.argsort(probs)[-top_k:]
                    
                    if jodis[i+1] in top_preds:
                        hits += 1
                    total += 1
                    
        hit_rate = hits / total if total > 0 else 0
        baseline = top_k / 100.0
        
        # Significance
        std_error = np.sqrt((baseline * (1 - baseline)) / total) if total > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        results = {
            "optimal_lag": optimal_lag,
            "hit_rate": round(hit_rate, 4),
            "baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "mi_scores": mi_scores
        }
        
        log_experiment("transfer_entropy_lag_optimization", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    analyzer = TransferEntropyAnalyzer(max_lag=7)
    print("--- Transfer Entropy Analysis (Lag 1-7) ---")
    jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int)
    
    # Analyze Jodi, Tens, and Units separately
    for name, series in [("Full Jodi", jodis), 
                         ("Tens Digit", jodis // 10), 
                         ("Units Digit", jodis % 10)]:
        print(f"\nTarget: {name}")
        mi = analyzer.calculate_lagged_mi(series)
        for lag, score in mi.items():
            marker = "<- OPTIMAL" if score == max(mi.values()) else ""
            print(f"  Lag {lag}: {score} {marker}")

    print("\n--- Optimal Lag Backtest ---")
    bt = analyzer.quick_backtest(df)
    print(f"Hit Rate using Optimal Lag: {bt['hit_rate']:.2%} (Baseline: {bt['baseline']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
